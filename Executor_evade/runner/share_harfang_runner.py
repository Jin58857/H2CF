import logging
import time
from typing import List
import matplotlib.pyplot as plt
import math

import numpy as np
import torch

from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner

logging.basicConfig(level=logging.INFO)

def _t2n(x):
    return x.detach().cpu().numpy()


class ShareHarfangRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay  # type: bool

        # policy & algorithm
        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        if self.use_selfplay:
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.share_obs_space,
                                             self.act_space)
        else:
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space,
                                             self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        if self.use_selfplay:

            from algorithms.utils.selfplay import get_algorithm
            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

            assert self.all_args.n_choose_opponents <= self.n_rollout_threads, \
                "Number of different opponents({}) must less than or equal to number of training threads({})!" \
                    .format(self.all_args.n_choose_opponents, self.n_rollout_threads)
            self.policy_pool = {'latest': self.all_args.init_elo}  # type: dict[str, float]
            self.opponent_policy = [
                Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
                for _ in range(self.all_args.n_choose_opponents)]
            self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            self.opponent_obs = np.zeros_like(self.buffer.obs[0])
            self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
            self.opponent_masks = np.ones_like(self.buffer.masks[0])

            if self.use_eval:
                self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space,
                                                   device=self.device)

            logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                         .format(self.all_args.selfplay_algorithm, self.all_args.n_choose_opponents))

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):


            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                # start_time = time.perf_counter()
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)
                # end_time = time.perf_counter()
                # elapsed_time = end_time - start_time
                # print(f"Program executed in {elapsed_time} seconds")

                data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                # print("执行步数：", infos)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.scenario_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            self.total_num_steps,
                            self.num_env_steps,
                            int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.buffer_size

                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                      np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        if self.use_selfplay:
            opponent_actions = np.zeros_like(actions)
            for policy_idx, policy in enumerate(self.opponent_policy):
                env_idx = self.opponent_env_split[policy_idx]
                opponent_action, opponent_rnn_states \
                    = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                 np.concatenate(self.opponent_rnn_states[env_idx]),
                                 np.concatenate(self.opponent_masks[env_idx]))
                opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
                self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
            actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data: List[np.ndarray]):
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones = dones.squeeze(axis=-1)
        dones_env = np.all(dones, axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]),
                                                       dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]),
                                                        dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        # [Selfplay] divide ego/opponent of collecting data TODO: shared_obs
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            self.opponent_masks = masks[:, self.num_agents // 2:, ...]

            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
            actions = actions[:, :self.num_agents // 2, ...]
            rewards = rewards[:, :self.num_agents // 2, ...]
            masks = masks[:, :self.num_agents // 2, ...]
            active_masks = active_masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, \
                           rnn_states_actor, rnn_states_critic, active_masks=active_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")

        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]),
                                           dtype=np.float32)

        eval_obs, eval_share_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
                                   dtype=np.float32)

        # [Selfplay] Choose opponent policy for evaluation
        if self.use_selfplay:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in
                                     range(self.all_args.n_choose_opponents)]
            assert self.eval_episodes >= self.all_args.n_choose_opponents, \
                f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.all_args.n_choose_opponents}"
            eval_each_episodes = self.eval_episodes // self.all_args.n_choose_opponents
            eval_cur_opponent_idx = 0
            logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")
            # TODO: use eval results to update elo

        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if self.use_selfplay and total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(
                    torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt'))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes + 1}/{self.eval_episodes})")

                # reset obs/rnn/mask
                eval_obs, eval_share_obs = self.eval_envs.reset()
                eval_masks = np.ones_like(eval_masks, dtype=np.float32)
                eval_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]
                eval_opponent_masks = np.ones_like(eval_masks, dtype=np.float32)
                eval_opponent_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # [Selfplay] get actions of opponent policy
            if self.use_selfplay:
                eval_opponent_actions, eval_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(eval_opponent_obs),
                                                    np.concatenate(eval_opponent_rnn_states),
                                                    np.concatenate(eval_opponent_masks))
                eval_opponent_rnn_states = np.array(
                    np.split(_t2n(eval_opponent_rnn_states), self.n_eval_rollout_threads))
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # [Selfplay] get ego reward
            if self.use_selfplay:
                eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]),
                                                          dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)
            # [Selfplay] reset opponent mask/rnn_states
            if self.use_selfplay:
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]
                eval_opponent_masks[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_masks.shape[1:]), dtype=np.float32)
                eval_opponent_rnn_states[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean()
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        self.log_info(eval_infos, total_num_steps)

        # [Selfplay] Reset opponent
        if self.use_selfplay:
            self.reset_opponent()
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs, render_share_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        if self.use_selfplay:
            policy_idx = self.render_opponent_index
            self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{policy_idx}.pt'))
            self.eval_opponent_policy.prep_rollout()
            # reset obs/rnn/mask
            render_obs, render_share_obs = self.envs.reset()
            render_masks = np.ones_like(render_masks, dtype=np.float32)
            render_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
            render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
            render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)

        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # [Selfplay] get actions of opponent policy
            if self.use_selfplay:
                render_opponent_actions, render_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                    np.concatenate(render_opponent_rnn_states),
                                                    np.concatenate(render_opponent_masks),
                                                    deterministic=True)
                render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
                render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
                render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            render_obs, render_share_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            if self.use_selfplay:
                render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
                render_obs = render_obs[:, :self.num_agents // 2, ...]

        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))



    @torch.no_grad()
    def render2(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index

        # 初始化 Matplotlib 的交互模式
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))
        im = None  # 用于存储图像对象

        # 设置图形标题和轴标签（可选）
        ax.set_title('实时注意力权重')
        ax.set_xlabel('注意力头')
        ax.set_ylabel('查询/键')

        # 假设 attn_weights_np 的形状是 (6, 12)，从1开始设置x轴和y轴的刻度
        num_rows, num_cols = 6, 7  # 根据实际矩阵的行列数来设置
        ax.set_xticks(np.arange(num_cols))  # 设置 x 轴的刻度在每个单元格的中间
        ax.set_yticks(np.arange(num_rows))  # 设置 y 轴的刻度在每个单元格的中间
        ax.set_xticklabels(np.arange(1, num_cols + 1))  # 设置 x 轴的刻度标签从1开始
        ax.set_yticklabels(np.arange(1, num_rows + 1))  # 设置 y 轴的刻度标签从1开始

        for episode in range(10):
            render_episode_rewards = 0
            render_obs, render_share_obs = self.envs.reset()
            render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
            render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

            while True:
                self.policy.prep_rollout()
                render_actions, render_rnn_states, attn_weights = self.policy.act2(np.concatenate(render_obs),
                                                                    np.concatenate(render_rnn_states),
                                                                    np.concatenate(render_masks),
                                                                    deterministic=True)
                render_actions = np.expand_dims(_t2n(render_actions), axis=0)
                render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

                # 将注意力权重从 GPU 转移到 CPU，并转换为 NumPy 数组
                attn_weights_np = attn_weights.cpu().numpy()
                # attn_weights_np = self.modify_attn_weights(attn_weights_np)

                # 如果是第一次绘制，初始化 im 对象
                if im is None:
                    im = ax.imshow(attn_weights_np, cmap='viridis', aspect='auto', vmin=0, vmax=0.5)
                    fig.colorbar(im, ax=ax)
                else:
                    im.set_data(attn_weights_np)

                # 刷新图形以显示更新
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Obser reward and next obs
                render_obs, render_share_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)

                render_episode_rewards += render_rewards
                if render_dones.all():
                    break

        # 关闭交互模式并显示最终图形
        plt.ioff()
        plt.show()

    def modify_attn_weights(self, attn_weights_np):
        """
        修改注意力权重矩阵，通过在对角线位置插入1，并将原有元素向右移动。

        参数：
        attn_weights_np (np.ndarray): 原始的注意力权重矩阵，形状为 (6, 11)

        返回：
        np.ndarray: 修改后的注意力权重矩阵，形状为 (6, 12)
        """
        num_rows, num_cols = attn_weights_np.shape
        modified_rows = []

        for i in range(num_rows):
            row = attn_weights_np[i]
            # 在第i个位置插入1
            modified_row = np.insert(row, i, 0.5)
            modified_rows.append(modified_row)

        # 将所有修改后的行堆叠成新的二维数组
        modified_attn_weights_np = np.vstack(modified_rows)
        return modified_attn_weights_np


    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        # [Selfplay] save policy & performance
        if self.use_selfplay:
            torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
            self.policy_pool[str(episode)] = self.all_args.init_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        # reset env
        obs, share_obs = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()
