"""
将人类偏好修改为连续值
"""

import torch
import numpy as np
from gym import spaces
from random import random

from envs.HighEnvSim.envs.conpre_envs.LowEnv_con import LowCommandEnv
from envs.HighEnvSim.utils.Constants import NormStates


class HighCommandEnv(LowCommandEnv):
    """高层环境，与元命令选择器进行交互"""

    def __init__(self, high_args, low_args, port, device=torch.device("cpu")):
        super().__init__(port, low_args, device)

        self.max_k_step = 200  # 每一个高级步包含的最大的低级步数
        self.action_num = 2  # 表示战术动作数量
        self.current_high_step = 0  # 当前的高级步数

        self.last_action = [-1] * self.ally_num  # 我方上一次选择的战术行为
        self.last_target = [-1] * self.ally_num  # 我方上一次选择的战术目标

        # 高层观测空间和动作空间定义
        self.high_obs_length = 25 + (self.ally_num - 1) * 19 + self.enemy_num * 17
        self.high_share_obs_length = self.high_obs_length
        self.observation_space = spaces.Box(low=-10., high=10., shape=(self.high_obs_length,))
        self.share_observation_space = spaces.Box(low=-10., high=10., shape=(self.high_share_obs_length,))
        self.action_space = spaces.MultiDiscrete([2, 3])


    def reset(self):
        """重置上层环境"""
        self.current_high_step = 0
        self.last_action = [-1] * self.ally_num
        self.last_target = [-1] * self.ally_num

        # 底层重置
        self.low_reset()

        # 初始化人类偏好及其编码
        self.human_preference = np.zeros((2, 3), dtype=np.float32)
        # self.human_preference[0][0] = 0.5

        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        action_mask = self.get_action_mask()

        return obs, share_obs, action_mask

    def step(self, actions):
        """上层环境步进函数"""
        reset_human_prefer = True

        self.current_high_step += 1
        info = {"current_high_step": self.current_high_step}
        done = np.full((self.ally_num, 1), False)

        # print(f"该高级步选择的元命令为：{actions}")

        # 得到上层的奖励
        reward_high = self.get_high_reward(actions)  # 得到上层给的奖励
        reward_low = 0  # 每一个高级步的累积奖励

        # 得到在有躲避偏好的情况下，此时在目标危险区域内的智能体
        ally_in_target_range = self.get_ally_in_target_range()

        for i in range(self.max_k_step):

            low_reward, low_done, end_action_low = self.low_step(actions)

            # 判断其他智能体这一步的上层动作是否从外面进入了危险区域
            end_action_mid, reward_mid = self.get_other_command_in_danger(ally_in_target_range)

            # 判断选择躲避目标的智能体是否躲避成功
            end_action_escape = self.get_end_action_escape(actions, ally_in_target_range)

            end_action = end_action_low or end_action_mid or end_action_escape

            reward_low += low_reward
            reward_low += reward_mid
            done = low_done

            # # # 人类偏好随机产生，将会重新选择元命令
            # if random() < 0.002:
            #     end_action = self.real_time_human_prefer()
            #     reset_human_prefer = False

            if end_action:
                # 触发特定事件，重新选择元命令
                break

        # if reset_human_prefer:
        #     self.human_preference = np.zeros((2, 3), dtype=np.float32)  # 人类对元命令的偏好

        reward = reward_high + reward_low

        self.update_last_command(actions)  # 更新上一次选择的元命令

        # 更新储存的双方信息
        self.ally_states, self.enemy_states = self.get_all_plane_states()

        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        action_mask = self.get_action_mask()

        return obs, share_obs, reward, done, info, action_mask

    def real_time_human_prefer(self):
        """
        实时修改人类对元命令的偏好的接口
        """
        human_preference = np.zeros((2, 3), dtype=np.float32)
        for i in range(self.enemy_group_num):
            random_action = np.random.randint(0, 2)
            human_preference[random_action][i] = np.random.uniform(0, 1)

        self.human_preference = human_preference

        return True

    def update_last_command(self, actions):
        """更新上一次选择的元命令"""
        for ally_id in range(self.ally_num):
            self.last_action[ally_id] = actions[ally_id][0]
            self.last_target[ally_id] = actions[ally_id][1]

    def get_high_reward(self, actions):
        """得到上层给的奖励"""
        rewards = [self.get_high_single_reward(ally_id, actions) for ally_id in range(self.ally_num)]
        all_ally_reward = np.array(rewards, dtype=np.float32).reshape(self.ally_num, 1)

        return all_ally_reward

    def get_high_single_reward(self, agent_id, actions):
        """
        得到上层对单个智能体的奖励
            1.选择的元命令变化的惩罚
            2.与人类偏好一致的奖励
        """
        own_reward = 0

        own_state = self.ally_states[agent_id]
        if own_state["health_level"] > 0:
            reward_maintain_command = self.get_reward_maintain_command(agent_id, actions)
            reward_human_prefer = self.get_reward_human_prefer(agent_id, actions)  # 选择的动作与人类偏好一致的奖励
            own_reward = reward_maintain_command + reward_human_prefer

        return own_reward

    def get_reward_human_prefer(self, agent_id, actions):
        """智能体选择的动作与人类的偏好一致的奖励"""
        own_action = actions[agent_id][0]
        own_target = actions[agent_id][1]

        if own_action == 0:
            # 表示智能体选择的是攻击某一组，符合偏好的奖励和与躲避某组冲突的惩罚
            attack_preference = self.human_preference[0][own_target]
            reward_attack_prefer = (attack_preference ** 2) * self.human_reward_max
            escape_preference = self.human_preference[1][own_target]
            reward_escape_prefer = -(escape_preference ** 2) * self.human_reward_max
            reward_human_prefer = reward_attack_prefer + reward_escape_prefer
        else:
            # 表示智能体选择的是躲避某一组，则只有在该组危险区域内的智能体才有奖励
            own_in_target_danger_range = self.is_in_target_danger_range(agent_id, own_target)
            if own_in_target_danger_range:
                # 表示该智能体在危险区域内
                preference = self.human_preference[1][own_target]
                reward_human_prefer = (preference ** 2) * self.human_reward_max
            else:
                reward_human_prefer = 0

        return reward_human_prefer


    def get_reward_maintain_command(self, agent_id, actions):
        """命令维持的奖励和变化的惩罚"""
        reward = 0
        if self.last_action[agent_id] != -1 and self.last_action[agent_id] != actions[agent_id][0]:
            # 表示动作变化
            reward = -3
        elif self.last_target[agent_id] != -1 and self.last_target[agent_id] != actions[agent_id][1]:
            # 表示目标变化
            reward = -3

        return reward

    def get_obs(self):
        """得到智能体的观测信息"""
        all_ally_obs = [self.get_single_obs(ally_id) for ally_id in range(self.ally_num)]
        return np.stack(all_ally_obs)

    def get_share_obs(self, obs):
        """得到智能体的全局观测"""
        return obs

    def get_action_mask(self):
        """得到动作掩码"""
        action_mask = np.ones((self.enemy_group_num, ), dtype=int)

        for group_index, group in enumerate(self.enemy_initial_group):
            if all(self.enemy_states[enemy_id]["health_level"] == 0 for enemy_id in group):
                action_mask[group_index] = 0

        all_action_mask = np.tile(action_mask[None, :], (self.ally_num, 1))

        return all_action_mask

    def get_single_obs(self, agent_id):
        """
        元命令选择器的观测信息
        自身信息:基本信息、上一次选择的元命令信息、人类对各个命令的偏好信息集合
        队友信息：基本信息、上一次选择的元命令信息
        敌方信息：基本信息、所属组别信息

        """
        own_feat = np.zeros((25,), dtype=np.float32)  # 自身信息
        ally_feats = np.zeros((self.ally_num - 1, 19), dtype=np.float32)  # 队友信息
        enemy_feats = np.zeros((self.enemy_num, 17), dtype=np.float32)  # 敌方信息

        own_state = self.ally_states[agent_id]

        if own_state["health_level"] > 0:
            # 如果该无人机的血量不为0
            own_move_vector = own_state["move_vector"]
            own_pos_vector = own_state["position"]

            # 位置相关
            own_x, own_y, own_z = own_state["position"]
            norm_own_x = (own_x / NormStates["bound_radius"])
            norm_own_z = (own_z / NormStates["bound_radius"])
            norm_own_y = (own_y / NormStates["bound_altitude_max"])

            # 速度相关
            own_vx, own_vy, own_vz = own_state["move_vector"]
            norm_own_vx = (own_vx / NormStates["speed_max"])
            norm_own_vz = (own_vz / NormStates["speed_max"])
            norm_own_vy = (own_vy / NormStates["speed_max"])
            speed = own_state["linear_speed"]  # 总的速度标量
            norm_speed = (speed / NormStates["speed_max"])

            # 姿态相关
            pitch, yaw, roll = own_state["Euler_angles"]  # 俯仰、偏航、滚转
            pitch_sin, pitch_cos = self.get_theta_sin_cos(pitch)
            yaw_sin, yaw_cos = self.get_theta_sin_cos(yaw)
            roll_sin, roll_cos = self.get_theta_sin_cos(roll)

            # 健康值
            health_level = 1

            # 自己上一次选择的元命令
            own_last_action = self.last_action[agent_id]
            if own_last_action == -1:
                last_action_encode = np.array([0, 0])
            else:
                last_action_encode = self.one_hot_encode_action(own_last_action)

            own_last_target = self.last_target[agent_id]
            if own_last_target == -1:
                last_target_encode = np.array([0, 0, 0])
            else:
                last_target_encode = self.one_hot_encode_target(own_last_target)

            # 人类对各个元命令的偏好信息
            human_prefer_encode = self.encode_human_preference()

            own_feat[0:25] = np.array([norm_own_x, norm_own_y, norm_own_z, norm_own_vx, norm_own_vy,
                                       norm_own_vz, pitch_sin, pitch_cos, yaw_sin, yaw_cos, roll_sin,
                                       roll_cos, norm_speed, health_level] + last_action_encode.tolist() +
                                      last_target_encode.tolist() + human_prefer_encode.tolist())

            # 队友信息
            ally_idx = 0
            for ally_id in range(self.ally_num):
                if ally_id == agent_id:
                    continue
                ally_id_state = self.ally_states[ally_id]

                if ally_id_state["health_level"] > 0:
                    ally_pos_vector = ally_id_state["position"]
                    ally_move_vector = ally_id_state["move_vector"]

                    # 相对位置
                    ally_x, ally_y, ally_z = ally_id_state["position"]
                    ally_relative_pos = np.array([ally_x - own_x, ally_y - own_y, ally_z - own_z])
                    norm_ally_relative_x = (ally_relative_pos[0] / NormStates["bound_radius"] / 2)
                    norm_ally_relative_y = (ally_relative_pos[1] / NormStates["relative_altitude_max"])
                    norm_ally_relative_z = (ally_relative_pos[2] / NormStates["bound_radius"] / 2)

                    # 速度相关
                    ally_vx, ally_vy, ally_vz = ally_id_state["move_vector"]
                    norm_ally_vx = (ally_vx / NormStates["speed_max"])
                    norm_ally_vy = (ally_vy / NormStates["speed_max"])
                    norm_ally_vz = (ally_vz / NormStates["speed_max"])

                    # 距离和角度相关，利用AA、ATA、HA、R、side表征,该函数直接返回各个角度的sin和cos值，然后是R和side
                    ally_ATA, ally_AA, ally_HA, ally_R = self.get_ATA_AA_HA_R(own_pos_vector, own_move_vector,
                                                                              ally_pos_vector, ally_move_vector)
                    norm_ally_R = (ally_R / NormStates["relative_distance_max"])

                    # 健康值
                    health_level = 1

                    # 队友上一次选择的元命令信息
                    ally_last_action = self.last_action[ally_id]
                    if ally_last_action == -1:
                        last_action_encode = np.array([0, 0])
                    else:
                        last_action_encode = self.one_hot_encode_action(ally_last_action)

                    ally_last_target = self.last_target[ally_id]
                    if ally_last_target == -1:
                        last_target_encode = np.array([0, 0, 0])
                    else:
                        last_target_encode = self.one_hot_encode_target(ally_last_target)

                    ally_feats[ally_idx, 0:19] = np.array([norm_ally_relative_x, norm_ally_relative_y,
                                                           norm_ally_relative_z, norm_ally_vx, norm_ally_vy,
                                                           norm_ally_vz, np.sin(ally_ATA), np.cos(ally_ATA),
                                                           np.sin(ally_AA), np.cos(ally_AA), np.sin(ally_HA),
                                                           np.cos(ally_HA), norm_ally_R, health_level] +
                                                          last_action_encode.tolist() + last_target_encode.tolist())
                ally_idx += 1

            # 敌方信息
            for enemy_id in range(self.enemy_num):
                enemy_id_state = self.enemy_states[enemy_id]

                if enemy_id_state["health_level"] > 0:
                    enemy_pos_vector = enemy_id_state["position"]
                    enemy_move_vector = enemy_id_state["move_vector"]

                    # 相对位置
                    enemy_x, enemy_y, enemy_z = enemy_id_state["position"]
                    enemy_relative_pos = np.array([enemy_x - own_x, enemy_y - own_y, enemy_z - own_z])
                    norm_enemy_relative_x = (enemy_relative_pos[0] / NormStates["bound_radius"] / 2)
                    norm_enemy_relative_y = (enemy_relative_pos[1] / NormStates["relative_altitude_max"])
                    norm_enemy_relative_z = (enemy_relative_pos[2] / NormStates["bound_radius"] / 2)

                    # 速度相关
                    enemy_vx, enemy_vy, enemy_vz = enemy_id_state["move_vector"]
                    norm_enemy_vx = (enemy_vx / NormStates["speed_max"])
                    norm_enemy_vy = (enemy_vy / NormStates["speed_max"])
                    norm_enemy_vz = (enemy_vz / NormStates["speed_max"])

                    # 距离和角度相关
                    enemy_ATA, enemy_AA, enemy_HA, enemy_R = self.get_ATA_AA_HA_R(own_pos_vector,
                                                                                  own_move_vector,
                                                                                  enemy_pos_vector,
                                                                                  enemy_move_vector)
                    norm_enemy_R = (enemy_R / NormStates["relative_distance_max"])

                    # 健康值
                    health_level = 1

                    # 敌方所属组别的独热编码
                    enemy_id_group = self.get_enemy_group(enemy_id)
                    enemy_hot_encode = self.one_hot_encode_target(enemy_id_group)

                    enemy_feats[enemy_id, 0:17] = np.array([norm_enemy_relative_x, norm_enemy_relative_y,
                                                            norm_enemy_relative_z, norm_enemy_vx, norm_enemy_vy,
                                                            norm_enemy_vz, np.sin(enemy_ATA), np.cos(enemy_ATA),
                                                            np.sin(enemy_AA), np.cos(enemy_AA), np.sin(enemy_HA),
                                                            np.cos(enemy_HA), norm_enemy_R, health_level] +
                                                           enemy_hot_encode.tolist())

        own_obs = np.concatenate([own_feat.flatten(), ally_feats.flatten(), enemy_feats.flatten()])

        return own_obs

    def encode_human_preference(self):
        """将人类偏好按照单调性编码"""
        attack_prefer_encode_feats = np.zeros((3, ), dtype=np.float32)
        escape_prefer_encode_feats = np.zeros((3, ), dtype=np.float32)

        # 得到每个敌方组的敌人的存活情况
        action_mask = self.get_action_mask()

        for group_id in range(self.enemy_group_num):
            if action_mask[0][group_id] == 0:
                # 表示该组所有敌方被击毁
                attack_prefer_encode_feats[group_id] = 0
                escape_prefer_encode_feats[group_id] = 0
            else:
                # 表示该组还活着
                attack_preference = self.human_preference[0][group_id]
                attack_prefer_encode_feats[group_id] = attack_preference # 将偏好归一化到[0.25, 0.5, 0.75, 1]

                escape_preference = self.human_preference[1][group_id]
                escape_prefer_encode_feats[group_id] = escape_preference

        # 将攻击和躲避的编码合并成6维的特征向量
        final_encoding = np.concatenate(
            [attack_prefer_encode_feats.flatten(), escape_prefer_encode_feats.flatten()], axis=0)

        return final_encoding


    def one_hot_encode_action(self, action_id):
        """对动作进行独热编码"""
        one_hot_encoded = np.zeros(self.action_num)
        one_hot_encoded[action_id] = 1
        return one_hot_encoded

    #######躲避与终止条件相关#######
    def get_end_action_escape(self, actions, ally_in_target_range):
        """当在敌方危险区域内的智能体成功躲避时，重新选择"""
        is_escape_complete = False

        for ally_id in range(self.ally_num):
            ally_state = self.ally_states[ally_id]
            ally_escape_target = actions[ally_id][1]
            if ally_state["health_level"] > 0:
                if ally_in_target_range[ally_escape_target][ally_id] == 1 and actions[ally_id][0] == 1:
                    # 表示躲避并且刚开始就在对应危险区域内
                    is_escape_complete = self.get_escape_complete(ally_id, actions[ally_id][1])
                    if is_escape_complete:
                        return is_escape_complete

        return is_escape_complete

    def get_escape_complete(self, agent_id, own_target):
        """判断是否躲避成功"""
        current_min_dis = np.inf
        own_state = self.ally_states[agent_id]
        own_x, own_y, own_z = own_state["position"]

        for enemy_id in self.enemy_initial_group[own_target]:
            enemy_state = self.enemy_states[enemy_id]
            if enemy_state["health_level"] > 0:
                enemy_x, enemy_y, enemy_z = enemy_state["position"]
                enemy_relative_pos = np.array([enemy_x - own_x, enemy_y - own_y, enemy_z - own_z])
                distance = np.linalg.norm(enemy_relative_pos)
                if distance < current_min_dis:
                    current_min_dis = distance

        if current_min_dis > self.task_hyper.safe_range:
            return True
        else:
            return False

    def is_in_target_danger_range(self, agent_id, own_target):
        """判断是否在应该躲避的目标的危险区域内"""
        current_min_dis = np.inf
        own_state = self.ally_states[agent_id]
        own_x, own_y, own_z = own_state["position"]

        for enemy_id in self.enemy_initial_group[own_target]:
            enemy_state = self.enemy_states[enemy_id]
            if enemy_state["health_level"] > 0:
                enemy_x, enemy_y, enemy_z = enemy_state["position"]
                enemy_relative_pos = np.array([enemy_x - own_x, enemy_y - own_y, enemy_z - own_z])
                distance = np.linalg.norm(enemy_relative_pos)
                if distance < current_min_dis:
                    current_min_dis = distance

        if current_min_dis < self.task_hyper.danger_range:
            return True
        else:
            return False

    def get_other_command_in_danger(self, ally_in_target_range):
        """如果其他命令导致其进入了应该躲避的目标的危险区域，则终止并且给该选择给予惩罚"""
        reward = np.zeros((self.ally_num, 1), np.float32)
        end_action = False

        human_prefer_escape_id = np.where(self.human_preference[1] != 0)[0]  # 躲避偏好不为0的组别索引
        for ally_id in range(self.ally_num):
            ally_state = self.ally_states[ally_id]
            if ally_state["health_level"] > 0:
                for group_id in human_prefer_escape_id:
                    if ally_in_target_range[group_id][ally_id] == 0:
                        # 判断智能体是否进入到了应该躲避的危险区域内
                        is_from_safe_to_danger = self.is_in_target_danger_range(ally_id, group_id)
                        if is_from_safe_to_danger:
                            end_action = True
                            preference = self.human_preference[1][group_id]
                            reward[ally_id] -= (preference ** 2) * self.human_reward_max

        return end_action, reward

    def get_ally_in_target_range(self):
        """对于有躲避偏好的智能体，记录我方每一个智能体是否在其危险区域内"""
        ally_in_target_range = np.zeros((self.enemy_group_num, self.ally_num), dtype=np.int32)

        for group_id in range(self.enemy_group_num):
            if self.human_preference[1][group_id] != 0:
                # 表示对躲避该组有偏好
                for ally_id in range(self.ally_num):
                    ally_state = self.ally_states[ally_id]
                    if ally_state["health_level"] > 0:
                        is_ally_in_danger_range = self.is_in_target_danger_range(ally_id, group_id)
                        if is_ally_in_danger_range:
                            ally_in_target_range[group_id][ally_id] = 1

        return ally_in_target_range

