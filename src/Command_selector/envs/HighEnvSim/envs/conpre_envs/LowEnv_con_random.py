"""
底层预训练模型，用于执行给定的元命令
TODO:
    1.攻击机制中加入终止机制
    2.仔细思考上层中选择击毁敌方后可以得到的奖励，是平分还是独享
"""
import copy
import os
import sys
from pathlib import Path
import gym
import torch
import random
import time
import math
import json
import numpy as np
from math import pi
from gym import spaces
from gym.utils import seeding

from envs.HighEnvSim.low_models.LowActorNet import AttackLowActor, EscapeLowActor
from envs.HighEnvSim.utils import dogfight_client as df
from envs.HighEnvSim.utils.Constants import NormStates
from envs.HighEnvSim.HumanControls.EnvSocket import EnvSocketServer

CONFIG_DIR = Path(__file__).resolve().parents[6] / "configs"


def _load_json_config(filename):
    """Return JSON content from configs/<filename> if present, otherwise None."""
    config_path = CONFIG_DIR / filename
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _require_config(values, required_keys, source_hint):
    """
    Ensure required keys exist; raise with a clear message if they were intentionally redacted.
    """
    values = values or {}
    missing = [key for key in required_keys if key not in values or values[key] is None]
    if missing:
        raise RuntimeError(
            f"Core configuration values were removed for the public release. "
            f"Provide {source_hint}. Missing keys: {', '.join(missing)}"
        )
    return values


class TaskConfig(object):
    REQUIRED_KEYS = [
        "attack_angle",
        "attack_distance",
        "hyper_feature_safe_distance",
        "danger_range",
        "safe_range",
        "step_punish",
        "out_bound_punish",
    ]

    def __init__(self):
        """属性配置；核心参数已从开源版本移除，请在 configs/task_config.json 中填写。"""
        task_config = _require_config(
            _load_json_config("task_config.json"),
            self.REQUIRED_KEYS,
            "configs/task_config.json with numeric values",
        )
        self.attack_angle = float(task_config["attack_angle"])  # 有效攻击角度
        self.attack_distance = float(task_config["attack_distance"])  # 有效攻击距离
        self.hyper_feature_safe_distance = float(task_config["hyper_feature_safe_distance"])  # 飞机之间的碰撞距离为50m

        # 敌方安全区域和危险区域定义
        self.danger_range = float(task_config["danger_range"])
        self.safe_range = float(task_config["safe_range"])

        # 奖励相关
        self.step_punish = float(task_config["step_punish"])
        self.out_bound_punish = float(task_config["out_bound_punish"])


class LowCommandEnv(object):
    def __init__(self, port, low_args, device=torch.device("cpu")):

        self.port = port
        self.low_args = low_args
        self.device = device

        self.ally_num = 6
        self.enemy_num = 6
        self.Plane_ID_ally = []
        self.Plane_ID_enemy = []

        # 变量定义
        self.current_step = 0
        self.initial_speed = 150
        self._create_records = False  # 表示是否需要记录环境
        self.end_high_action = False  # 上层命令的终止信号，也就是特定事件发生时需要重新做出高级选择
        self.enemy_initial_yaws = {}  # 用于记录每架飞机在该回合的yaw

        self.task_hyper = TaskConfig()
        self.all_ally_dones = np.full((self.ally_num, 1), False)

        # 敌方分组
        self.enemy_group_num = 3
        self.enemy_initial_group = [[] for _ in range(self.enemy_group_num)]  # 初始化组别列表

        # 预训练底层网络定义
        self.attack_obs_length = 17 + (self.ally_num - 1) * 15 + self.enemy_num * 17
        self.escape_obs_length = 17 + 1 * 14 + self.enemy_num * 18
        self.attack_observation_space = spaces.Box(low=-10., high=10., shape=(self.attack_obs_length,))
        self.escape_observation_space = spaces.Box(low=-10., high=10., shape=(self.escape_obs_length,))
        self.attack_action_space = spaces.MultiDiscrete([11, 11, 11])
        self.escape_action_space = spaces.MultiDiscrete([11, 11, 11])

        # 定义和加载预训练模型
        self.attack_policy = AttackLowActor(self.attack_action_space, self.device)
        self.escape_policy = EscapeLowActor(self.escape_action_space, self.device)

        self.attack_masks = np.ones((self.ally_num, 1), dtype=np.float32)
        self.escape_masks = np.ones((self.ally_num, 1), dtype=np.float32)
        self.attack_rnn_states = np.zeros((self.ally_num, 1, 128), dtype=np.float32)
        self.escape_rnn_states = np.zeros((self.ally_num, 1, 128), dtype=np.float32)

        # 攻击机制相关
        self.ally_launch_attack = [False] * self.ally_num
        self.enemy_launch_attack = [False] * self.enemy_num
        self.ally_being_attack = [False] * self.ally_num
        self.enemy_being_attack = [False] * self.enemy_num

        self.reward_attack_group = [0] * self.ally_num  # 每一回合智能体的击毁奖励存储

        # 环境连接与模型加载
        self.connect_harfang()
        self.load_low_model()
        # self.seed()

        # 人类偏好相关变量定义
        self.human_preference = np.zeros((2, 3), dtype=np.float32)  # 人类对元命令的偏好
        self.human_reward_max = 40
        # self.pre_levels = 4  # 将人类偏好离散化为4个等级，分别为0，1，2，3
        # self.human_pre_reward = [[0, 2.5, 3.5, 6], [0, 2.5, 3.5, 6]]  # 对应不同偏好人类给予的奖励
        self.human_encode_set = {}  # 人类偏好信息编码

        # 创建socket服务
        socket_host = '127.0.0.1'
        socket_port = 9999
        self.socket_server = EnvSocketServer(host=socket_host, port=socket_port)
        self.socket_server.env = self

        # 环境重置信息，辅助前端偏好输入
        self.reset_flag = False

        # 初始化指标统计
        self.total_episodes = 0
        self.completed_episodes = 0  # 任务完成的回合数
        self.time_out_episodes = 0  # 超时没有完成的回合数
        self.total_ally_losses = 0  # 总战损数
        self.total_enemy_kills = 0  # 总击毁数
        self.total_time_consumption = 0  # 总消耗时间（步数）

    def randomize_enemy_groups(self):
        """
        为每个敌机随机分配一个组别（0, 1, 2），允许某些组别为空。
        """
        self.enemy_initial_group = [[] for _ in range(self.enemy_group_num)]
        enemy_ids = list(range(self.enemy_num))
        # random.shuffle(enemy_ids)  # 打乱敌机顺序（可选）

        for enemy_id in enemy_ids:
            group_id = random.randint(0, self.enemy_group_num - 1)  # 随机选择组别
            self.enemy_initial_group[group_id].append(enemy_id)

    def low_reset(self):
        """底层环境重置"""
        self.current_step = 0
        self._create_records = False  # 和Tacview显示相关
        self.initial_yaw = np.random.uniform(-pi, pi)  # 初始状态角度
        high_actions = np.full((self.ally_num, 2), -1)

        # 重置攻击机制
        self.ally_launch_attack = [False] * self.ally_num
        self.enemy_launch_attack = [False] * self.enemy_num
        self.ally_being_attack = [False] * self.ally_num
        self.enemy_being_attack = [False] * self.enemy_num

        self.reward_attack_group = [0] * self.ally_num  # 每一回合智能体的击毁奖励存储

        # 重置done
        self.all_ally_dones = np.full((self.ally_num, 1), False)

        # 重置rnn参数
        self.attack_masks = np.ones((self.ally_num, 1), dtype=np.float32)
        self.escape_masks = np.ones((self.ally_num, 1), dtype=np.float32)
        self.attack_rnn_states = np.zeros((self.ally_num, 1, 128), dtype=np.float32)
        self.escape_rnn_states = np.zeros((self.ally_num, 1, 128), dtype=np.float32)

        # 重置底层环境
        self.reset_machine()

        self.ally_states, self.enemy_states = self.get_all_plane_states()

        # 重置后发送当前状态
        self.reset_flag = True
        self.send_positions_to_frontend(high_actions)

        # 重置每回合的临时指标
        self.current_episode_ally_losses = 0  # 当前回合的战损数
        self.current_episode_enemy_kills = 0  # 当前回合的击毁数
        self.episode_completed = False  # 当前回合是否完成任务
        self.episode_time = 0  # 当前回合消耗的时间（步数）

    def low_step(self, high_actions):
        """底层环境步进函数"""
        self.current_step += 1
        self.end_high_action = False
        self.reset_flag = False

        self.set_enemy_policy(self.enemy_states)  # 敌方策略

        # 根据上层选择的元命令得到此时对应网络的观测信息
        attack_all_obs, escape_all_obs = self.get_low_obs(high_actions)
        attack_actions, _, self.attack_rnn_states = self.attack_policy(attack_all_obs, self.attack_rnn_states,
                                                                       self.attack_masks, deterministic=True)
        escape_actions, _, self.escape_rnn_states = self.escape_policy(escape_all_obs, self.escape_rnn_states,
                                                                       self.escape_masks, deterministic=True)
        # 将上层动作进行转化，然后应用到环境中
        self.apply_action(high_actions, attack_actions, escape_actions)

        self.ally_states, self.enemy_states = self.get_all_plane_states()

        # 显示飞机序号
        self.show_plane_id(high_actions)

        # 得到这一步的奖励
        low_step_reward = self.get_low_reward(high_actions)

        # 判断接近出界的时候让其重新选择元命令
        self.get_close_bound_end_action()

        # 因为计算奖励后飞机的状态发生变化，所以需要更新所有飞机状态
        self.ally_states, self.enemy_states = self.get_all_plane_states()

        low_step_done = self.get_low_done()

        # 每一步发送一次当前状态
        self.send_positions_to_frontend(high_actions)

        # 记录当前回合的消耗时间
        self.episode_time += 1

        if np.all(low_step_done):
            # 如果回合结束，您可以选择在这里调用报告方法
            self.report_metrics()

        return low_step_reward, low_step_done, self.end_high_action

    def set_human_preference(self, preference_data):
        """
        设置人类偏好。
        preference_data: 二维列表，例如 [[0.5, 0.7, 0.3], [0.2, 0.0, 0.9]]
        """
        self.human_preference = np.array(preference_data, dtype=np.float32)
        print("已更新人类偏好为：", self.human_preference)
        self.end_high_action = True
    def compute_threat_levels(self):
        """
        计算每一组敌方的威胁值，威胁值基于每组中存活的飞机数量。
        威胁值范围在0到1之间。
        返回：
            threat_levels (list of dict): 每组的威胁值列表，格式如：
                [
                    {"group_id": 0, "threat_level": 0.5},
                    {"group_id": 1, "threat_level": 1.0},
                    {"group_id": 2, "threat_level": 0.0}
                ]
        """
        threat_levels = []
        for group_id, group in enumerate(self.enemy_initial_group):
            alive_planes = 0
            for enemy_id in group:
                if self.enemy_states[enemy_id]["health_level"] > 0:
                    alive_planes += 1
            # 计算威胁值：存活飞机数除以该组总飞机数
            threat_level = alive_planes / len(group) if len(group) > 0 else 0.0
            threat_levels.append({
                "group_id": int(group_id),
                "threat_level": float(threat_level)
            })
        return threat_levels

    def send_positions_to_frontend(self, high_actions):
        """
        将友方和敌方的位置信息、速度及姿态发送给前端。
        现在包括每组敌方的威胁值。
        确保数据格式与前端期望一致，包括死亡飞机的健康信息。
        """
        ally_positions = []
        for ally_id, ally_state in enumerate(self.ally_states):
            x, y, z = ally_state["position"]
            vx, vy, vz = ally_state["move_vector"]
            pitch, yaw, roll = ally_state["Euler_angles"]  # 获取姿态角度
            ally_positions.append({
                "id": f"ally{ally_id + 1}",
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
                "command": high_actions[ally_id].tolist(),  # 确保是列表格式
                "rotation": {
                    "pitch": float(math.degrees(pitch)),  # 转换为度
                    "yaw": float(math.degrees(yaw)),
                    "roll": float(math.degrees(roll))
                },
                "health_level": ally_state["health_level"]  # 始终发送健康值
            })

        enemy_positions = []
        for enemy_id, enemy_state in enumerate(self.enemy_states):
            # 始终包含所有敌方飞机（包括死亡的飞机）
            x, y, z = enemy_state["position"]
            vx, vy, vz = enemy_state["move_vector"]
            pitch, yaw, roll = enemy_state["Euler_angles"]  # 获取姿态角度
            group_id = self.get_enemy_group(enemy_id)
            enemy_positions.append({
                "id": f"enemy{enemy_id + 1}",
                "group": int(group_id) if group_id is not None else -1,  # 如果未分组，设为-1
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
                "rotation": {
                    "pitch": float(math.degrees(pitch)),
                    "yaw": float(math.degrees(yaw)),
                    "roll": float(math.degrees(roll))
                },
                "health_level": enemy_state["health_level"]  # 包含健康值
            })

        # 计算每组的威胁值
        threat_levels = self.compute_threat_levels()

        data_to_send = {
            "type": "update_positions",
            "reset": self.reset_flag,
            "ally": ally_positions,
            "enemy": enemy_positions,
            "human_preference": self.human_preference.tolist(),
            "enemy_groups": threat_levels  # 添加威胁值信息
        }

        # 递归转换数据以确保 JSON 序列化成功
        data_to_send = self.convert_to_serializable(data_to_send)

        message = json.dumps(data_to_send) + "\n"
        self.socket_server.send(message)

    def convert_to_serializable(self, obj):
        """
        递归地将 NumPy 数据类型转换为 Python 数据类型，以便 JSON 序列化。
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def get_close_bound_end_action(self):
        """在飞机有出界危险时，终止本次命令执行"""
        for ally_id in range(self.ally_num):
            ally_state = self.ally_states[ally_id]
            if ally_state["health_level"] > 0:
                pos_vector = ally_state["position"]
                x, y, z = pos_vector
                if abs(x) > (NormStates["bound_radius"] - 1000) or abs(z) > (NormStates["bound_radius"] - 1000):
                    self.end_high_action = True
                    return
                if y > (NormStates["bound_altitude_max"] - 600) or y < (NormStates["bound_altitude_min"] + 600):
                    self.end_high_action = True
                    return

    def get_low_reward(self, high_actions):
        """得到每一步的奖励"""
        # 击毁奖励共享机制、被击毁惩罚单独承受
        self.reward_attack_group = [0] * self.ally_num

        # 每一步一个智能体只能攻击一次敌方
        self.ally_launch_attack = [False] * self.ally_num
        self.enemy_launch_attack = [False] * self.enemy_num

        rewards = [self.get_low_single_reward(ally_id) for ally_id in range(self.ally_num)]

        process_rewards = self.get_reward_common_attack(high_actions, rewards)  # 击毁敌方的共同奖励

        all_ally_reward = np.array(process_rewards, dtype=np.float32).reshape(self.ally_num, 1)

        self.clear_death()

        return all_ally_reward

    def get_reward_common_attack(self, high_actions, rewards):
        """将击毁敌方的奖励平分"""
        process_rewards = copy.deepcopy(rewards)

        # 分组攻击者，按目标编号分配队友
        target_to_agents = {}

        for agent_id, action in enumerate(high_actions):
            action_type = action[0]
            target = action[1]  # 目标编号

            if action_type == 0:
                # 如果行动是攻击
                if target not in target_to_agents:
                    target_to_agents[target] = []
                target_to_agents[target].append(agent_id)

        # 对每一个目标组计算奖励
        for target, agent_ids in target_to_agents.items():
            # 过滤死亡的智能体
            alive_agents = [agent_id for agent_id in agent_ids if self.ally_states[agent_id]["health_level"] > 0]
            # 将击杀目标的己方分为两类，一类是贡献大的，一类是贡献小的
            attack_agents = self.get_attack_agents(alive_agents)  # 从存活智能体中得到贡献大的几个

            num_alive_agents = len(alive_agents)
            num_attack_agents = len(attack_agents)

            if num_alive_agents > 0 and num_attack_agents > 0:
                # 计算该目标的总奖励
                total_reward = sum(self.reward_attack_group[agent_id] for agent_id in alive_agents)

                # 计算平分的奖励
                reward_per_agent = total_reward / num_attack_agents  # 给有实际贡献的智能体的奖励
                reward_per_less_agent = 1

                for agent_id in alive_agents:
                    if agent_id in attack_agents:
                        process_rewards[agent_id] += reward_per_agent
                    else:
                        process_rewards[agent_id] += reward_per_less_agent

        return process_rewards

    def get_attack_agents(self, alive_agents):
        """得到击杀敌方智能体，相对重要的智能体"""
        first_attack_agent_id = -1
        attack_agents = []

        for agent_id in alive_agents:
            if self.reward_attack_group[agent_id] > 0:
                first_attack_agent_id = agent_id

        if first_attack_agent_id != -1:
            first_agent_id_state = self.ally_states[first_attack_agent_id]
            first_agent_pos_vector = first_agent_id_state["position"]
            # 遍历活着的这几个智能体，找到离的近的智能体
            for ally_id in alive_agents:
                if ally_id == first_attack_agent_id:
                    attack_agents.append(ally_id)
                    continue
                ally_id_state = self.ally_states[ally_id]
                ally_id_pos_vector = ally_id_state["position"]
                delta_x, delta_y, delta_z = (first_agent_pos_vector[0] - ally_id_pos_vector[0],
                                             first_agent_pos_vector[1] - ally_id_pos_vector[1],
                                             first_agent_pos_vector[2] - ally_id_pos_vector[2])
                distance = np.linalg.norm([delta_x, delta_y, delta_z])
                if distance < 3000:
                    attack_agents.append(ally_id)

        return attack_agents

    def get_low_obs(self, high_actions):
        """根据上层选择的元命令得到所有底层网络的观测"""
        all_attack_low_obs = [self.get_low_attack_single_obs(ally_id, high_actions, self.ally_states, self.enemy_states) for ally_id in range(self.ally_num)]
        all_escape_low_obs = [self.get_low_escape_single_obs(ally_id, high_actions, self.ally_states, self.enemy_states) for ally_id in range(self.ally_num)]

        return np.stack(all_attack_low_obs), np.stack(all_escape_low_obs)

    def get_low_done(self):
        """判断是否需要重启环境"""
        for ally_id in range(self.ally_num):
            self.all_ally_dones[ally_id] = self.get_low_single_done(ally_id)

        game_over = self.done_win_game()  # 一方全部死亡

        if self.current_step > 8000:
            # print("环境", self.port, "运行超时，进行重置")
            self.all_ally_dones = np.full((self.ally_num, 1), True)
            self.end_high_action = True
            self.total_episodes += 1
            self.time_out_episodes += 1

        if game_over:
            # print("环境", self.port, "一方全部死亡，进行重置")
            # 判断任务完成的情况
            ally_alive = sum([self.ally_states[ally_id]["health_level"] > 0 for ally_id in range(self.ally_num)])
            enemy_alive = sum([self.enemy_states[enemy_id]["health_level"] > 0 for enemy_id in range(self.enemy_num)])

            if enemy_alive == 0 and ally_alive > 0:
                # 任务完成
                self.episode_completed = True
                self.completed_episodes += 1

            elif ally_alive == 0 and enemy_alive > 0:
                # 任务失败
                pass  # 仅记录战损数和击毁数

            # 记录消耗的时间
            self.episode_time = self.current_step

            # 更新总消耗时间
            self.total_time_consumption += self.episode_time

            # 更新总战损和击毁数
            self.total_ally_losses += self.current_episode_ally_losses
            self.total_enemy_kills += self.current_episode_enemy_kills

            # 更新总回合数
            self.total_episodes += 1

            self.all_ally_dones = np.full((self.ally_num, 1), True)
            self.end_high_action = True

        # 记录当前回合的时间步
        self.episode_time = self.current_step

        return self.all_ally_dones

    def done_win_game(self):
        """一方全部死亡终止"""
        game_over = False
        ally_alive = [self.ally_states[ally_id]["health_level"] for ally_id in range(self.ally_num)]
        enemy_alive = [self.enemy_states[enemy_id]["health_level"] for enemy_id in range(self.enemy_num)]

        if sum(ally_alive) == 0 or sum(enemy_alive) == 0:
            if sum(ally_alive) == 0:
                # print("我方全部死亡，进行重置")
                game_over = True
            else:
                # print("敌方被全部消灭，进行重置")
                game_over = True
        return game_over
    def get_low_single_done(self, agent_id):
        """得到智能体是否终止"""
        done = False
        own_state = self.ally_states[agent_id]
        if own_state["health_level"] == 0:
            done = True
        return done

    def apply_action(self, high_actions, attack_actions, escape_actions):
        """根据上层选择的元命令，执行下层动作"""
        true_actions = torch.zeros_like(attack_actions)

        for ally_id in range(self.ally_num):
            if high_actions[ally_id][0] == 0:
                true_actions[ally_id] = attack_actions[ally_id]
            else:
                true_actions[ally_id] = escape_actions[ally_id]

        for i in range(true_actions.shape[0]):
            agent_id_state = self.ally_states[i]
            if agent_id_state["health_level"] > 0:
                exe_action = self.normalize_action(true_actions[i])
                df.set_plane_pitch(self.Plane_ID_ally[i], float(exe_action[0]))
                df.set_plane_roll(self.Plane_ID_ally[i], float(exe_action[1]))
                df.set_plane_yaw(self.Plane_ID_ally[i], float(exe_action[2]))
                df.set_plane_linear_speed(self.Plane_ID_ally[i], self.initial_speed)

        df.update_scene()

    def normalize_action(self, action):
        """将离散动作表示为连续值"""
        norm_act = np.zeros(action.shape)  # 初始化一个和输入动作同形状的数组
        norm_act[0] = action[0] * 2. / (self.attack_action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.attack_action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.attack_action_space.nvec[2] - 1.) - 1.

        return norm_act

    def get_low_attack_single_obs(self, agent_id, high_actions, ally_states, enemy_states):
        """得到特定智能体的攻击观测"""
        own_feat = np.zeros((17,), dtype=np.float32)  # 自身信息
        ally_feats = np.zeros((self.ally_num - 1, 15), dtype=np.float32)  # 队友信息
        enemy_feats = np.zeros((self.enemy_num, 17), dtype=np.float32)  # 敌方信息

        own_state = ally_states[agent_id]

        if own_state["health_level"] > 0:
            # 如果该无人机健康值不为0
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

            # 自己编码后的元命令
            own_command = high_actions[agent_id]
            own_target = own_command[1]
            own_target_encode = self.one_hot_encode_target(own_target)

            own_feat[0:17] = np.array([norm_own_x, norm_own_y, norm_own_z, norm_own_vx, norm_own_vy,
                                       norm_own_vz, pitch_sin, pitch_cos, yaw_sin, yaw_cos, roll_sin,
                                       roll_cos, norm_speed, health_level] + own_target_encode.tolist())

            # 队友信息
            ally_idx = 0
            for ally_id in range(self.ally_num):
                if ally_id == agent_id:
                    continue
                ally_id_state = ally_states[ally_id]

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

                    # 友方要攻击的目标
                    ally_same_target = 0  # 表示该友方目标是否与我的相同
                    ally_id_command = high_actions[ally_id]
                    if np.array_equal(own_command, ally_id_command):
                        # 表示该友方与我方目标相同
                        ally_same_target = 1

                    ally_feats[ally_idx, 0:15] = np.array([norm_ally_relative_x, norm_ally_relative_y,
                                                           norm_ally_relative_z, norm_ally_vx, norm_ally_vy,
                                                           norm_ally_vz, np.sin(ally_ATA), np.cos(ally_ATA),
                                                           np.sin(ally_AA), np.cos(ally_AA), np.sin(ally_HA),
                                                           np.cos(ally_HA), norm_ally_R, health_level, ally_same_target])
                ally_idx += 1

            # 敌方信息
            for enemy_id in range(self.enemy_num):
                enemy_id_state = enemy_states[enemy_id]

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

    def fine_nearst_ally(self, agent_id, ally_states):
        """查找与自己距离最近的己方智能体的编号"""
        own_state = ally_states[agent_id]
        current_min_dis = np.inf
        current_min_ally_id = -1

        own_x, own_y, own_z = own_state["position"]

        for ally_id in range(self.ally_num):
            if ally_id == agent_id:
                continue
            ally_id_state = ally_states[ally_id]

            if ally_id_state["health_level"] > 0:
                ally_x, ally_y, ally_z = ally_id_state["position"]
                ally_relative_pos = np.array([ally_x - own_x, ally_y - own_y, ally_z - own_z])
                distance = np.linalg.norm(ally_relative_pos)
                if distance < current_min_dis:
                    current_min_dis = distance
                    current_min_ally_id = ally_id

        return current_min_ally_id, current_min_dis

    def get_low_escape_single_obs(self, agent_id, high_actions, ally_states, enemy_states):
        """得到特定智能体的躲避观测"""
        own_feat = np.zeros((17,), dtype=np.float32)  # 自身信息
        ally_feats = np.zeros((1, 14), dtype=np.float32)  # 与其最近的队友信息
        enemy_feats = np.zeros((self.enemy_num, 18), dtype=np.float32)  # 敌方信息

        own_state = ally_states[agent_id]

        if own_state["health_level"] > 0:
            # 如果该无人机健康值不为0
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

            # 自己编码后的元命令
            own_command = high_actions[agent_id]
            own_target = own_command[1]  # 自己的目标组别
            own_target_encode = self.one_hot_encode_target(own_target)

            own_feat[0:17] = np.array([norm_own_x, norm_own_y, norm_own_z, norm_own_vx, norm_own_vy,
                                       norm_own_vz, pitch_sin, pitch_cos, yaw_sin, yaw_cos, roll_sin,
                                       roll_cos, norm_speed, health_level] + own_target_encode.tolist())

            # 最近队友的信息
            nearst_ally_id, _ = self.fine_nearst_ally(agent_id, ally_states)
            if nearst_ally_id != -1:
                # 表示有最近己方
                ally_id_state = ally_states[nearst_ally_id]
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

                ally_feats[0, 0:14] = np.array([norm_ally_relative_x, norm_ally_relative_y,
                                                norm_ally_relative_z, norm_ally_vx, norm_ally_vy,
                                                norm_ally_vz, np.sin(ally_ATA), np.cos(ally_ATA),
                                                np.sin(ally_AA), np.cos(ally_AA), np.sin(ally_HA),
                                                np.cos(ally_HA), norm_ally_R, health_level])

            # 敌方信息
            for enemy_id in range(self.enemy_num):
                enemy_id_state = enemy_states[enemy_id]

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

                    # 该敌方是否为自己目标的标志
                    if own_target == enemy_id_group:
                        is_own_target = 1
                    else:
                        is_own_target = 0

                    enemy_feats[enemy_id, 0:18] = np.array([norm_enemy_relative_x, norm_enemy_relative_y,
                                                            norm_enemy_relative_z, norm_enemy_vx, norm_enemy_vy,
                                                            norm_enemy_vz, np.sin(enemy_ATA), np.cos(enemy_ATA),
                                                            np.sin(enemy_AA), np.cos(enemy_AA), np.sin(enemy_HA),
                                                            np.cos(enemy_HA), norm_enemy_R, health_level,
                                                            is_own_target] +
                                                           enemy_hot_encode.tolist())

        own_obs = np.concatenate([own_feat.flatten(), ally_feats.flatten(), enemy_feats.flatten()])

        return own_obs

    #####底层累积奖励相关#####
    def get_low_single_reward(self, agent_id):
        """
        得到单个智能体的奖励
        出界惩罚
        1.击毁敌方的奖励（按照共同奖励来给）
        2.步数惩罚

        """
        own_reward = 0

        own_state = self.ally_states[agent_id]

        if own_state["health_level"] > 0:
            reward_out_bound = self.get_reward_out_bound(agent_id, own_state)  # 出界惩罚
            reward_step_punish = self.get_reward_step_punish()  # 步数惩罚
            reward_attack = self.get_reward_attack(agent_id, own_state, self.enemy_states)  # 攻击或被攻击的奖励

            own_reward = 1.0 * reward_out_bound + 1.0 * reward_step_punish + 1.0 * reward_attack

        return own_reward

    def get_reward_attack(self, agent_id, own_state, enemy_states):
        """击毁奖励和被击毁惩罚"""
        reward_attack = 0
        punish_be_attacked = 0  # 被击毁的惩罚

        own_pos_vector = own_state["position"]
        own_move_vector = own_state["move_vector"]

        # 判断是否与每一个敌方发生攻击行为
        for enemy_id in range(self.enemy_num):
            enemy_id_state = enemy_states[enemy_id]
            if enemy_id_state["health_level"] > 0:
                # 判断二者的相对角度
                enemy_pos_vector = enemy_id_state["position"]
                enemy_move_vector = enemy_id_state["move_vector"]

                enemy_AO, enemy_TA, enemy_R = self.get_AO_TA_R(enemy_pos_vector, enemy_move_vector,
                                                               own_pos_vector, own_move_vector)
                own_AO, own_TA, own_R = self.get_AO_TA_R(own_pos_vector, own_move_vector,
                                                         enemy_pos_vector, enemy_move_vector)
                # 敌方攻击
                if abs(enemy_AO) < self.task_hyper.attack_angle and enemy_R < self.task_hyper.attack_distance:
                    # 我方在敌方攻击区域内
                    if self.ally_being_attack[agent_id] == False and self.enemy_launch_attack[enemy_id] == False:
                        # 我方未被攻击并且敌方还未发射导弹
                        if enemy_R < 200:
                            hit_prob = 0
                        elif enemy_R < self.task_hyper.attack_distance:
                            hit_prob = np.exp(-enemy_R / (self.task_hyper.attack_distance + 500))
                        else:
                            hit_prob = 0

                        hit = random.random() < hit_prob
                        self.ally_being_attack[agent_id] = hit

                        if hit:
                            # 表示被敌方击杀
                            punish_be_attacked -= 12
                            print("我方被击杀")
                            self.end_high_action = True
                        else:
                            punish_be_attacked -= 1.2
                            self.end_high_action = True

                        self.enemy_launch_attack[enemy_id] = True

                # 我方攻击
                if abs(own_AO) < self.task_hyper.attack_angle and own_R < self.task_hyper.attack_distance:
                    # 敌方在我方攻击区域内
                    if self.enemy_being_attack[enemy_id] == False and self.ally_launch_attack[agent_id] == False:
                        # 敌方未被攻击并且我方还未发射导弹
                        if own_R < 200:
                            hit_prob = 0
                        elif own_R < self.task_hyper.attack_distance:
                            hit_prob = np.exp(-own_R / (self.task_hyper.attack_distance + 500))
                        else:
                            hit_prob = 0
                        hit = random.random() < hit_prob

                        self.enemy_being_attack[enemy_id] = hit

                        if hit:
                            # 表示成功击杀敌方
                            reward_attack += 30
                            print("击杀敌方")
                            self.end_high_action = True
                            self.reward_attack_group[agent_id] += reward_attack
                        else:
                            punish_be_attacked += 2

                        self.ally_launch_attack[agent_id] = True

        return punish_be_attacked

    def get_reward_out_bound(self, agent_id, own_state):
        """得到智能体的出界惩罚"""
        reward_out_bound = 0

        pos_vector = own_state["position"]
        x, y, z = pos_vector
        if abs(x) > NormStates["bound_radius"] or abs(z) > NormStates["bound_radius"]:
            reward_out_bound -= self.task_hyper.out_bound_punish
            self.ally_being_attack[agent_id] = True  # 出界血量变为0
            self.end_high_action = True
            # print(f"飞机{agent_id}水平面方向出界")

        if y > NormStates["bound_altitude_max"] or y < NormStates["bound_altitude_min"]:
            reward_out_bound -= self.task_hyper.out_bound_punish
            self.ally_being_attack[agent_id] = True  # 出界血量变为0
            self.end_high_action = True
            # print(f"飞机{agent_id}飞出限定高度")

        return reward_out_bound

    def get_reward_step_punish(self):
        """步数惩罚"""
        return self.task_hyper.step_punish

    ######敌方策略相关#######
    def set_enemy_policy(self, enemy_states):
        """敌方直飞"""
        for enemy_id in range(self.enemy_num):
            enemy_state = enemy_states[enemy_id]
            if enemy_state["health_level"] > 0:
                df.set_plane_linear_speed(self.Plane_ID_enemy[enemy_id], 100)
                # pass
        df.update_scene()

    def reset_machine(self):
        """重置内置函数"""
        # 定义我方飞机的固定初始位置（单位：米）
        fixed_ally_positions = {
            1: {'x': 0, 'y': 4000, 'z': 0},
            2: {'x': 300, 'y': 4000, 'z': 300},
            3: {'x': 300, 'y': 4000, 'z': -300},
            4: {'x': -300, 'y': 4000, 'z': 300},
            5: {'x': -300, 'y': 4000, 'z': -300},
            6: {'x': 0, 'y': 4500, 'z': 0},
        }
        for ally_id in range(1, self.ally_num + 1):
            position = fixed_ally_positions[ally_id]
            initial_x = position['x']
            initial_y = position['y']
            initial_z = position['z']
            initial_yaw = np.random.uniform(-pi, pi)  # 初始状态角度

            # 重置我方飞机位置和状态
            plane_id = f"ally_{ally_id}"
            df.rearm_machine(plane_id)
            df.reset_machine(plane_id)
            df.reset_machine_matrix(plane_id, initial_x, initial_y, initial_z, 0, initial_yaw, 0)
            df.retract_gear(plane_id)
            df.set_plane_linear_speed(plane_id, self.initial_speed)

        # 随机分组敌机
        self.randomize_enemy_groups()

        # 定义敌方飞机的基准位置（单位：米）
        base_positions = [
            (4500, 5000),  # 组1的基准位置
            (-8000, 0),  # 组2的基准位置
            (7000, -5000)  # 组3的基准位置
        ]

        # 重置敌方yaw记录
        self.enemy_initial_yaws = {}

        for group_id, group in enumerate(self.enemy_initial_group):
            if not group:
                continue  # 跳过空组
            base_x, base_z = base_positions[group_id]
            # 为每个组分配一个随机的偏移量中心
            group_center_offset = (random.uniform(-500, 500), random.uniform(-500, 500))

            for enemy_id in group:
                plane_id = f"ennemy_{enemy_id + 1}"
                # 将敌机分布在组中心附近
                offset_x = random.uniform(-500, 500) + group_center_offset[0]
                offset_z = random.uniform(-500, 500) + group_center_offset[1]
                enemy_x = base_x + offset_x
                enemy_z = base_z + offset_z
                enemy_y = 5000  # 固定高度，根据需要调整
                enemy_yaw = self.calculate_yaw(enemy_x, enemy_z)

                # 记录敌机yaw
                self.enemy_initial_yaws[plane_id] = enemy_yaw

                df.rearm_machine(plane_id)
                df.reset_machine(plane_id)
                df.reset_machine_matrix(plane_id, enemy_x, enemy_y, enemy_z, 0, enemy_yaw, 0)
                df.retract_gear(plane_id)
                df.set_plane_linear_speed(plane_id, 150)
                df.activate_ia(plane_id)

        df.update_scene()

    ########功能性函数相关######
    def get_all_plane_states(self):
        """得到这一步所有智能体的观测信息"""
        ally_states = []
        enemy_states = []
        for plane_id in self.Plane_ID_ally:
            ally_states.append(df.get_plane_state(plane_id))
        for plane_id in self.Plane_ID_enemy:
            enemy_states.append(df.get_plane_state(plane_id))
        return ally_states, enemy_states

    def load_low_model(self):
        """加载预训练模型参数（核心路径已移除，需自行提供）。"""
        model_config = _load_json_config("model_paths.json") or {}
        self.attack_model_path = os.getenv("H2CF_ATTACK_MODEL_PATH") or model_config.get("attack_model_path")
        self.escape_model_path = os.getenv("H2CF_ESCAPE_MODEL_PATH") or model_config.get("escape_model_path")

        if not self.attack_model_path or not self.escape_model_path:
            raise RuntimeError(
                "Core model paths were removed for the public release. "
                "Set H2CF_ATTACK_MODEL_PATH/H2CF_ESCAPE_MODEL_PATH or add configs/model_paths.json "
                "with 'attack_model_path' and 'escape_model_path'."
            )

        self.attack_policy.load_state_dict(torch.load(self.attack_model_path, map_location=torch.device('cpu')))
        self.escape_policy.load_state_dict(torch.load(self.escape_model_path, map_location=torch.device('cpu')))
        self.attack_policy.eval()
        self.escape_policy.eval()

    def connect_harfang(self):
        """
        主要负责连接模拟器，以及敌方和我方的一些信息
        """
        sim_config = _load_json_config("sim_connection.json") or {}
        host = os.getenv("H2CF_SIM_HOST") or (sim_config.get("host") if sim_config else None)
        if not host:
            raise RuntimeError(
                "Core simulator endpoint removed for the public release. "
                "Set H2CF_SIM_HOST or provide configs/sim_connection.json with field 'host'."
            )
        df.connect(host, self.port)
        time.sleep(2)
        df.disable_log()
        df.set_client_update_mode(True)
        renderless = bool(sim_config.get("renderless", False)) if sim_config else False
        df.set_renderless_mode(renderless)

        for i in range(1, self.ally_num + 1):
            self.Plane_ID_ally.append("ally_" + str(i))

        for i in range(1, self.enemy_num + 1):
            self.Plane_ID_enemy.append("ennemy_" + str(i))

    def seed(self, seed=None):
        """
        设置环境的种子
        """
        # self.np_random, seed = seeding.np_random(seed)
        # 使用 Gym 的 seeding 模块生成随机数生成器
        self.np_random, seed = seeding.np_random(seed)

        # 设置 Python 内置 random 模块的种子
        random.seed(seed)

        # 设置 NumPy 的随机种子
        np.random.seed(seed)
        return [seed]

    def show_plane_id(self, high_actions):
        """显示飞机标号"""
        for index, ally_id in enumerate(self.Plane_ID_ally):
            own_state = self.ally_states[index]
            df.display_vector(own_state["position"], own_state["move_vector"], str(high_actions[index]), [0, 0], [1, 0, 0, 1], 0.02)
        for index, enemy_id in enumerate(self.Plane_ID_enemy):
            enemy_state = self.enemy_states[index]
            df.display_vector(enemy_state["position"], enemy_state["move_vector"], str(self.get_enemy_group(index)), [0, 0], [0, 0, 1, 1], 0.02)
        df.update_scene()

    def clear_death(self):
        """每一步清除被攻击的智能体"""
        for ally_id in range(self.ally_num):
            ally_id_state = self.ally_states[ally_id]
            if ally_id_state["health_level"] > 0 and self.ally_being_attack[ally_id]:
                df.set_health(self.Plane_ID_ally[ally_id], 0)
                self.current_episode_ally_losses += 1  # 更新战损数


        for enemy_id in range(self.enemy_num):
            enemy_id_state = self.enemy_states[enemy_id]
            if enemy_id_state["health_level"] > 0 and self.enemy_being_attack[enemy_id]:
                df.set_health(self.Plane_ID_enemy[enemy_id], 0)
                self.current_episode_enemy_kills += 1  # 更新击毁数

        df.update_scene()

    ####  额外函数  ####
    def one_hot_encode_target(self, target_id):
        """对目标进行独热编码"""
        # 创建一个全零数组，长度等于敌人总数
        one_hot_encoded = np.zeros(self.enemy_group_num)
        # 设置对应的索引为1
        one_hot_encoded[target_id] = 1
        return one_hot_encoded

    def get_enemy_group(self, enemy_id):
        """得到敌方所属的组别序号"""
        for group_index, group in enumerate(self.enemy_initial_group):
            if enemy_id in group:
                return group_index
        print("未找到相应敌方所属组别")
        return None

    def get_ATA_AA_HA_R(self, own_pos_vector, own_move_vector, other_pos_vector, other_move_vector):
        """
        根据相对位置计算ATA， AA, HA, R
        ATA:在水平面上，我方速度向量与相对位置向量的夹角，-pi到pi
        AA：敌方速度向量与相对位置向量的夹角,-pi到pi
        HA：高度角,-pi/2到pi/2
        R：距离
        """
        own_x, own_y, own_z = own_pos_vector
        own_vx, own_vy, own_vz = own_move_vector
        other_x, other_y, other_z = other_pos_vector
        other_vx, other_vy, other_vz = other_move_vector

        own_v_h = np.array([own_vx, own_vz])  # 自己在水平面上的速度方向
        relative_pos_h = np.array([other_x - own_x, other_z - own_z])  # 在水平面上与另一个的相对位置
        other_v_h = np.array([other_vx, other_vz])  # 另一个在水平面上的速度方向

        ATA = self.angle_between_vectors(own_v_h, relative_pos_h)  # 天线方向角
        AA = self.angle_between_vectors(other_v_h, relative_pos_h)  # 目标角

        relative_pos = np.array([other_x - own_x, other_y - own_y, other_z - own_z])
        R = np.linalg.norm(relative_pos)  # 距离
        HA = np.arcsin(np.clip(relative_pos[1] / R, -1.0, 1.0))  # 高度角

        return ATA, AA, HA, R

    def get_AO_TA_R(self, own_pos_vector, own_move_vector, other_pos_vector, other_move_vector):
        """计算攻击角，尾追角、距离"""
        own_x, own_y, own_z = own_pos_vector
        own_vx, own_vy, own_vz = own_move_vector
        other_x, other_y, other_z = other_pos_vector
        other_vx, other_vy, other_vz = other_move_vector

        delta_x, delta_y, delta_z = other_x - own_x, other_y - own_y, other_z - own_z
        R = np.linalg.norm([delta_x, delta_y, delta_z])
        own_v = np.linalg.norm([own_vx, own_vy, own_vz])
        other_v = np.linalg.norm([other_vx, other_vy, other_vz])

        proj_dist = delta_x * own_vx + delta_y * own_vy + delta_z * own_vz
        AO = np.arccos(np.clip(proj_dist / (R * own_v + 1e-8), -1, 1))
        proj_dist = delta_x * other_vx + delta_y * other_vy + delta_z * other_vz
        TA = np.arccos(np.clip(proj_dist / (R * other_v + 1e-8), -1, 1))

        return AO, TA, R

    def angle_between_vectors(self, v1, v2):
        """计算两个向量之间的夹角，结果范围在 -pi 到 pi."""
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-8)  # 将 v1 单位化
        unit_v2 = v2 / np.linalg.norm(v2 + 1e-8)  # 将 v2 单位化

        dot_product = np.dot(unit_v1, unit_v2)  # 计算点积
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 计算角度

        # 确定夹角的方向（正为逆时针，负为顺时针）
        cross_product = np.cross(unit_v1, unit_v2)
        if cross_product < 0:
            angle = -angle

        return angle

    def get_theta_sin_cos(self, theta):
        """得到一个角度的sin和cos"""
        return np.sin(theta), np.cos(theta)

    def calculate_yaw(self, x, z):
        """等级2下：计算航向角"""
        return math.atan2(-x, -z)

    def close(self):
        """关闭环境"""
        pass

    def report_metrics(self):
        """打印或保存当前的指标统计"""
        if self.total_episodes == 0:
            print("尚未完成任何回合。")
            return

        # 计算任务完成率
        task_completion_rate = self.completed_episodes / self.total_episodes * 100

        # 计算战损率（每回合平均损失的我方飞机数）
        average_ally_loss_rate = self.total_ally_losses / self.total_episodes

        # 计算击毁率（每回合平均击毁的敌方飞机数）
        average_enemy_kill_rate = self.total_enemy_kills / self.total_episodes

        # 计算平均消耗时间（每回合平均步数）
        average_time_consumption = self.total_time_consumption / self.total_episodes

        print(f"总回合数: {self.total_episodes}")
        print(f"任务完成回合数: {self.completed_episodes}")
        print(f"超时回合数: {self.time_out_episodes}")
        print(f"任务完成率: {task_completion_rate:.2f}%")
        print(f"平均战损率: {average_ally_loss_rate:.2f}")
        print(f"平均击毁率: {average_enemy_kill_rate:.2f}")
        print(f"平均消耗时间: {average_time_consumption:.2f} 步")

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        """
        主要用来控制环境是否显示在tacview上
        """
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True

            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * 0.08
                f.write(f"#{timestamp:.2f}\n")
                for ally_id in range(self.task.ally_num):
                    ally_id_state = df.get_plane_state(self.task.Plane_ID_ally[ally_id])
                    x1 = ally_id_state["position"][0] / 100000 + 20
                    y1 = ally_id_state["position"][2] / 100000 + 20
                    z1 = ally_id_state["position"][1] * 1.5 + 3000
                    roll1 = -ally_id_state["roll_attitude"]
                    pitch1 = ally_id_state["pitch_attitude"]
                    heading1 = ally_id_state["heading"]
                    health = ally_id_state["health_level"]

                    if health == 0:
                        data = (
                            f"A0{ally_id + 1}00, T={x1}|{y1}|{z1}|{roll1}|{pitch1}|{heading1},Name=F16,Color=Black\n")
                    else:
                        data = (
                            f"A0{ally_id + 1}00, T={x1}|{y1}|{z1}|{roll1}|{pitch1}|{heading1},Name=F16,Color=Red\n")
                    f.write(data)

                for enemy_id in range(self.task.enemy_num):
                    enemy_id_state = df.get_plane_state(self.task.Plane_ID_enemy[enemy_id])

                    x1 = enemy_id_state["position"][0] / 100000 + 20
                    y1 = enemy_id_state["position"][2] / 100000 + 20
                    z1 = enemy_id_state["position"][1] * 1.5 + 3000
                    roll1 = -enemy_id_state["roll_attitude"]
                    pitch1 = enemy_id_state["pitch_attitude"]
                    heading1 = enemy_id_state["heading"]
                    health = enemy_id_state["health_level"]
                    if health == 0:
                        data = (
                            f"B0{enemy_id + 1}00, T={x1}|{y1}|{z1}|{roll1}|{pitch1}|{heading1},Name=F16,Color=Yellow\n")
                    else:
                        data = (
                            f"B0{enemy_id + 1}00, T={x1}|{y1}|{z1}|{roll1}|{pitch1}|{heading1},Name=F16,Color=Blue\n")
                    f.write(data)





