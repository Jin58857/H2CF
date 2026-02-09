"""
躲避子策略训练：躲避对应的敌方，敌方采用专家策略追逐
"""
import os
import sys
import random
import copy
import numpy as np
from gym import spaces
from math import pi
import math

sys.path.append(os.path.dirname(os.path.dirname((os.path.realpath(__file__)))))

import utils.dogfight_client as df
from utils.Constants import NormStates


class TaskConfig(object):
    def __init__(self):
        # 属性参数相关
        self.attack_angle = pi / 30  # 有效攻击角度
        self.attack_distance = 1000  # 有效攻击距离
        self.hyper_feature_safe_distance = 50  # 飞机之间的碰撞距离为50m

        # 奖励参数相关
        self.hyper_reward_out_bound = 60  # 出界惩罚
        self.hyper_reward_distance_punish = 2  # 安全距离惩罚

        # 躲避命令下相关参数
        self.danger_range = 2500
        self.safe_range = 5000


class MultiCombatTask(object):
    def __init__(self):
        # 环境初始变量设置
        self.ally_num = 6
        self.enemy_num = 6
        self.Plane_ID_ally = []
        self.Plane_ID_enemy = []

        self.initial_speed = 150  # 飞机的初始速度

        self.task_hyper = TaskConfig()  # 与参数相关的类
        self.all_ally_dones = np.full((self.ally_num, 1), False)

        # 指定状态空间和动作空间
        self.load_observation_space()
        self.load_action_space()

        # 与飞机躲避目标相关
        self.enemy_group_num = 3  # 将敌方分为3组
        self.command_target = [enemy_id for enemy_id in range(self.enemy_group_num)]
        self.ally_command = {}  # 用来储存原始分配的目标
        self.encoded_commands = {}  # 用来储存编码后的目标
        self.reset_command_variables()  # 初始化随机分配

        # 与击毁机制有关
        self.ally_launch_attack = [False] * self.ally_num
        self.enemy_launch_attack = [False] * self.enemy_num
        self.ally_being_attack = [False] * self.ally_num
        self.enemy_being_attack = [False] * self.enemy_num

        # 与分配的目标之间的最小距离
        self.last_target_min_dis = [np.inf] * self.ally_num

    @property
    def num_agents(self):
        """这里只控制我方智能体"""
        return self.ally_num

    def load_observation_space(self):
        """ 指定观测空间、共享观测空间"""
        self.obs_length = 17 + (self.ally_num - 1) * 14 + self.enemy_num * 17
        self.share_obs_length = 17 + (self.ally_num - 1) * 17 + self.enemy_num * 17
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.share_obs_length,))

    def load_action_space(self):
        """指定动作空间"""
        self.action_space = spaces.MultiDiscrete([11, 11, 11])

    def reset_task(self):
        """与任务相关的变量重置"""
        self.all_ally_dones = np.full((self.ally_num, 1), False)

        # 与击毁机制有关
        self.ally_launch_attack = [False] * self.ally_num
        self.enemy_launch_attack = [False] * self.enemy_num
        self.ally_being_attack = [False] * self.ally_num
        self.enemy_being_attack = [False] * self.enemy_num

        # 重置元命令相关变量
        self.reset_command_variables()

        # 与分配的目标之间的最小距离
        self.last_target_min_dis = [np.inf] * self.ally_num

    def get_obs(self, agent_id, ally_states, enemy_states):
        """
        与躲避相关的观测信息
        自身信息：
        队友信息：
        敌方信息：
        """
        own_feat = np.zeros((17, ), dtype=np.float32)  # 自身信息
        ally_feats = np.zeros((self.ally_num - 1, 14), dtype=np.float32)  # 队友信息
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
            own_target_group = self.ally_command[agent_id]
            own_command = self.encoded_commands[agent_id]

            own_feat[0:17] = np.array([norm_own_x, norm_own_y, norm_own_z, norm_own_vx, norm_own_vy,
                                       norm_own_vz, pitch_sin, pitch_cos, yaw_sin, yaw_cos, roll_sin,
                                       roll_cos, norm_speed, health_level] + own_command.tolist())

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

                    ally_feats[ally_idx, 0:14] = np.array([norm_ally_relative_x, norm_ally_relative_y,
                                                           norm_ally_relative_z, norm_ally_vx, norm_ally_vy,
                                                           norm_ally_vz, np.sin(ally_ATA), np.cos(ally_ATA),
                                                           np.sin(ally_AA), np.cos(ally_AA), np.sin(ally_HA),
                                                           np.cos(ally_HA), norm_ally_R, health_level])
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

    def get_share_obs(self, agent_id, ally_states, enemy_states):
        """
        得到对应的全局观测
        """
        own_feat = np.zeros((17, ), dtype=np.float32)  # 自身信息
        ally_feats = np.zeros((self.ally_num - 1, 17), dtype=np.float32)  # 队友信息
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
            own_command = self.encoded_commands[agent_id]

            own_feat[0:17] = np.array([norm_own_x, norm_own_y, norm_own_z, norm_own_vx, norm_own_vy,
                                       norm_own_vz, pitch_sin, pitch_cos, yaw_sin, yaw_cos, roll_sin,
                                       roll_cos, norm_speed, health_level] + own_command.tolist())

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

                    # 友方目标
                    ally_command = self.encoded_commands[ally_id]

                    ally_feats[ally_idx, 0:17] = np.array([norm_ally_relative_x, norm_ally_relative_y,
                                                           norm_ally_relative_z, norm_ally_vx, norm_ally_vy,
                                                           norm_ally_vz, np.sin(ally_ATA), np.cos(ally_ATA),
                                                           np.sin(ally_AA), np.cos(ally_AA), np.sin(ally_HA),
                                                           np.cos(ally_HA), norm_ally_R, health_level] +
                                                          ally_command.tolist())
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

    def get_reward(self, agent_id, ally_states, enemy_states):
        """
        躲避任务的奖励设计
        出界惩罚
        每一步的存活奖励
        与其他飞机的安全距离惩罚
        击毁惩罚

        在目标危险区域内的固定惩罚
        远离目标的奖励

        """
        own_reward = 0
        own_state = ally_states[agent_id]

        if own_state["health_level"] > 0:
            reward_out_bound = self.get_reward_out_bound(agent_id, own_state)  # 智能体出界惩罚
            reward_safe_distance = self.get_reward_safe_distance(agent_id, own_state, ally_states, enemy_states)  # 安全距离惩罚
            reward_survival = self.get_reward_survival()  # 智能体每一步的存活奖励

            reward_danger_range = self.get_reward_danger_range(agent_id, ally_states, enemy_states)  # 在敌方危险区域的惩罚和安全距离的奖励

            reward_attack = self.get_reward_attack(agent_id, own_state, ally_states, enemy_states)  # 击毁奖励

            # print("reward_out_bound:", reward_out_bound,
            #       "reward_safe_distance:", reward_safe_distance,
            #       "reward_survival:", reward_survival,
            #       "reward_danger_range:", reward_danger_range,
            #       "reward_attack:", reward_attack)

            own_reward = (1.0 * reward_out_bound + 1.0 * reward_safe_distance +
                          1.0 * reward_survival + 1.0 * reward_danger_range + 1.0 * reward_attack)

        return own_reward

    def get_reward_attack(self, agent_id, own_state, ally_states, enemy_states):
        """
        攻击和被攻击的基础奖励,利用什么角度来判断是否攻击到是个问题
        """
        reward = 0

        own_pos_vector = own_state["position"]
        own_move_vector = own_state["move_vector"]

        own_command_target_group_id = self.ally_command[agent_id]  # 表示我的目标组别

        # 判断是否与每一个enemy产生攻击行为
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
                        enemy_id_group = self.get_enemy_group(enemy_id)

                        if hit:
                            # 表示被敌方击杀
                            if own_command_target_group_id == enemy_id_group:
                                reward -= 20
                                print("我方被应该躲避的敌方击杀")
                            else:
                                reward -= 10
                                print("我方被其他敌方击杀")
                        else:
                            reward -= 3
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
                        enemy_id_group = self.get_enemy_group(enemy_id)

                        if hit:
                            # 表示成功击杀敌方
                            if own_command_target_group_id == enemy_id_group:
                                # 击杀了对应的目标
                                reward -= 25
                                print("我方击杀了应该躲避的敌方")
                            else:
                                reward += 10
                                print("我方击杀了其他的敌方")

                            # 击杀之后重新分配组别
                            self.after_attack_assign_command(enemy_id, enemy_id_group)

                        else:
                            # 未成功击杀
                            if own_command_target_group_id == enemy_id_group:
                                # 表示攻击命令下，没能成功击杀
                                reward -= 1
                            else:
                                reward += 0.5

                        self.ally_launch_attack[agent_id] = True

        return reward

    def get_reward_danger_range(self, agent_id, ally_states, enemy_states):
        """在敌方危险区域内的惩罚和奖励"""
        reward_danger_range = 0
        current_min_dis = np.inf

        own_state = ally_states[agent_id]
        own_x, own_y, own_z = own_state["position"]

        own_target_group = self.ally_command[agent_id]

        for enemy_id in self.command_groups_running[own_target_group]:
            enemy_state = enemy_states[enemy_id]
            enemy_x, enemy_y, enemy_z = enemy_state["position"]
            if enemy_state["health_level"] > 0:
                relative_pos = np.array([enemy_x - own_x, enemy_y - own_y, enemy_z - own_z])
                distance = np.linalg.norm(relative_pos)
                if distance < current_min_dis:
                    current_min_dis = distance

        if current_min_dis < self.task_hyper.danger_range:
            reward_danger_range -= 0.05

        if current_min_dis > self.task_hyper.safe_range and current_min_dis != np.inf:
            reward_danger_range += 0.05

        return reward_danger_range

    def get_reward_survival(self):
        """得到该智能体每一步的存活奖励"""
        survival_reward = 0.01
        return survival_reward

    def get_reward_out_bound(self, agent_id, own_state):
        """得到智能体的出界惩罚"""
        reward_out_bound = 0

        pos_vector = own_state["position"]
        x, y, z = pos_vector
        if abs(x) > NormStates["bound_radius"] or abs(z) > NormStates["bound_radius"]:
            reward_out_bound -= self.task_hyper.hyper_reward_out_bound
            self.ally_being_attack[agent_id] = True  # 出界血量变为0
            print(f"飞机{agent_id}水平面方向出界")

        if y > NormStates["bound_altitude_max"] or y < NormStates["bound_altitude_min"]:
            reward_out_bound -= self.task_hyper.hyper_reward_out_bound
            self.ally_being_attack[agent_id] = True  # 出界血量变为0
            print(f"飞机{agent_id}飞出限定高度")

        return reward_out_bound

    def get_reward_safe_distance(self, agent_id, own_state, ally_states, enemy_states):
        """与其他飞机的安全距离惩罚"""
        safe_reward = 0
        own_pos_vector = own_state["position"]

        # 与队友的安全距离惩罚
        for ally_id in range(self.ally_num):
            if ally_id == agent_id:
                continue
            ally_id_state = ally_states[ally_id]
            ally_pos_vector = ally_id_state["position"]
            delta_x, delta_y, delta_z = (ally_pos_vector[0] - own_pos_vector[0], ally_pos_vector[1] - own_pos_vector[1],
                                        ally_pos_vector[2] - own_pos_vector[2])
            distance = np.linalg.norm([delta_x, delta_y, delta_z])

            if distance < self.task_hyper.hyper_feature_safe_distance:
                safe_reward -= self.task_hyper.hyper_reward_distance_punish

        # 与敌方的安全距离惩罚
        for enemy_id in range(self.enemy_num):
            enemy_id_state = enemy_states[enemy_id]
            enemy_pos_vector = enemy_id_state["position"]
            delta_x, delta_y, delta_z = (enemy_pos_vector[0] - own_pos_vector[0], enemy_pos_vector[1] - own_pos_vector[1],
                                         enemy_pos_vector[2] - own_pos_vector[2])
            distance = np.linalg.norm([delta_x, delta_y, delta_z])

            if distance < self.task_hyper.hyper_feature_safe_distance:
                safe_reward -= self.task_hyper.hyper_reward_distance_punish

        return safe_reward



    def get_done(self, agent_id, ally_states):
        """判断该智能体是否参与"""
        done = False
        own_state = ally_states[agent_id]
        if own_state["health_level"] == 0:
            # 血量为0终止
            done = True
        else:
            # 血量不为0时
            done = self.done_out_bound(own_state)  # 出界终止
        return done

    def done_out_bound(self, own_state):
        """出界终止"""
        x, y, z = own_state["position"]
        if abs(x) > NormStates["bound_radius"] or abs(z) > NormStates["bound_radius"]:
            return True
        if y > NormStates["bound_altitude_max"] or y < NormStates["bound_altitude_min"]:
            return True
        return False

    def done_win_game(self, ally_states, enemy_states):
        """一方全部死亡终止"""
        game_over = False
        ally_alive = [ally_states[ally_id]["health_level"] for ally_id in range(self.ally_num)]
        enemy_alive = [enemy_states[enemy_id]["health_level"] for enemy_id in range(self.enemy_num)]

        if sum(ally_alive) == 0 or sum(enemy_alive) == 0:
            if sum(ally_alive) == 0:
                print("我方全部死亡，进行重置")
                game_over = True
            else:
                print("敌方被全部消灭，进行重置")
                game_over = True
        return game_over


    ### 动作应用 ###
    def apply_action(self, actions, ally_states):
        """
        三个舵偏+油门偏移[0.1, 1]
        """
        for i in range(actions.shape[0]):
            agent_i_state = ally_states[i]
            if agent_i_state["health_level"] == 0:
                continue
            exe_action = self.normalize_action(actions[i])
            df.set_plane_pitch(self.Plane_ID_ally[i], float(exe_action[0]))
            df.set_plane_roll(self.Plane_ID_ally[i], float(exe_action[1]))
            df.set_plane_yaw(self.Plane_ID_ally[i], float(exe_action[2]))
            df.set_plane_linear_speed(self.Plane_ID_ally[i], self.initial_speed)

        df.update_scene()

    def normalize_action(self, action):
        """将离散动作表示为连续值"""
        norm_act = np.zeros(action.shape)  # 初始化一个和输入动作同形状的数组
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.

        return norm_act

    ### 环境重置与敌方动作应用 ###
    def set_enemy_policy(self, ally_states, enemy_states):
        """设置敌方追逐策略：AI"""
        for enemy_id in range(self.enemy_num):
            enemy_id_state = enemy_states[enemy_id]
            if enemy_id_state["health_level"] > 0:
                # 敌机可控
                df.set_plane_linear_speed(self.Plane_ID_enemy[enemy_id], 100)

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

        # 定义敌方飞机的基准位置（单位：米）
        enemy_pairs = {
            1: {'base_position': (0, 2500), 'formation_offset': [(-300, 0), (300, 0)]},  # 正北7km，V型队形
            2: {'base_position': (-2500, 0), 'formation_offset': [(0, -300), (0, 300)]},  # 正西7km，垂直队形
            3: {'base_position': (2500, -2000), 'formation_offset': [(-300, 300), (300, -300)]}  # 东南7km-5km，斜向队形
        }

        # 重置敌方yaw记录
        self.enemy_initial_yaws = {}

        for pair_id, pair_info in enemy_pairs.items():
            base_x, base_z = pair_info['base_position']
            offsets = pair_info['formation_offset']

            for offset_id, (dx, dz) in enumerate(offsets):
                enemy_id = (pair_id - 1) * 2 + offset_id + 1  # 假设每对有两个敌机，编号连续
                plane_id = f"ennemy_{enemy_id}"
                enemy_x = base_x + dx
                enemy_z = base_z + dz
                enemy_y = 5000  # 固定高度，根据需要调整
                # enemy_yaw = np.random.uniform(-pi, pi)  # 随机初始化敌方 yaw
                enemy_yaw = self.calculate_yaw(enemy_x, enemy_z)

                # 记录敌机 yaw
                self.enemy_initial_yaws[plane_id] = enemy_yaw

                df.rearm_machine(plane_id)
                df.reset_machine(plane_id)
                df.reset_machine_matrix(plane_id, enemy_x, enemy_y, enemy_z, 0, enemy_yaw, 0)
                df.retract_gear(plane_id)
                df.set_plane_linear_speed(plane_id, 100)
                # df.set_plane_autopilot_speed(plane_id, 100)
                # df.activate_autopilot(plane_id)
                df.activate_ia(plane_id)

        df.update_scene()


    ### 目标分配有关 ###
    def reset_command_variables(self):
        """重置元命令用到的变量"""
        self.command_groups = [[0, 1], [2, 3], [4, 5]]
        self.command_target_running = copy.deepcopy(self.command_target)  # 表示此时含有的组别
        self.command_groups_running = copy.deepcopy(self.command_groups)
        self.assign_random_commands()  # 随机分配元命令
        self.encode_ally_commands()  # 编码元命令

    def assign_random_commands(self):
        """为每个智能体随机分配目标"""
        for ally_id in range(self.ally_num):
            target = random.choice(self.command_target)
            self.ally_command[ally_id] = target

    def encode_ally_commands(self):
        """对命令进行独热编码"""
        target_size = len(self.command_target)

        # 创建目标的映射字典
        target_map = {target: i for i, target in enumerate(self.command_target)}

        for agent_id, target in self.ally_command.items():
            # 独热编码目标
            target_encoded = np.zeros(target_size)
            target_encoded[target_map[target]] = 1

            # 合并动作和目标的编码
            self.encoded_commands[agent_id] = target_encoded

    def one_hot_encode_target(self, target_id):
        """对目标编号进行独热编码"""
        # 创建一个全零数组，长度等于敌人总数
        one_hot_encoded = np.zeros(self.enemy_group_num)
        # 设置对应的索引为1
        one_hot_encoded[target_id] = 1
        return one_hot_encoded

    def get_enemy_group(self, enemy_id):
        """得到敌方所属的组别序号"""
        for group_index, group in enumerate(self.command_groups):
            if enemy_id in group:
                return group_index
        print("未找到相应敌方所属组别")
        return None

    def after_attack_assign_command(self, enemy_id, enemy_id_group):
        """每一次击杀enemy_id后判断和分配元命令"""
        self.command_groups_running[enemy_id_group].remove(enemy_id)  # 从该敌机所属的组别中剔除该敌机
        if not self.command_groups_running[enemy_id_group]:
            # 该列表为空，表示该组别的敌机已经没有了
            self.command_target_running.remove(enemy_id_group)  # 清除该组别
            self.assign_command_same_target(enemy_id_group)  # 给以该组别为目标的智能体重新分配目标
            self.encode_ally_commands()  # 重新进行编码

    def assign_command_same_target(self, enemy_id_group):
        """给选择了enemy_id的智能体重新分配元命令"""
        matching_ally_ids = [ally_id for ally_id, target in self.ally_command.items() if
                             target == enemy_id_group]
        # 判断此时所有组别是否为空
        if self.command_target_running:
            for ally_id in matching_ally_ids:
                self.assign_random_commands_running(ally_id)
        else:
            # 表示没有敌人了
            print("所有组别被清除")

    def assign_random_commands_running(self, agent_id):
        """表示在运行过程中为智能体agent_id分配元命令"""
        target = random.choice(self.command_target_running)
        self.ally_command[agent_id] = target


    ### 额外用到的函数 ###
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








