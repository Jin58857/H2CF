"""
负责连接环境与加载任务
"""
import os
import sys
import copy
import time
import numpy as np
from gym.utils import seeding
from math import pi
import math
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.dogfight_client as df
from .EnvSocket import EnvSocketServer
from tasks.escape_level_1_v2 import MultiCombatTask

CONFIG_DIR = Path(__file__).resolve().parents[5] / "configs"


def _load_sim_connection():
    """
    Load simulator host/render flags; values are redacted from the public repo.
    """
    config_path = CONFIG_DIR / "sim_connection.json"
    host = os.environ.get("H2CF_SIM_HOST")
    renderless = os.environ.get("H2CF_SIM_RENDERLESS")

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            host = host or data.get("host")
            if renderless is None:
                renderless = data.get("renderless", True)

    if not host:
        raise RuntimeError(
            "Core simulator endpoint removed for the public release. "
            "Set H2CF_SIM_HOST or provide configs/sim_connection.json with field 'host'."
        )

    renderless_flag = True
    if renderless is not None:
        renderless_flag = str(renderless).lower() not in ("0", "false", "no", "off")

    return host, renderless_flag


class MultiCombatEnv(object):
    """
    仿真环境接口
    """
    def __init__(self, port):
        # 基本的参数设置
        self.port = port
        self.current_step = 0
        self._create_records = False
        self.render_step = 0
        self.load()  # 主要用于加载任务和模拟器环境
        # 创建socket服务
        socket_host = '127.0.0.1'
        socket_port = 9999
        self.socket_server = EnvSocketServer(host=socket_host, port=socket_port)
        self.socket_server.env = self

        # 环境重置信息，辅助前端偏好输入
        self.reset_flag = False
    @property
    def observation_space(self):
        return self.task.observation_space

    @property
    def action_space(self):
        return self.task.action_space

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    @property
    def num_agents(self):
        return self.task.num_agents

    def load(self):
        self.load_task()
        self.connect_harfang()
        self.seed()

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
                "id": f"Red{ally_id + 1}",
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
            group_id = self.task.get_enemy_group(enemy_id)
            enemy_positions.append({
                "id": f"Blue{enemy_id + 1}",
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
        # threat_levels = self.compute_threat_levels()

        data_to_send = {
            "type": "update_positions",
            "reset": self.reset_flag,
            "ally": ally_positions,
            "enemy": enemy_positions,
            # "human_preference": self.human_preference.tolist(),
            # "enemy_groups": threat_levels  # 添加威胁值信息
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

    def reset(self):
        """环境重置"""
        self.current_step = 0
        self._create_records = False  # 和Tacview显示相关
        self.initial_yaw = np.random.uniform(-pi, pi)  # 初始状态角度

        # 重置任务变量
        self.task.reset_task()

        # 重置双方初始状态
        self.task.reset_machine()

        # 获取所有飞机的状态并缓存
        self.ally_states, self.enemy_states = self.get_all_plane_states()

        ally_obs = self.get_obs()
        ally_share_obs = self.get_state()

        self.task.initial_target_min_dis(self.ally_states, self.enemy_states)
        # 重置后发送当前状态
        high_actions = np.zeros((self.task.ally_num, 2), dtype=int)
        for ally_id in range(self.task.ally_num):
            high_actions[ally_id][1] = self.task.ally_command[ally_id]
        self.reset_flag = True
        self.send_positions_to_frontend(high_actions)
        return ally_obs, ally_share_obs

    def step(self, action):
        """步进函数"""
        self.current_step += 1
        info = {"current_step": self.current_step}

        # 敌方与我方的动作应用
        self.task.set_enemy_policy(self.ally_states, self.enemy_states)
        self.task.apply_action(action, self.ally_states)

        # 获取更新后的状态
        self.ally_states, self.enemy_states = self.get_all_plane_states()

        # 显示飞机标号
        # self.show_plane_id()

        # 计算状态转移后的奖励，在计算奖励后更新双方的死亡状态
        ally_reward = self.get_reward()

        # 更新状态缓存
        self.ally_states, self.enemy_states = self.get_all_plane_states()

        # 得到终止条件
        ally_done = self.get_done()
        ally_obs = self.get_obs()
        ally_share_obs = self.get_state()

        # 每一步发送一次当前状态
        high_actions = np.zeros((self.task.ally_num, 2), dtype=int)
        for ally_id in range(self.task.ally_num):
            high_actions[ally_id][1] = self.task.ally_command[ally_id]
        self.send_positions_to_frontend(high_actions)

        return ally_obs, ally_share_obs, ally_reward, ally_done, info

    def get_obs(self):
        """得到观测"""
        all_ally_obs = [self.task.get_obs(ally_id, self.ally_states, self.enemy_states) for ally_id in range(self.task.ally_num)]

        return np.stack(all_ally_obs)

    def get_state(self):
        """得到全局状态"""
        all_ally_share_obs = [self.task.get_share_obs(ally_id, self.ally_states, self.enemy_states) for ally_id in range(self.task.ally_num)]

        return np.stack(all_ally_share_obs)

    def get_reward(self):
        """得到奖励"""

        # 每一步一个智能体只能攻击一次敌方
        self.task.ally_launch_attack = [False] * self.task.ally_num
        self.task.enemy_launch_attack = [False] * self.task.enemy_num

        rewards = [self.task.get_reward(ally_id, self.ally_states, self.enemy_states) for ally_id in range(self.task.ally_num)]

        all_ally_reward = np.array(rewards, dtype=np.float32).reshape(self.task.ally_num, 1)

        self.clear_death()

        return all_ally_reward

    def get_done(self):
        """
        得到飞机是否需要终止
        """
        for i in range(self.task.ally_num):
            self.task.all_ally_dones[i] = self.task.get_done(i, self.ally_states)

        game_over = self.task.done_win_game(self.ally_states, self.enemy_states)  # 一方全部死亡终止

        if self.current_step > 7000:
            print("环境", self.port, "运行超时，进行重置")
            self.task.all_ally_dones = np.full((self.task.ally_num, 1), True)
        if game_over:

            self.task.all_ally_dones = np.full((self.task.ally_num, 1), True)

        return self.task.all_ally_dones

    def get_all_plane_states(self):
        ally_states = []
        enemy_states = []
        for plane_id in self.task.Plane_ID_ally:
            ally_states.append(df.get_plane_state(plane_id))
        for plane_id in self.task.Plane_ID_enemy:
            enemy_states.append(df.get_plane_state(plane_id))
        return ally_states, enemy_states

    def close(self):
        """关闭环境"""
        pass

    def seed(self, seed=None):
        """
        设置环境的种子
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def connect_harfang(self):
        """
        主要负责连接模拟器，以及敌方和我方的一些信息
        """
        host, renderless = _load_sim_connection()
        df.connect(host, self.port)
        time.sleep(2)
        df.disable_log()
        df.set_client_update_mode(True)
        df.set_renderless_mode(renderless)

        for i in range(1, self.task.ally_num + 1):
            self.task.Plane_ID_ally.append("ally_" + str(i))

        for i in range(1, self.task.enemy_num + 1):
            self.task.Plane_ID_enemy.append("ennemy_" + str(i))

    def load_task(self):
        """该函数用于加载特定的任务"""
        self.task = MultiCombatTask()

    def clear_death(self):
        """每一回合清除被攻击的智能体"""
        for ally_id in range(self.task.ally_num):
            ally_id_state = self.ally_states[ally_id]
            if ally_id_state["health_level"] > 0 and self.task.ally_being_attack[ally_id]:
                df.set_health(self.task.Plane_ID_ally[ally_id], 0)

        for enemy_id in range(self.task.enemy_num):
            enemy_id_state = self.enemy_states[enemy_id]
            if enemy_id_state["health_level"] > 0 and self.task.enemy_being_attack[enemy_id]:
                df.set_health(self.task.Plane_ID_enemy[enemy_id], 0)

        df.update_scene()

    def show_plane_id(self):
        """
        用来实时展示飞机的标号
        """
        for ally_id in range(self.task.ally_num):
            own_state = self.ally_states[ally_id]
            plane_id = self.task.Plane_ID_ally[ally_id]
            df.display_vector(own_state["position"], own_state["move_vector"], plane_id, [0, 0], [1, 0, 0, 1], 0.02)
        for enemy_id in range(self.task.enemy_num):
            enemy_state = self.enemy_states[enemy_id]
            df.display_vector(enemy_state["position"], enemy_state["move_vector"], str(self.task.get_enemy_group(enemy_id)), [0, 0], [0, 0, 1, 1], 0.02)
        df.update_scene()

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

















