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
from tasks.overall_combat_v3_level_2 import MultiCombatTask

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

        # # 创建socket服务
        # socket_host = '127.0.0.1'
        # socket_port = 9999
        # self.socket_server = EnvSocketServer(host=socket_host, port=socket_port)
        # self.socket_server.env = self
        #
        # # 环境重置信息，辅助前端偏好输入
        # self.reset_flag = False

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

        self.task.initial_enemy_dis(self.ally_states, self.enemy_states)

        return ally_obs, ally_share_obs

    def step(self, action):
        """步进函数"""
        self.current_step += 1
        info = {"current_step": self.current_step}
        self.reset_flag = False
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
        self.task.reward_attack_group = [0] * self.task.ally_num  # 每一步击毁奖励重置

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
            self.task.all_ally_dones[i] = self.task.get_done(i, self.ally_states, self.enemy_states)

        game_over = self.task.done_win_game(self.ally_states, self.enemy_states)  # 一方全部死亡终止
        enemy_out_range = self.task.enemy_out_range(self.enemy_states)  # 敌方任意一个出界

        if self.current_step > 8000:
            print("环境", self.port, "运行超时，进行重置")
            self.task.all_ally_dones = np.full((self.task.ally_num, 1), True)
        if game_over:
            print("环境", self.port, "一方全部死亡，进行重置")
            self.task.all_ally_dones = np.full((self.task.ally_num, 1), True)
        if enemy_out_range:
            print("敌方出界重置环境")
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
            df.display_vector(enemy_state["position"], enemy_state["move_vector"], 'enemy', [0, 0], [0, 0, 1, 1], 0.02)
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

















