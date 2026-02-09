#!/usr/bin/env python
import sys
import os
import traceback
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from config import get_config
from runner.share_harfang_runner import ShareHarfangRunner
from envs.HarfangSim.envs.AirCombatEnv import MultiCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv

base_train_port = 53100
base_eval_port = 53099
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():

            time.sleep(1)
            env = MultiCombatEnv(base_train_port + rank)

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):


    def get_env_fn(rank):
        def init_env():

            env = MultiCombatEnv(base_eval_port + rank)

            env.seed(all_args.seed * 50000 + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="AirCombatEnv", help="Which scenario to run on")
    parser.add_argument("--exp_class", type=str, default="train", help="exp is train or test")
    parser.add_argument("--env_name", type=str, default="AirCombatEnv", help="Which envs to run on")
    parser.add_argument("--task_name", type=str, default="escape_level_1_v2", help="Which task to run on")

    parser.add_argument("--num_agents", type=int, default=6, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_platform / all_args.experiment_name / all_args.exp_class / all_args.env_name / all_args.task_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" +
                              str(all_args.experiment_name)
                              + "-" + str(all_args.task_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    runner = ShareHarfangRunner(config)

    try:
        # runner.run()
        runner.render2()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()

"""
2024/4/27
该环境当前初始化已经完成debug
下一步计划：
    根据参考的close代码，修改环境中返回的观测，使其一致
"""
if __name__ == "__main__":
    main(sys.argv[1:])
