# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
import pickle
import json
from distutils.util import strtobool
from PIL import Image

import safety_gym, gym
from safety_gym.envs.engine import Engine

import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="traj_uniform",
        help="root directory for storing trajectories")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--num_episodes", type=int, default=100,
        help="number of episodes to collect data for")
    parser.add_argument("--num_steps", type=int, default=1024,
        help="how many steps to go in the environment")
    parser.add_argument("--task", default="goal", type=str, 
        help="Specifying the task in the environment")
    parser.add_argument("--hazards_num", type=int, default=0, 
        help="Number of hazardous regions in the environment")
    parser.add_argument("--gremlins_num", type=int, default=0, 
        help="Number of gremlins in the environment")
    parser.add_argument("--buttons_num", type=int, default=0, 
        help="Number of buttons in the environment")
    parser.add_argument("--vases_num", type=int, default=0, 
        help="Number of vases in the environment")
    parser.add_argument("--pillars_num", type=int, default=0, 
        help="Number of pillars in the environment")
    parser.add_argument("--width", default=227, type=int, 
        help="width of the ego-image saved for the robot")
    parser.add_argument("--height", default=227, type=int, 
        help="height of the ego-image saved for the robot")
    parser.add_argument("--vision", type=int, default=1, 
        help="whether to store images or not ")
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    args = parser.parse_args()
    return args



def get_config(args):
    config = {
        'robot_base': 'xmls/car.xml',
        'task': args.task,
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        'constrain_hazards': True,
        'constrain_gremlins': True,
        'constrain_vases': True,
        'constrain_pillars': True,
        'constrain_buttons': True,
        'hazards_num': args.hazards_num,
        'vases_num': args.vases_num,
        'pillars_num': args.pillars_num,
        'gremlins_num': args.gremlins_num,
        'observation_flatten': False,
        'observe_vision': bool(args.vision),
        'vision_size': (args.width, args.height)
    }
    return config 






def make_env(args):
    config = get_config(args)
    env = Engine(config)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


if __name__ == "__main__":
    args = parse_args()
    run_name = f"Uniform_H{args.hazards_num}__G{args.gremlins_num}__V{args.vases_num}__P{args.pillars_num}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            #monitor_gym=True,
            #save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    storage_path = os.path.join(os.getcwd(), args.data_path, run_name)
    os.system("rm -rf %s"%(storage_path))
    os.makedirs(storage_path)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    # env setup
    envs = make_env(args)
    obs, _ = envs.reset()
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # num_updates = args.total_timesteps // args.batch_size
    return_, cum_cost, ep_cost = 0.0, np.array([0.]), np.array([0.])
    episode, cost_list, fear = 0, [], [] 
    traj_obs, traj_rewards, traj_actions, traj_costs = None, None, None, None
    for episode in range(args.num_episodes):
        traj_path = os.path.join(storage_path, "traj_%d"%episode)
        os.makedirs(os.path.join(traj_path, "info"))
        os.makedirs(os.path.join(traj_path, "lidar"))
        if args.vision:
            os.makedirs(os.path.join(traj_path, "rgb"))
        for step in range(0, args.num_steps):
            action = envs.action_space.sample()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, dummy, info = envs.step(action)
            # if not done:
            cost = info["cost"]
            # else:
            #     print(info, done)
            #     cost = info["final_info"][0]["cost"]
            if args.vision:
                im = Image.fromarray(np.array(next_obs['vision']*255).astype(np.uint8))
                im.save(os.path.join(traj_path, "rgb", "%d.png"%step))
                del next_obs['vision']
            info_dict = {'reward': reward, 'done': done, 'cost': cost, 'prev_action': action} #, 'prev_obs_rgb': obs['vision']}
            info_dict.update(obs)
            ## Saving the info for this step
            f1 = open(os.path.join(traj_path, "info", "%d.pkl"%step), "wb")
            pickle.dump(info_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
            f1.close()
            # del obs['vision']
            ## Saving data from other sensors (particularly lidar)
            f2 = open(os.path.join(traj_path, "lidar", "%d.pkl"%step), "wb")
            pickle.dump(next_obs, f2, protocol=pickle.HIGHEST_PROTOCOL)
            f2.close()
            obs = next_obs
            if done:
                done=False
                obs, _ = envs.reset()
                break
    envs.close()
    writer.close()
