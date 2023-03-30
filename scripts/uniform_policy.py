# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
import pickle
from distutils.util import strtobool

import safety_gym, gym
import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from safety_gym.envs.engine import Engine


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", type=str, default="./traj_single", 
        help="storage directory with uniformly random policy") 
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--num-episodes", type=int, default=1000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--hazards_num", type=int, default=0,
        help="number of hazards in the environment")
    parser.add_argument("--vases_num", type=int, default=0,
        help="number of vases in the environment")
    parser.add_argument("--pillars_num", type=int, default=0,
        help="number of pillars in the environment")
    parser.add_argument("--gremlins_num", type=int, default=0,
        help="number of gremlins in the environment")
    parser.add_argument("--task", type=str, default="goal", 
        help="task to do in the environment (push / button / goal)")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--change_config_schedule", type=int, default=5,
        help="change the configuration of the environment every x episodes")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--model_name", type=str, default="ppo", 
        help="name of the model" ) 
    parser.add_argument("--config_seed", type=int, default=1,
        help="seed to sample the right configuration.")
    args = parser.parse_args()
    #args.batch_size = int(args.num_envs * args.num_steps)
    #args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args



def get_random_config(args):
    config = {
        'robot_base': 'xmls/car.xml',
        'task': args.task,
        #'observe_goal_lidar': True,
        #'observe_box_lidar': True,
        #'observe_box_lidar': True,
        #'observe_hazards': True,
        #'observe_vases': True,
        "sensors_obs": [],
        'constrain_hazards': True,
        'constrain_vases': True,
        'constrain_gremlins': True,
        'constrain_pillars': True,
        'vases_velocity_cost': 0.0, 
        'hazards_num': args.hazards_num,
        'vases_num': args.vases_num,
        'pillars_num': args.pillars_num,
        'gremlins_num': args.gremlins_num,
        'observation_flatten': True,
        'observe_vision': True,
        "sensors_hinge_joints": False,
        "sensors_ball_joints": False,
        "sensors_angle_components": False,
    }
    return config 






def make_env(args, seed, idx, capture_video, run_name, gamma):
    def thunk():
        config = get_random_config(args)
        print("Config: ", config)
        env = Engine(config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #if capture_video:
        #    if idx == 0:
        #        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = f"Uniform_H{args.hazards_num}__G{args.gremlins_num}__V{args.vases_num}__P{args.pillars_num}"
    #run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    storage_path = os.path.join(os.getcwd(), args.storage_dir, run_name)
    os.system("rm -rf %s"%(storage_path))
    os.makedirs(storage_path)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    print(envs.single_action_space, envs.single_observation_space)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device) 

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    _ = envs.reset()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    cost = 0 
    # num_updates = args.total_timesteps // args.batch_size
    return_, cum_cost, ep_cost = 0.0, np.array([0.]), np.array([0.])
    episode, cost_list, fear = 0, [], [] 
    traj_obs, traj_rewards, traj_actions, traj_costs = None, None, None, None
    for episode in range(args.num_episodes):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            costs[step] = cost
            action = torch.Tensor(envs.action_space.sample())
            actions[step] = action
            #envs.render()
            #import time
            #time.sleep(1)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, dummy, info = envs.step(action.cpu().numpy())
            next_obs = next_obs 
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            return_ += args.gamma * reward 
            if not done:
                cost = torch.Tensor(info["cost"]).to(device).view(-1)
                ep_cost += info["cost"]; cum_cost += info["cost"]
            else:
                cost = torch.Tensor(np.array([info["final_info"][0]["cost"]])).to(device).view(-1)
                ep_cost += np.array([info["final_info"][0]["cost"]]); cum_cost += np.array([info["final_info"][0]["cost"]])
            if done:
                wandb.log({"episodic_return": return_}, step=global_step)
                wandb.log({"episodic_cost": ep_cost}, step=global_step)
                wandb.log({"cummulative_cost": cum_cost}, step=global_step)
                print(f"global_step={global_step}, episodic_return={return_}")
                return_ = 0
                ep_cost = np.array([0.])
                episode += 1
                traj_obs = obs if traj_obs == None else torch.concat([traj_obs, obs], axis=1)
                traj_rewards = rewards if traj_rewards == None else torch.concat([traj_rewards, rewards], axis=1)
                traj_actions = actions if traj_actions == None else torch.concat([traj_actions, actions], axis=1)
                traj_costs   = costs   if traj_costs   == None else torch.concat([traj_costs,   costs  ], axis=1)
                torch.save(traj_obs, os.path.join(storage_path, "obs.pt"))
                torch.save(traj_rewards, os.path.join(storage_path, "rewards.pt"))
                torch.save(traj_actions, os.path.join(storage_path, "actions.pt"))
                torch.save(traj_costs,   os.path.join(storage_path, "costs.pt"))
                #wandb.save(os.path.join(storage_path, "obs.pt"))
                #wandb.save(os.path.join(storage_path, "rewards.pt"))
                #wandb.save(os.path.join(storage_path, "actions.pt"))
                #wandb.save(os.path.join(storage_path, "costs.pt"))
                break
    envs.close()
    writer.close()
