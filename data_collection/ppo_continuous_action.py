# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import safety_gym, gym
import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from safety_gym.envs.engine import Engine
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from PIL import Image
import pickle 
# import wandb 


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", type=str, default="./ppo/traj_single",
        help="storage directory with ppo policy")
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
    parser.add_argument("--total-timesteps", type=int, default=1000000,
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
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
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
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--model_name", type=str, default="ppo", 
        help="name of the model" ) 
    ### Environment specifications 
    parser.add_argument("--hazards_num", type=int, default=4,
        help="number of hazards in the environment")
    parser.add_argument("--vases_num", type=int, default=0,
        help="number of vases in the environment")
    parser.add_argument("--pillars_num", type=int, default=0,
        help="number of pillars in the environment")
    parser.add_argument("--gremlins_num", type=int, default=0,
        help="number of gremlins in the environment")
    parser.add_argument("--task", type=str, default="goal",
        help="task to do in the environment (push / button / goal)")
    parser.add_argument("--width", default=227, type=int,
        help="width of the ego-image saved for the robot")
    parser.add_argument("--height", default=227, type=int,
        help="height of the ego-image saved for the robot")
    parser.add_argument("--vision", type=int, default=0,
        help="whether to store images or not ")
    parser.add_argument("--lidar_num_bins", type=int, default=50,
        help="number of bins for lidar measurement")

    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def get_config(args):
    config = {
        'robot_base': 'xmls/car.xml',
        'task': args.task,
        'lidar_num_bins': args.lidar_num_bins,
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




def get_random_config1(args):
    config = {
        'robot_base': 'xmls/car.xml',
        'task': args.task,
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        #"sensors_obs": [],
        'constrain_hazards': True,
        'constrain_vases': True,
        'constrain_gremlins': True,
        'constrain_pillars': True,
        'vases_velocity_cost': 0.0,
        'hazards_num': args.hazards_num,
        'vases_num': args.vases_num,
        'pillars_num': args.pillars_num,
        'gremlins_num': args.gremlins_num,
        'observation_flatten': False,
        'observe_vision': args.vision,
        'vision_size': (227, 227)
        #"sensors_hinge_joints": False,
        #"sensors_ball_joints": False,
        #"sensors_angle_components": False,
    }
    return config


def make_env(args, seed, idx, capture_video, run_name, gamma):
    # def thunk():
    config = get_config(args)
    print("Config: ", config)
    env = Engine(config)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    #if capture_video:
    #    if idx == 0:
    #        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

    # return thunk



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 5, padding="same")
        self.conv4 = nn.Conv2d(32, 64, 5, padding="same")
        self.fc1 = nn.Linear(64 * 14 * 14, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  ## 113 * 113
        x = self.pool(F.relu(self.conv2(x)))  ## 56  * 56 
        x = self.pool(F.relu(self.conv3(x)))  ## 28  * 28
        x = self.pool(F.relu(self.conv4(x)))  ## 14  * 14 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x 


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        #self.state_encoder = StateEncoder()
        self.critic = nn.Sequential(
            #layer_init(nn.Linear(120, 64)),
            layer_init(nn.Linear(174, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            #layer_init(nn.Linear(120, 64)),
            layer_init(nn.Linear(174, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        #x = torch.transpose(x, 1, 3)
        #x = self.state_encoder(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        #x = torch.transpose(x, 1, 3)
        #x = self.state_encoder(x) 
        #print(x.size())
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd#.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    storage_path =  args.storage_dir
    os.system("rm -rf %s"%(storage_path))
    os.makedirs(storage_path) 
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
        )
    writer = SummaryWriter("runs/new_run")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args, args.seed, 0, args.capture_video, "run_name", args.gamma)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    #print(envs.single_observation_space['vision'].shape)
    # ALGO Logic: Storage setup
    action_space_shape = envs.action_space.shape
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space['vision'].shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    del next_obs["vision"]
    next_obs = np.array(np.hstack([k.ravel() for k in next_obs.values()]))
    next_obs = torch.Tensor(next_obs).to(device)
    obs_space_shape = next_obs.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + next_obs.shape).to(device)

    #next_obs = torch.Tensor(envs.reset()[0]['vision']).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    return_, cum_cost, ep_cost = 0.0, np.array([0.]), np.array([0.])
    cost, cost_list, fear = 0, [], []
    episode = 0 
    traj_obs, traj_rewards, traj_actions, traj_costs = None, None, None, None
    for update in range(1, num_updates + 1):
        traj_path = os.path.join(storage_path, "ep_%d"%episode)
        os.makedirs(traj_path)
        os.makedirs(os.path.join(traj_path, "info"))
        os.makedirs(os.path.join(traj_path, "lidar"))
        if args.vision:
            os.makedirs(os.path.join(traj_path, "rgb"))
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step, 0] = next_obs
            dones[step, 0] = next_done
            costs[step, 0] = cost

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step, 0] = action
            logprobs[step, 0] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, dummy, info = envs.step(action.cpu().numpy())
            cost = info["cost"]
            # else:
            #     print(info, done)
            #     cost = info["final_info"][0]["cost"]
            if args.vision:
                im = Image.fromarray(np.array(next_obs['vision']*255).astype(np.uint8))
                im.save(os.path.join(traj_path, "rgb", "%d.png"%step))
                del next_obs['vision']
            info_dict = {'reward': reward, 'done': done, 'cost': cost, 'prev_action': action} #, 'prev_obs_rgb': obs['vision']}
            #info_dict.update(obs)
            ## Saving the info for this step
            f1 = open(os.path.join(traj_path, "info", "%d.pkl"%step), "wb")
            pickle.dump(info_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
            f1.close()

            
            ## Saving data from other sensors (particularly lidar)
            f2 = open(os.path.join(traj_path, "lidar", "%d.pkl"%step), "wb")
            pickle.dump(next_obs, f2, protocol=pickle.HIGHEST_PROTOCOL)
            f2.close()
            next_obs = np.array(np.hstack([k.ravel() for k in next_obs.values()]))
            # print(next_obs.shape)
            #next_obs = next_obs['vision']
            rewards[step, 0] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
            return_ += args.gamma * reward 
            # if not done:
            cost = torch.Tensor([info["cost"]]).to(device).view(-1)
            ep_cost += info["cost"]; cum_cost += info["cost"]
            if done:
                episode += 1 
                print(f"global_step={global_step}, episodic_return={return_}, episodic_cost={ep_cost}")
                return_ = 0
                ep_cost = np.array([0.])
                envs.reset()
                break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space_shape)
        #b_obs = obs.reshape((-1,) + envs.single_observation_space['vision'].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    envs.close()
    writer.close()
