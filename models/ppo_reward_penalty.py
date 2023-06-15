# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import safety_gym, gym 
#import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
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
    parser.add_argument("--use-risk-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use the risk model or not.")

    ## Environment specifications  
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
    #parser.add_argument("--use-reward-penalty", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #    help="Add reward penalty for unsafe states")
    parser.add_argument("--reward-penalty", type=float, default=0,
        help="coefficient of the value function")
    parser.add_argument("--early_termination", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="whether to terminate an episode if the unsafe state is reached")


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args





def get_config(args):
    config = {
        'robot_base': 'xmls/car.xml',
        'task': args.task,
        'lidar_num_bins': 50,
        'early_termination': args.early_termination,
        #'goal_size': 0.3,
        #'goal_keepout': 0.305,
        #'hazards_size': 0.2,
        #'hazards_keepout': 0.18,
        #'lidar_max_dist': 3,
        'observe_goal_lidar': True,
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
        'observation_flatten': True,
        #'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        #'observe_vision': bool(args.vision),
        #'vision_size': (args.width, args.height)
    }
    return config



from safety_gym.envs.engine import Engine


def make_env(args, idx, capture_video, run_name, gamma):
    def thunk():
        #if capture_video:
        #    env = gym.make(env_id, render_mode="rgb_array")
        #else:
        #    env = gym.make(env_id)
        config = get_config(args)
        print("Config: ", config)
        env = Engine(config)
        env.seed(args.seed)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.relu(self.dropout(self.fc4(x)))
        out = self.softmax(self.out(x))
        return out


risk_est = RiskEst(obs_size=174) #np.array(envs.single_observation_space.shape).prod())
risk_est.to("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
risk_est.load_state_dict(torch.load("./pretrained/risk_model_Jun13.pt", map_location=device))



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        #self.critic = nn.Sequential(
        #    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(64, 64)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(64, 1), std=1.0),
        #)
        #self.risk_est = RiskEst(obs_size=np.array(envs.single_observation_space.shape).prod())
        self.risk_encoder_actor = nn.Sequential(
            layer_init(nn.Linear(2, 12)),
            nn.Tanh())
        self.risk_encoder_critic = nn.Sequential(
            layer_init(nn.Linear(2, 12)),
            nn.Tanh())
        #self.actor_mean = nn.Sequential(
        #    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(64, 64)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        #)
        self.actor_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.actor_fc2 = layer_init(nn.Linear(76, 76))
        self.actor_fc3 = layer_init(nn.Linear(76, np.prod(envs.single_action_space.shape)), std=0.01)
        
        self.critic_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.critic_fc2 = layer_init(nn.Linear(76, 76))
        self.critic_fc3 = layer_init(nn.Linear(76, 1), std=0.01)

        self.tanh = nn.Tanh() 
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, risk):
        risk_enc = self.risk_encoder_critic(risk)
        x1 = self.tanh(self.critic_fc1(x))
        x2 = torch.cat([risk_enc, x1], axis=1)
        x3 = self.tanh(self.critic_fc2(x2))
        val = self.critic_fc3(x3)

        return val

    def get_action_and_value(self, x, risk, action=None):
        risk_enc = self.risk_encoder_actor(risk)
        x1 = self.tanh(self.actor_fc1(x))
        x2 = torch.cat([risk_enc, x1], axis=1)
        x3 = self.tanh(self.actor_fc2(x2))
        action_mean = self.actor_fc3(x3)
        #action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, risk)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    risks = torch.zeros((args.num_steps, args.num_envs, 2)).to(device) 
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()#seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()
    cum_cost, ep_cost = np.array([0.]), np.array([0.]) 
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                risk = torch.zeros(([1,2])).to(device)
                if args.use_risk_model:
                    id_risk = torch.argmax(risk_est(next_obs), axis=1)
                    risk[:, id_risk] = 1
                risks[step] = risk
                #print(risk.size())
                action, logprob, _, value = agent.get_action_and_value(next_obs, risk)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            if not done:
                cost = torch.Tensor(infos["cost"]).to(device).view(-1)
                ep_cost += infos["cost"]; cum_cost += infos["cost"]
            else:
                cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)
                ep_cost += np.array([infos["final_info"][0]["cost"]]); cum_cost += np.array([infos["final_info"][0]["cost"]])

            reward = torch.Tensor(reward).to(device).view(-1)
            reward -= cost * args.reward_penalty

            rewards[step] = reward
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if done:
                print("Done")

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_cost", ep_cost[0], global_step)
                writer.add_scalar("charts/cummulative_cost", cum_cost[0], global_step)
                ep_cost = np.array([0.])
                 
        # bootstrap value if not done
        with torch.no_grad():
            risk = torch.zeros(([1,2])).to(device)
            if args.use_risk_model:
                id_risk = torch.argmax(risk_est(next_obs), axis=1)
                risk[:, id_risk] = 1
            #print(next_obs.size())
            next_value = agent.get_value(next_obs, risk).reshape(1, -1)
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_risks = risks.reshape((-1, 2)) 

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_risks[mb_inds], b_actions[mb_inds])
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

    envs.close()
    writer.close()
