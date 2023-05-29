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

from comet_ml import Experiment

import hydra




def get_config():
    config = {
        'robot_base': 'xmls/car.xml',
        'task': 'goal',
        'lidar_num_bins': 50,
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
        'hazards_num': 4, #cfg.ppo.hazards_num,
        'vases_num': 0, #cfg.ppo.vases_num,
        'pillars_num': 0, #cfg.ppo.pillars_num,
        'gremlins_num': 0, #cfg.ppo.gremlins_num,
        'observation_flatten': True,
        #'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        #'observe_vision': bool(cfg.ppo.vision),
        #'vision_size': (cfg.ppo.width, cfg.ppo.height)
    }
    return config



from safety_gym.envs.engine import Engine


def make_env(cfg, idx, capture_video, run_name, gamma):
    def thunk():
        #if capture_video:
        #    env = gym.make(env_id, render_mode="rgb_array")
        #else:
        #    env = gym.make(env_id)
        config = get_config()
        print("Config: ", config)
        env = Engine(config)
        env.seed(cfg.ppo.seed)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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
risk_est.load_state_dict(torch.load("./models/risk_model.pt", map_location=device))



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
        #self.actor_mean  nn.Sequential(
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

#@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg):
    run_name = f"{cfg.ppo.env_id}__{cfg.ppo.exp_name}__{cfg.ppo.seed}__{int(time.time())}"

    experiment = Experiment(
        api_key="FlhfmY238jUlHpcRzzuIw3j2t",
        project_name="risk-aware-exploration",
        workspace="hbutsuak95",
    )

    batch_size = int(cfg.ppo.num_envs * cfg.ppo.num_steps)
    minibatch_size = int(batch_size // cfg.ppo.num_minibatches)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.ppo.seed)
    np.random.seed(cfg.ppo.seed)
    torch.manual_seed(cfg.ppo.seed)
    torch.backends.cudnn.deterministic = cfg.ppo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.ppo.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg, i, cfg.ppo.capture_video, run_name, cfg.ppo.gamma) for i in range(cfg.ppo.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + envs.single_observation_space.shape).to(device)
    risks = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs, 2)).to(device) 
    actions = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    rewards = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    dones = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    values = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()#seed=cfg.ppo.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.ppo.num_envs).to(device)
    num_updates = cfg.ppo.total_timesteps // batch_size
    video_filenames = set()
    cum_cost, ep_cost = np.array([0.]), np.array([0.]) 
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.ppo.num_steps):
            global_step += 1 * cfg.ppo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                risk = torch.zeros(([1,2])).to(device)
                if cfg.ppo.use_risk_model:
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
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if not done:
                cost = torch.Tensor(infos["cost"]).to(device).view(-1)
                ep_cost += infos["cost"]; cum_cost += infos["cost"]
            else:
                cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)
                ep_cost += np.array([infos["final_info"][0]["cost"]]); cum_cost += np.array([infos["final_info"][0]["cost"]])
            if done:
                print("Done")
            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episode_cost={ep_cost}")
                experiment.log_metric("charts/episodic_return", info["episode"]["r"], global_step)
                experiment.log_metric("charts/episodic_length", info["episode"]["l"], global_step)
                experiment.log_metric("charts/episodic_cost", ep_cost[0], global_step)
                experiment.log_metric("charts/cummulative_cost", cum_cost[0], global_step)
                ep_cost = np.array([0.])
                 
        # bootstrap value if not done
        with torch.no_grad():
            risk = torch.zeros(([1,2])).to(device)
            if cfg.ppo.use_risk_model:
                id_risk = torch.argmax(risk_est(next_obs), axis=1)
                risk[:, id_risk] = 1
            #print(next_obs.size())
            next_value = agent.get_value(next_obs, risk).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.ppo.num_steps)):
                if t == cfg.ppo.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
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
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_risks[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.ppo.clip_coef,
                        cfg.ppo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

            if cfg.ppo.target_kl != "None":
                if approx_kl > cfg.ppo.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        experiment.log_metric("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        experiment.log_metric("losses/value_loss", v_loss.item(), global_step)
        experiment.log_metric("losses/policy_loss", pg_loss.item(), global_step)
        experiment.log_metric("losses/entropy", entropy_loss.item(), global_step)
        experiment.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)
        experiment.log_metric("losses/approx_kl", approx_kl.item(), global_step)
        experiment.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)
        experiment.log_metric("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        experiment.log_metric("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # if cfg.ppo.track and cfg.ppo.capture_video:
        #     for filename in os.listdir(f"videos/{run_name}"):
        #         if filename not in video_filenames and filename.endswith(".mp4"):
        #             wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
        #             video_filenames.add(filename)

    envs.close()
    # writer.close()


if __name__ == "__main__":
    train()
