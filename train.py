import gymnasium as gym
import wandb
import os
import panda_gym
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of the optimizer")   
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
    parser.add_argument("--use-sde", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--use-risk-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use the risk model or not.")

    args = parser.parse_args()
    # fmt: on
    return args



config = parse_args()

def make_env():
    config = parse_args()
    env = gym.make(config.env_id)
    # env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env



if __name__ == "__main__":
    config = parse_args()
    wandb.init(
        config=config,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        # monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )
    env = DummyVecEnv([make_env])
    model = PPO('MultiInputPolicy', env, tensorboard_log=f"runs/ppo",
                learning_rate=config.learning_rate, n_steps=config.num_steps, 
                batch_size=config.batch_size, n_epochs=config.update_epochs, 
                gamma=config.gamma, gae_lambda=config.gae_lambda, clip_range=config.clip_coef,
                normalize_advantage=config.norm_adv, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm, use_sde=config.use_sde, sde_sample_freq=-1, 
                target_kl=config.target_kl, stats_window_size=100, policy_kwargs=None,
                verbose=1, seed=config.seed, device="auto", _init_setup_model=True)
    model.learn(total_timesteps=config.total_timesteps)
    wandb.finish()
