import os
import pickle
import torch
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dirs(traj_path, episode):
        #try:
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "lidar"))
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "info"))
        
        #except:
        #    pass

        


def store_data(next_obs, info_dict, traj_path, episode, step_log):
        #, 'prev_obs_rgb': obs['vision']}
        #info_dict.update(obs)
        ## Saving the info for this step
        f1 = open(os.path.join(traj_path, "traj_%d"%episode, "info", "%d.pkl"%step_log), "wb")
        pickle.dump(info_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
        f1.close()
        # del obs['vision']
        ## Saving data from other sensors (particularly lidar)
        f2 = open(os.path.join(traj_path, "traj_%d"%episode, "lidar", "%d.pkl"%step_log), "wb")
        pickle.dump(next_obs, f2, protocol=pickle.HIGHEST_PROTOCOL)
        f2.close()


def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }

    return activation_dict[name]



def make_state_action_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        actions = torch.load(os.path.join(data_path, "actions.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        state_action_risk_data = None
        for idx in range(1, len(ep_len)):
                start, end = int(ep_len[idx-1]), int(ep_len[idx])
                print(start, end)
                obs_idx = obs[start:end]
                actions_idx = actions[start:end]
                risks_idx = risks[start:end]
                print(obs_idx.size(), actions_idx.size(), risks_idx.size())
                sar_data = torch.cat([obs_idx[:-1], actions_idx[1:], risks_idx[1:]], axis=1)
                state_action_risk_data = sar_data if state_action_risk_data is None else torch.cat([state_action_risk_data, sar_data], axis=0)
        torch.save(state_action_risk_data, os.path.join(data_path, "state_action_risk.pt"))
        return state_action_risk_data

def make_state_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        return torch.cat([obs, risks], axis=1)

def combine_data(data_path, type="state_risk"):
        for env in os.listdir(data_path):
                env_path = os.path.join(data_path, env)
                all_data = None
                for run in os.listdir(env_path):
                        run_path = os.path.join(env_path, run)
                        if type == "state_risk":
                                try:
                                        data = make_state_risk_data(run_path)
                                except:
                                        pass
                        else:
                                try:
                                        data = make_state_action_risk_data(run_path)
                                except:
                                        pass
                all_data = data if all_data is None else torch.cat([all_data, data], axis=0)
        torch.save(all_data, os.path.join(env_path, "all_%s.pt"%type))



class ReplayBuffer:
        def __init__(self, buffer_size=1000000, data_path="./data/"):
                self.obs = None 
                self.next_obs = None
                self.actions = None 
                self.rewards = None 
                self.dones = None
                self.risks = None 
                self.dist_to_fails = None 
                self.costs = None
                self.data_path = data_path
                self.buffer_size = buffer_size 

        def add(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                #self.obs = obs if self.obs is None else torch.concat([self.obs, obs], axis=0)
                self.next_obs = next_obs if self.next_obs is None else torch.concat([self.next_obs, next_obs], axis=0)
                #self.actions = action if self.actions is None else torch.concat([self.actions, action], axis=0)
                #self.rewards = reward if self.rewards is None else torch.concat([self.rewards, reward], axis=0)
                #self.dones = done if self.dones is None else torch.concat([self.dones, done], axis=0)
                self.risks = risk if self.risks is None else torch.concat([self.risks, risk], axis=0)
                #self.costs = cost if self.costs is None else torch.concat([self.costs, cost], axis=0)
                #self.dist_to_fails = dist_to_fail if self.dist_to_fails is None else torch.concat([self.dist_to_fails, dist_to_fail], axis=0)

        def __len__(self):
                return 0 if self.next_obs is None else self.next_obs.size()[0]
        
        def sample(self, sample_size):
                if self.next_obs.size()[0] > self.buffer_size:
                    self.next_obs = self.next_obs[-self.buffer_size:]
                    self.risks = self.risks[-self.buffer_size:]
                idx = range(self.next_obs.size()[0])
                sample_idx = np.random.choice(idx, sample_size)
                return {"obs": None, #self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": None, #self.actions[sample_idx],
                        "rewards": None, #self.rewards[sample_idx],
                        "dones": None, #self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": None, #self.costs[sample_idx],
                        "dist_to_fail": None} #self.dist_to_fails[sample_idx]}
        
        def sample_balanced(self, sample_size):
                idx = range(self.obs.size()[0])
                print(self.risks.size())
                
                idx_risky = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 1]
                idx_safe  = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 0]
                sample_idx = np.array(list(np.random.choice(idx_risky, sample_size/2)) + list(np.random.choice(idx_safe, sample_size/2)))
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}
                  

        def slice_data(self, min_idx, max_idx):
                idx = range(min_idx, max_idx)
                sample_idx = idx #np.random.choice(idx, sample_size)
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}        

        def save(self):
            torch.save(self.next_obs, os.path.join(self.data_path, "all_obs.pt"))
            torch.save(self.risks, os.path.join(self.data_path, "all_risks.pt"))



class ReplayBufferBalanced:
        def __init__(self, buffer_size=100000):
                self.obs_risky = None 
                self.next_obs_risky = None
                self.actions_risky = None 
                self.rewards_risky = None 
                self.dones_risky = None
                self.risks_risky = None 
                self.dist_to_fails_risky = None 
                self.costs_risky = None

                self.obs_safe = None 
                self.next_obs_safe = None
                self.actions_safe = None 
                self.rewards_safe = None 
                self.dones_safe = None
                self.risks_safe = None 
                self.dist_to_fails_safe = None 
                self.costs_safe = None

        def add_risky(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_risky = obs if self.obs_risky is None else torch.concat([self.obs_risky, obs], axis=0)
                self.next_obs_risky = next_obs if self.next_obs_risky is None else torch.concat([self.next_obs_risky, next_obs], axis=0)
                self.actions_risky = action if self.actions_risky is None else torch.concat([self.actions_risky, action], axis=0)
                self.rewards_risky = reward if self.rewards_risky is None else torch.concat([self.rewards_risky, reward], axis=0)
                self.dones_risky = done if self.dones_risky is None else torch.concat([self.dones_risky, done], axis=0)
                self.risks_risky = risk if self.risks_risky is None else torch.concat([self.risks_risky, risk], axis=0)
                self.costs_risky = cost if self.costs_risky is None else torch.concat([self.costs_risky, cost], axis=0)
                self.dist_to_fails_risky = dist_to_fail if self.dist_to_fails_risky is None else torch.concat([self.dist_to_fails_risky, dist_to_fail], axis=0)

        def add_safe(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_safe = obs if self.obs_safe is None else torch.concat([self.obs_safe, obs], axis=0)
                self.next_obs_safe = next_obs if self.next_obs_safe is None else torch.concat([self.next_obs_safe, next_obs], axis=0)
                self.actions_safe = action if self.actions_safe is None else torch.concat([self.actions_safe, action], axis=0)
                self.rewards_safe = reward if self.rewards_safe is None else torch.concat([self.rewards_safe, reward], axis=0)
                self.dones_safe = done if self.dones_safe is None else torch.concat([self.dones_safe, done], axis=0)
                self.risks_safe = risk if self.risks_safe is None else torch.concat([self.risks_safe, risk], axis=0)
                self.costs_safe = cost if self.costs_safe is None else torch.concat([self.costs_safe, cost], axis=0)
                self.dist_to_fails_safe = dist_to_fail if self.dist_to_fails_safe is None else torch.concat([self.dist_to_fails_safe, dist_to_fail], axis=0)

        
        def sample(self, sample_size):
                idx_risky = range(self.obs_risky.size()[0])
                idx_safe = range(self.obs_safe.size()[0])

                sample_risky_idx = np.random.choice(idx_risky, int(sample_size/2))
                sample_safe_idx = np.random.choice(idx_safe, int(sample_size/2))

                return {"obs": torch.cat([self.obs_risky[sample_risky_idx], self.obs_safe[sample_safe_idx]], 0),
                        "next_obs": torch.cat([self.next_obs_risky[sample_risky_idx], self.next_obs_safe[sample_safe_idx]], 0),
                        "actions": torch.cat([self.actions_risky[sample_risky_idx], self.actions_safe[sample_safe_idx]], 0),
                        "rewards": torch.cat([self.rewards_risky[sample_risky_idx], self.rewards_safe[sample_safe_idx]], 0),
                        "dones": torch.cat([self.dones_risky[sample_risky_idx], self.dones_safe[sample_safe_idx]], 0),
                        "risks": torch.cat([self.risks_risky[sample_risky_idx], self.risks_safe[sample_safe_idx]], 0),
                        "costs": torch.cat([self.costs_risky[sample_risky_idx], self.costs_safe[sample_safe_idx]], 0),
                        "dist_to_fail": torch.cat([self.dist_to_fails_risky[sample_risky_idx], self.dist_to_fails_safe[sample_safe_idx]], 0),}
        



                        

                
