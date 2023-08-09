import os
import pickle
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


                        

                