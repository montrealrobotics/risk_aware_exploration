import os
import pickle
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

