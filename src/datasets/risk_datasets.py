import sys
import os
import pickle
import argparse
import numpy as np
from random import shuffle
from PIL import Image 
from sklearn.metrics import * 

#import pybullet_envs  # noqa
import wandb
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

class TrajDataset(Dataset):
    def __init__(self, root_dir, dataset_name="train"):
        self.root_dir = root_dir
        self.files_list = os.listdir(root_dir)
        self.dataset_name = dataset_name 

    def __len__(self):
        return len(self.files_list)*1000

    def __getitem__(self, idx):
        idx = idx % len(self.files_list)
        traj_idx = self.files_list[idx]
        idx = np.random.randint(0, 1000)
        x = Image.open(os.path.join(self.root_dir, traj_idx, "rgb", "%d.png"%idx))
        x = torch.transpose(torch.Tensor(np.array(x)), 0, 2)
        info = pickle.load(open(os.path.join(self.root_dir, traj_idx, "info", "%d.pkl"%idx), "rb"))
        y =  torch.zeros(2)
        # cost = 0 if info["cost"] == 0 else 1 
        y[int(info["cost"])] = 1
        #if info["cost"] == 1 and self.dataset_name == "test":
        #    print(y)
        #print(y)
        return x, y
    
class CostDataset(Dataset):
    def __init__(self, root_dir, dataset_name="train"):
        self.root_dir = root_dir
        self.safe_files = os.listdir(os.path.join(root_dir, "safe"))
        self.unsafe_files = os.listdir(os.path.join(root_dir, "unsafe"))

        self.dataset_name = dataset_name 

    def __len__(self):
        return int((len(self.safe_files) + len(self.unsafe_files)))

    def __getitem__(self, idx):
        y = np.random.choice([0, 1])
        label = "safe" if y == 0 else "unsafe"
        files = self.safe_files if y == 0 else self.unsafe_files
        idx = idx % int(len(os.listdir(os.path.join(self.root_dir, label))))
        X = pickle.load(open(os.path.join(self.root_dir, label, files[idx]), "rb"))
        #X = Image.open(os.path.join(self.root_dir, label, files[idx]))
        #X = np.hstack([k.ravel() for k in X.values()])
        X = torch.Tensor(np.array(X))
        #X = torch.transpose(torch.Tensor(np.array(X)), 2, 0)
        Y = torch.zeros(2)
        Y[y] = 1

        return X, Y

class BinCostDataset(Dataset):
    def __init__(self, root_dir, dataset_type="png"):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.files_zero = os.listdir(os.path.join(self.root_dir, "0"))
        self.files_one  = os.listdir(os.path.join(self.root_dir, "1"))

    def __len__(self):
        return len(self.files_zero) + len(self.files_one)  

    def __getitem__(self, idx):
        y = torch.zeros(2)
        ## Sampling equally from both classes 
        if np.random.randn() <= 0.5:
            label = "1"
            y[1] = 1.
            file_list = self.files_one
        else:
            label = "0"
            y[0] = 1
            file_list = self.files_zero
        idx = idx % len(file_list)

        if self.dataset_type == "png":
            x = Image.open(os.path.join(self.root_dir, label, file_list[idx]))
            x = torch.transpose(torch.Tensor(np.array(x)), 0, 2)
        elif self.dataset_type == "tensor":
            x = torch.load(os.path.join(self.root_dir, label, file_list[idx]))
        return x, y

        
class RiskStateDataset(nn.Module):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, idx):
        y = torch.zeros(2)
        y[int(self.targets[idx][0])] = 1.0
        return self.inputs[idx], y


class RiskStateActionDataset(nn.Module):
    def __init__(self, states, actions, targets):
        self.states = states
        self.actions = actions
        self.targets = targets

    def __len__(self):
        return self.states.size()[0]

    def __getitem__(self, idx):
        y = torch.zeros(2)
        y[int(self.targets[idx][0])] = 1.0
        return self.states[idx], self.actions[idx], y



class RiskDataset(nn.Module):
    def __init__(self, data, action=False, action_size=2, one_hot=True):
        self.data = data
        self.one_hot = one_hot 
        self.action = action 
        self.action_size = action_size

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        if self.one_hot:
            y = torch.zeros(2)
            y[int(self.data[idx][-1])] = 1.0
        else:
            y = self.data[idx][-1]
        if self.action:
            return self.data[idx][:-(self.action_size+1)], self.data[idx][-(self.action_size+1):-1], y
        else:
            return self.data[idx][:-1], y



class RiskyDataset(nn.Module):
    def __init__(self, obs, actions, risks, action=False, continuous_risk=False, fear_clip=None, fear_radius=None, one_hot=True):
        self.obs = obs
        self.risks = risks
        self.actions = actions
        self.one_hot = one_hot
        self.action = action
        self.fear_clip = fear_clip 
        self.fear_radius = fear_radius
        self.continuous_risk = continuous_risk

    def __len__(self):
        return self.obs.size()[0]

    def get_binary_risk(self, idx):
        if self.one_hot:
            y = torch.zeros(2)
            y[int(self.risks[idx] <= self.fear_radius)] = 1.0
        else:
            y = int(self.risks[idx] <= self.fear_radius)
    
    def get_continuous_risk(self, idx):
        if self.fear_clip is not None:
            return 1. / torch.clip(self.risks[idx], 0, self.fear_clip)
        else:
            return 1. / self.risks[idx]

    def __getitem__(self, idx):
        if self.continuous_risk:
            y = self.get_continuous_risk(idx)
        else:
            y = self.get_binary_risk(idx)

        if self.action:
            return self.obs[idx], self.actions[idx], y
        else:
            return self.obs[idx], y
~
