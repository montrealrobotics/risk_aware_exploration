import sys
import os
import argparse
import numpy as np
from random import shuffle
#import pybullet_envs  # noqa
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F

#from torch.utils.tensorboard import SummaryWriter





def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./traj",
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
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
        help="learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=100, 
        help="batch size for the stochastic gradient descent" ) 
    parser.add_argument("--validate-every", type=int, default=100,
        help="validate every x SGD steps")
    parser.add_argument("--num_iterations", type=int, default=100, 
        help="number of times to go over the entire dataset during training")
    parser.add_argument("--fc1_size", type=int, default=128, 
        help="size of the first layer of the mlp")
    parser.add_argument("--fc2_size", type=int, default=128,
        help="size of the second layer of the mlp")
    parser.add_argument("--fc3_size", type=int, default=128,
        help="size of the third layer of the mlp")
    parser.add_argument("--fc4_size", type=int, default=128,
        help="size of the fourth layer of the mlp")
    parser.add_argument("--lr_schedule", type=float, default=0.99,
        help="schedule for the learning rate decay " ) 
    return parser.parse_args()


class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=256, fc3_size=128, fc4_size=64, out_size=1):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        out = self.softmax(self.out(x))
        return out 



class CNNRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 5, padding="same") 
        self.fc1 = nn.Linear(32 * 7 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ## 60 * 40 
        x = self.pool(F.relu(self.conv1(x))) ## 30 * 20 
        x = self.pool(F.relu(self.conv2(x))) ## 15 * 10  
        x = self.pool(F.relu(self.conv3(x))) ## 7  * 5
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = self.softmax(self.fc4(x))
        return x



args = parse_args()

wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
        )

input_data = torch.load(os.path.join(args.data_path, "all_obs.pt"))
targets = torch.load(os.path.join(args.data_path, "all_cost.pt")) 

## Visual data 

#input_data = input_data[:, :7200]
#input_data = input_data.reshape((input_data.size()[0], 40, 60, 3))
#input_data = torch.transpose(input_data, 1, 3)

#targets = torch.load(os.path.join(args.data_path, "all_fear.pt"))
targets = targets.squeeze()


print(input_data.size(), targets.size())


max_data, min_data = torch.max(input_data), torch.min(input_data)
max_targets, min_targets = torch.max(targets), torch.min(targets)


## Max Min Normalization
input_data = (input_data - min_data) / (max_data - min_data) 
#targets = (targets - min_targets) / (max_targets - min_targets) 



idx = list(range(input_data.size()[0]))
shuffle(idx)
train_idx = idx[:int(len(idx)*0.7)]
test_idx = idx[int(len(idx)*0.7):]
train_data, train_targets = input_data[train_idx, :], targets[train_idx, :]
test_data,  test_targets  = input_data[test_idx,  :], targets[test_idx,  :]



#risk_est = CNNRisk()
risk_est = RiskEst(obs_size=input_data.size()[-1], fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                            fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=2)
#mse = nn.MSELoss()
index_targets = torch.argmax(targets, axis=1)
print(torch.sum(index_targets==0)/torch.sum(index_targets==1))
bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1., torch.sum(index_targets==0)/torch.sum(index_targets==1)]).long().cuda())

optimizer = optim.Adam(risk_est.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                          gamma=args.lr_schedule) # Multiplicative factor of learning rate decay.

bce.to('cuda')
#mse.to('cuda')
risk_est.to('cuda')
test_data.to('cuda')
test_targets.to('cuda')

print(train_data.size(), test_data.size())
def evaluate_model(model, data, targets, batch_size=1000):
    loss = torch.Tensor([0.0]).to('cuda') 
    for batch_idx in range(int(data.size()[0]/batch_size)):
        if batch_idx == data.size()[0]/batch_size: 
            batch_data = data[batch_idx*batch_size:,:].to('cuda')
            batch_targets = targets[batch_idx*batch_size:,:].to('cuda')
        else:
            batch_data = data[batch_idx*batch_size:(batch_idx+1)*batch_size,:].to('cuda')
            batch_targets = targets[batch_idx*batch_size:(batch_idx+1)*batch_size,:].to('cuda')
        batch_pred = model(batch_data)
        print(torch.mean(batch_pred), torch.var(batch_pred))
        loss += bce(batch_pred.to('cuda'), batch_targets.to('cuda')) 
        print("Test MSE Loss : ", loss) 
        return loss 


def train_model(model, data, targets, test_data, test_targets, batch_size=args.batch_size):
    global_step = 0 
    for ep in range(args.num_iterations):
        optimizer.step()
        scheduler.step()
        for batch_idx in range(int(data.size()[0]/batch_size)):
            global_step += 1 
            if batch_idx == data.size()[0]/batch_size:
                batch_data = data[batch_idx*batch_size:,:].to('cuda')
                batch_targets = targets[batch_idx*batch_size:,:].to('cuda')
            else:
                batch_data = data[batch_idx*batch_size:(batch_idx+1)*batch_size,:].to('cuda')
                batch_targets = targets[batch_idx*batch_size:(batch_idx+1)*batch_size,:].to('cuda')
            batch_pred = model(batch_data)
            loss = bce(batch_pred, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("Train Loss: ", loss)
            wandb.log({"Train MSE": loss}, step=global_step)
            if batch_idx % args.validate_every == 0:
                test_loss = evaluate_model(model, test_data, test_targets)
                wandb.log({"MSE": test_loss}, step=global_step)





train_model(risk_est, train_data, train_targets, test_data, test_targets, args.batch_size)
