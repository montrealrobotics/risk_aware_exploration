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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
    parser.add_argument("--validate-every", type=int, default=10,
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
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
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

    def forward(self, x):
        x = self.bnorm1(self.relu(self.fc1(x)))
        x = self.bnorm2(self.relu(self.fc2(x)))
        x = self.bnorm3(self.relu(self.fc3(x)))
        x = self.bnorm4(self.relu(self.fc4(x)))
        out = self.softmax(self.out(x))
        return out 



class CNNRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 5, padding="same") 
        self.conv4 = nn.Conv2d(32, 64, 5, padding="same") 
        self.fc1 = nn.Linear(64 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ## 60 * 40 
        x = self.pool(F.relu(self.conv1(x))) ## 113 * 113 
        x = self.pool(F.relu(self.conv2(x))) ## 56 * 56  
        x = self.pool(F.relu(self.conv3(x))) ## 28  * 28
        x = self.pool(F.relu(self.conv4(x))) ## 14 * 14
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = self.softmax(self.fc4(x))
        return x



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
        #Image.open(os.path.join(self.root_dir, label, files[idx]))
        X = np.hstack([k.ravel() for k in X.values()])
        X = torch.Tensor(np.array(X))

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

        



## Dataset and Dataloader 


### Splitting episodes into train and test 
def load_loaders(args):
    num_episodes = 1000
    episodes = list(range(1, num_episodes))
    shuffle(episodes)

    print(episodes)
    train_episodes = episodes[:int(0.8*num_episodes)]
    test_episodes  = episodes[int(0.8*num_episodes):]

    train_dataset = CostDataset(root_dir=os.path.join(args.data_path, "train"))
    # train_dataset = TrajDataset(root_dir=os.path.join(args.data_path, "train"))
    # train_dataset = BinCostDataset(root_dir=os.path.join(args.data_path, "train"))
    # train_dataset = TrajDataset(root_dir=args.data_path, dataset_type="cost", episode_list=train_episodes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

    test_dataset = CostDataset(root_dir=os.path.join(args.data_path, "test"), dataset_name="test")
    # test_dataset = TrajDataset(root_dir=os.path.join(args.data_path, "test"), dataset_name="test")
    # test_dataset = BinCostDataset(root_dir=os.path.join(args.data_path, "test"))
    # test_dataset = TrajDataset(root_dir=args.data_path, dataset_type="cost", episode_list=test_episodes)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader 

class RiskTrainer():
    def __init__(self, args, train_loader, test_loader, device=torch.device('cuda')):
        self.args = args
        self.test_schedule = args.validate_every 
        self.train_loader  = train_loader
        self.test_loader   = test_loader 
        self.device = device
        #self.model = CNNRisk()
        self.model = RiskEst(obs_size=54, fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                             fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=2)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, args.weight]).to(device))
        self.optim = optim.Adam(self.model.parameters(), args.learning_rate) 
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=args.lr_schedule)
        self.global_step = 0 


    def train(self):
        self.model.train()
        for ep in range(self.args.num_iterations):
            self.optim.step()
            self.scheduler.step()
            train_loss = 0
            for batch in self.train_loader:
                self.global_step += 1
                pred_y = self.model(batch[0].to(self.device))
                loss = self.criterion(pred_y, batch[1].to(self.device))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                with torch.no_grad():
                    train_loss += loss.item()
            print("Episode %d ---- Loss: %.4f"%(ep, train_loss))
            wandb.log({"train_loss": train_loss})
            if ep % self.test_schedule == 0:
                self.test()

    def test(self):
        self.model.eval()
        test_loss = 0
        pred, true = [], []
        for batch_idx, (X, y) in enumerate(self.test_loader):
            with torch.no_grad():
                pred_y = self.model(X.to(self.device))
                test_loss += self.criterion(pred_y, y).item()
                y_pred, y_true = torch.argmax(pred_y.squeeze()), torch.argmax(y.squeeze())
                pred.append(y_pred.item())
                true.append(y_true.item())
                #if y_pred.item() == 1 or y_true.item() == 1:
                #    print(y_pred.item(), y_true.item())
        pred, true = np.array(pred), np.array(true)
        f1 = f1_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        accuracy = accuracy_score(true, pred)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        print("-------------------------------------------------------------------------------------------------")
        print("Test Loss: %.4f"%test_loss)
        print()
        print("Accuracy %.4f   Precision: %.4f    Recall: %.4f     F1: %.4f"%(accuracy, precision, recall, f1))
        print()
        print("TP %.4f   FP: %.4f    FN: %.4f     TN: %.4f"%(tp, fp, fn, tn))
        print("-------------------------------------------------------------------------------------------------")
        wandb.log({"test_loss": test_loss})
        wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})
        wandb.log({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
        return test_loss



if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = parse_args()
    wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
    )
    train_loader, test_loader = load_loaders(args)
    risktrainer = RiskTrainer(args, train_loader, test_loader) 
    risktrainer.train()
