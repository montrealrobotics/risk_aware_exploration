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

from src.models.risk_models import * 
from src.utils import * 
from src.datasets.risk_datasets import *




def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./traj",
        help="the name of this experiment")
    parser.add_argument("--env", type=str, default="SafetyCarButton1Gymnasium-v0",
        help="the name of this experiment")
    parser.add_argument("--dataset_type", type=str, default="state_risk",
        help="what dataset to use state or state_action?")
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
    parser.add_argument("--batch_size", type=int, default=1000, 
        help="batch size for the stochastic gradient descent" ) 
    parser.add_argument("--validate-every", type=int, default=2000,
        help="validate every x SGD steps")
    parser.add_argument("--num_iterations", type=int, default=100, 
        help="number of times to go over the entire dataset during training")
    parser.add_argument("--obs-size", type=int, default=72, 
        help="size of the first layer of the mlp")
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
    parser.add_argument("--model-type", type=str, default="mlp",
                    help="which network to use for the risk model")
    parser.add_argument("--fear_radius", type=int, default=5, help="radius around the dangerous objects to consider fearful. ")
    parser.add_argument("--continuous-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    
    return parser.parse_args()





### Splitting episodes into train and test 
def load_loaders(args):
    dataset = torch.load(os.path.join(args.data_path, args.env, "all_%s.pt"%args.dataset_type))
    np.random.seed(args.seed)
    dataset_size = dataset.size()[0]

    idx = list(range(dataset.size()[0]))
    shuffle(idx)

    train_idx = idx[:int(0.8*dataset_size)]
    test_idx = idx[int(0.8*dataset_size):]

    train_dataset = dataset[train_idx,:]
    test_dataset = dataset[test_idx, :]

    if args.dataset_type == "state_action_risk":
        train_dataset = RiskDataset(train_dataset, action=True)
        test_dataset = RiskDataset(test_dataset, action=True)
        
    else:
        train_dataset = RiskDataset(train_dataset, action=False)
        test_dataset = RiskDataset(test_dataset, action=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cuda'))#, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    return train_loader, test_loader 



class RiskTrainer():
    def __init__(self, args, train_loader, test_loader, device=torch.device('cuda')):
        self.args = args
        self.test_schedule = args.validate_every 
        self.train_loader  = train_loader
        self.test_loader   = test_loader 
        self.device = device
        if args.model_type == "mlp":
            if args.continuous_risk:
                self.model = RiskEst(obs_size=args.obs_size, fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                                fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=1, continuous_risk=True)
            else:
                self.model = RiskEst(obs_size=args.obs_size, fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                                fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=2, continuous_risk=False)
        elif args.model_type == "bayesian":
            if args.continuous_risk:
                self.model = BayesRiskEst1(obs_size=args.obs_size, fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                                fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=1)
            else:
                self.model = BayesRiskEst(obs_size=args.obs_size, fc1_size=args.fc1_size, fc2_size=args.fc2_size,\
                                fc3_size=args.fc3_size, fc4_size=args.fc4_size, out_size=2)
        if args.model_type == "bayesian":
            if args.continuous_risk:
                self.criterion = nn.GaussianNLLLoss()
            else:
                self.criterion = nn.NLLLoss(weight=torch.Tensor([1, args.weight]).to(device))
        else:
            if args.continuous_risk:
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, args.weight]).to(device))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, args.weight]).to(device))
        self.optim = optim.Adam(self.model.parameters(), args.learning_rate) 
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=args.lr_schedule)
        self.global_step = 0 


    def train(self):
        self.model.train()
        for ep in range(self.args.num_iterations):
            self.optim.step()
            self.scheduler.step()
            train_loss, best_val_loss = 0, 9999999
            pred, true = [], []
            for batch in tqdm.tqdm(self.train_loader):
                if self.args.continuous_risk and self.args.model_type =="bayesian":
                    pred_mu, pred_logvar = self.model(batch[0].to(self.device).squeeze())
                else:
                    pred_y = self.model(batch[0].to(self.device).squeeze())
                if not self.args.continuous_risk:
                    y_pred, y_true = torch.argmax(pred_y, axis=1), torch.argmax(batch[1].squeeze(), axis=1)
                    pred.extend(list(y_pred.detach().cpu().numpy()))
                    true.extend(list(y_true.detach().cpu().numpy()))

                if self.args.model_type == "bayesian":
                    if self.args.continuous_risk:
                        loss = self.criterion(pred_mu, batch[1].squeeze().to(self.device), torch.exp(pred_logvar))
                    else:
                        loss = self.criterion(pred_y, torch.argmax(batch[1].squeeze(), axis=1).to(self.device))
                else:
                    loss = self.criterion(pred_y, batch[1].squeeze().to(self.device))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                with torch.no_grad():
                    train_loss += loss.item()
                ## Validation Phase 
                if self.global_step % self.test_schedule == 0:
                    val_loss = self.test()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "agent.pt"))
                        wandb.save("agent.pt")
                self.global_step += 1
                ##---------------------------------------------------------------------------------------##
            pred, true = np.array(pred), np.array(true)
            accuracy, precision, recall = accuracy_score(true, pred), precision_score(true, pred), recall_score(true, pred)
            print("-----------------------------Training Scores ----------------------------------------")
            print("Episode %d ---- Loss: %.4f"%(ep, train_loss))
            print("Accuracy %.4f | Precision %.4f | Recall %.4f"%(accuracy, precision, recall))
            wandb.log({"train_loss": train_loss}, step=self.global_step)
            wandb.log({"train_precision": precision, "train_accuracy": accuracy, "train_recall": recall}, step=self.global_step)


    def test(self):
        self.model.eval()
        test_loss = 0
        pred, true = [], []
        for batch_idx, (X, y) in enumerate(self.test_loader):
            with torch.no_grad():
                if self.args.continuous_risk and self.args.model_type =="bayesian":
                    pred_mu, pred_logvar = self.model(X.to(self.device))
                else:
                    pred_y = self.model(X.to(self.device))

                if self.args.model_type == "bayesian":
                    if self.args.continuous_risk:
                        test_loss += self.criterion(pred_mu, y.squeeze().to(self.device), torch.exp(pred_logvar))
                    else:
                        test_loss += self.criterion(pred_y, torch.argmax(y.squeeze(), axis=1).to(self.device))
                else:
                    test_loss += self.criterion(pred_y.squeeze(), y.squeeze().to(self.device)).item()

                if not self.args.continuous_risk:
                    y_pred, y_true = torch.argmax(pred_y, axis=1), torch.argmax(y.squeeze(), axis=1)
                    pred.extend(list(y_pred.detach().cpu().numpy()))
                    true.extend(list(y_true.detach().cpu().numpy()))
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
        wandb.log({"test_loss": test_loss}, step=self.global_step)
        wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}, step=self.global_step)
        wandb.log({"tp": tp, "fp": fp, "tn": tn, "fn": fn}, step=self.global_step)
        return test_loss



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
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