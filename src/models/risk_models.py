
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import get_activation

class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=False, activation='relu', continuous_risk=False):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.continuous_risk = continuous_risk

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
        self.activation = get_activation(activation)
        self.softmax = get_activation("softmax")

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))    
        
        if self.continuous_risk:
            out = self.sigmoid(self.out(x))
        else:
            out = self.softmax(self.out(x))
        return out

        return out

class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=False, activation='relu'):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
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
        self.activation = get_activation(activation)

        self.logsoftmax = get_activation("logsoftmax")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))        

        out = self.logsoftmax(self.out(x))
        return out


class BayesRiskEst1(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=256, fc3_size=128, fc4_size=64, out_size=1):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)


        self.mean_fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean_fc4 = nn.Linear(fc3_size, fc4_size)
        self.mean_out = nn.Linear(fc4_size, out_size)

        self.logvar_fc3 = nn.Linear(fc2_size, fc3_size)
        self.logvar_fc4 = nn.Linear(fc3_size, fc4_size)
        self.logvar_out = nn.Linear(fc4_size, out_size)


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
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean  = self.relu(self.mean_fc3(x))
        mean  = self.relu(self.mean_fc4(mean))
        mean  = self.sigmoid(self.mean_out(mean))

        logvar = self.relu(self.logvar_fc3(x))
        logvar = self.relu(self.logvar_fc3(x))
        logvar = self.sigmoid(self.logvar_out(x))

        #x = self.bnorm3(self.relu(self.dropout(self.fc3(x))))
        #x = self.bnorm4(self.relu(self.dropout(self.fc4(x))))
        #out = self.logsoftmax(self.out(x))
        return mean, logvar

