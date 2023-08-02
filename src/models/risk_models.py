
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import get_activation

class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=False, activation='relu'):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm

        self.acti
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
        
        out = self.softmax(self.out(x))
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