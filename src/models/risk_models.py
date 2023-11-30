
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }

    return activation_dict[name]


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


class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=True, activation='relu', model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.model_type = model_type
        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        #self.fc2 = nn.Linear(fc1_size, fc2_size)
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

    def forward(self, x, action=None):
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            if self.model_type == "state_action_risk":
                x1 = self.bnorm1_action(self.activation(self.fc1_action(action)))
                x = torch.cat([x, x1], axis=1)
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            if self.model_type == "state_action_risk":
                x1 = self.activation(self.fc1_action(action))
                x = torch.cat([x, x1], axis=1)

            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))

        out = self.logsoftmax(self.out(x))
        return out




class BayesRiskEstCont(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=1, model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.model_type = model_type
        self.action_size = action_size

        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        self.mean_fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean_fc4 = nn.Linear(fc3_size, fc4_size)
        self.mean_out = nn.Linear(fc4_size, out_size)

        self.logvar_fc3 = nn.Linear(fc2_size, fc3_size)
        self.logvar_fc4 = nn.Linear(fc3_size, fc4_size)
        self.logvar_out = nn.Linear(fc4_size, out_size)


        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.mean_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.mean_bnorm4 = nn.BatchNorm1d(fc4_size)

        #self.var_bnorm1 = nn.BatchNorm1d(fc1_size)
        #self.var_bnorm2 = nn.BatchNorm1d(fc2_size)
        self.var_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.var_bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, action=None):
        x = self.bnorm1(self.relu(self.fc1(x)))
        if self.model_type == "state_action_risk":
            x1 = self.bnorm1_action(self.relu(self.fc1_action(action)))
            x = torch.cat([x, x1], axis=1)

        x = self.bnorm2(self.relu(self.fc2(x)))

        mean  = self.mean_bnorm3(self.relu(self.mean_fc3(x)))
        mean  = self.mean_bnorm4(self.relu(self.mean_fc4(mean)))
        mean  = self.sigmoid(self.mean_out(mean))

        logvar = self.var_bnorm3(self.relu(self.logvar_fc3(x)))
        logvar = self.var_bnorm4(self.relu(self.logvar_fc4(x)))
        logvar = self.sigmoid(self.logvar_out(x))

        #x = self.bnorm3(self.relu(self.dropout(self.fc3(x))))
        #x = self.bnorm4(self.relu(self.dropout(self.fc4(x))))
        #out = self.logsoftmax(self.out(x))
        return mean, logvar

