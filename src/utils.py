import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU()
        "sigmoid": nn.Sigmoid()
        "tanh": nn.Tanh()
        "softmax": nn.Softmax(dim=1)
        "logsoftmax": nn.LogSoftmax(dim=1)
    }