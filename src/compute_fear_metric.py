import os
import sys
import torch 
import numpy as np 




data_path = sys.argv[1]

def calculate_fear(costs):
    ''''

    Calculating fear by going back in time and counting the distance
    to the nearest positive cost forward in time.

    '''
    fear = torch.full(costs.size(), costs.size()[0])
    counter = costs.size()[0]
    flag = 0
    for i in reversed(range(costs.size()[0])):
        if costs[i] > 0:
            counter = 0
            flag = 1
        elif counter < costs.size()[0] and flag == 1:
            counter += 1
        elif flag == 0:
            fear[i] = 9999
            continue
        fear[i] = counter

    return fear




for run_name in os.listdir(data_path):
    print(run_name)
    run_path = os.path.join(data_path, run_name)
    costs = torch.load(os.path.join(run_path, "costs.pt")) 
    fear =  torch.zeros_like(costs)
    for episode in range(costs.size()[-1]):
        fear[:,episode] = calculate_fear(costs[:,episode])
    torch.save(fear, os.path.join(run_path, "fear.pt")) 




