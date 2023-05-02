import sys
import os 
#import torch 
import argparse 
import tqdm
import pickle
import numpy as np

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(),"traj"), help="path to the data directory")
    parser.add_argument("--dest_path", type=str, default="./dest_traj", help="destination directory to store processed files") 
    parser.add_argument("--fear_radius", type=int, default=12, help="radius around the dangerous objects to consider fearful. ")
    parser.add_argument("--filter_end", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="whether to filter the end of the peisodes where no information exists.")
    parser.add_argument("--binary_fear", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to keep fear as a binary 0 / 1 value.") 
    return parser.parse_args() 



def operate_traj(episode, root_path, storage_path, fear_radius=12, split_train_test=True):
    if split_train_test:
        store = "train" if np.random.random() <= 0.8 else "test" 
    traj_path = os.path.join(root_path, "traj_%d"%episode)
    info_path = os.path.join(traj_path, "info")
    rgb_path = os.path.join(traj_path, "rgb")
    lidar_path = os.path.join(traj_path, "lidar")
    flag, counter = 0, fear_radius
    for i in reversed(range(len(os.listdir(info_path)))):
        info = pickle.load(open(os.path.join(info_path, "%d.pkl"%i), "rb"))
        if info["cost"] > 0: 
            flag = 1
            counter = fear_radius
        elif flag == 1:
            counter -= 1 
        if counter <= 0: 
            flag = 0 
        if flag == 0: 
            os.system("cp -r %s %s"%(os.path.join(rgb_path, "%d.png"%i), os.path.join(storage_path, store, "safe", "%d_%d.png"%(episode, i))))
            #os.system("cp -r %s %s"%(os.path.join(lidar_path, "%d.pkl"%i), os.path.join(storage_path, store, "safe", "%d_%d.pkl"%(episode, i))))
        else:
            os.system("cp -r %s %s"%(os.path.join(rgb_path, "%d.png"%i), os.path.join(storage_path, store, "unsafe", "%d_%d.png"%(episode, i))))
            #os.system("cp -r %s %s"%(os.path.join(lidar_path, "%d.pkl"%i), os.path.join(storage_path, store, "unsafe", "%d_%d.pkl"%(episode, i))))



if __name__ == "__main__":
    args = parse_args()
    os.system("rm -rf %s"%(os.path.join(args.dest_path, "train", "safe")))
    os.system("rm -rf %s"%(os.path.join(args.dest_path, "train", "unsafe")))
    os.system("rm -rf %s"%(os.path.join(args.dest_path, "test", "safe")))
    os.system("rm -rf %s"%(os.path.join(args.dest_path, "test", "unsafe")))

    os.makedirs(os.path.join(args.dest_path, "train", "safe"))
    os.makedirs(os.path.join(args.dest_path, "train", "unsafe"))
    os.makedirs(os.path.join(args.dest_path, "test", "safe"))
    os.makedirs(os.path.join(args.dest_path, "test", "unsafe"))

    for traj in tqdm.tqdm(range(len(os.listdir(args.data_path)))):
        operate_traj(traj, args.data_path, args.dest_path, args.fear_radius)





'''
all_obs, all_fear, all_cost = None, None, None
for run_name in os.listdir(args.data_path):
    print(run_name)
    run_path = os.path.join(args.data_path, run_name)
    fear = torch.load(os.path.join(run_path, "fear.pt"), map_location=torch.device("cpu"))
    cost = torch.load(os.path.join(run_path, "costs.pt"), map_location=torch.device("cpu"))
    obs  = torch.load(os.path.join(run_path, "obs.pt"), map_location=torch.device("cpu"))
    print(cost.size(), fear.size(), obs.size())
    cost, fear, obs = torch.flatten(cost), torch.flatten(fear), torch.flatten(obs, start_dim=0, end_dim=1)
    ## Removing the data points where the fear cannot be calculated denoted by token 9999
    if args.filter_end:
        obs = obs[fear!=9999,:]
        cost = cost[fear!=9999]
        fear = fear[fear!=9999]
    fear = torch.unsqueeze(fear, 1)
    all_obs = obs if all_obs is None else torch.cat([all_obs, obs], axis=0)
    all_fear = fear if all_fear is None else torch.cat([all_fear, fear], axis=0)
    all_cost = cost if all_cost is None else torch.cat([all_cost, cost], axis=0)

print(all_fear.size())
print(torch.sum(all_fear < args.fear_radius) / torch.sum(all_fear > args.fear_radius))

## Convert costs into binary 
all_cost[all_cost>0] = 1

if args.binary_fear: 
    all_fear[all_fear<=args.fear_radius] = 1
    all_fear[all_fear>args.fear_radius] = 0

## Converting to one-hot encoding 

all_fear = all_fear.to(torch.int64)
all_fear = torch.nn.functional.one_hot(all_fear, num_classes=2)
all_fear = all_fear.float()

all_cost = all_cost.to(torch.int64)
all_cost = torch.nn.functional.one_hot(all_cost, num_classes=2)
all_cost = all_cost.float()


print(all_fear.size(), all_obs.size())
torch.save(all_obs, os.path.join(args.data_path, "all_obs.pt"))
torch.save(all_cost, os.path.join(args.data_path, "all_cost.pt"))
torch.save(all_fear, os.path.join(args.data_path, "all_fear.pt"))

'''
