import os 
import sys 
import pickle 
import argparse 
import tqdm
import numpy as np


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(),"traj"), help="path to the data directory")
    #parser.add_argument("--dest_path", type=str, default="./dest_traj", help="destination directory to store processed files")
    parser.add_argument("--fear_radius", type=int, default=12, help="radius around the dangerous objects to consider fearful. ")
    parser.add_argument("--filter_end", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="whether to filter the end of the peisodes where no information exists.")
    parser.add_argument("--binary_fear", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to keep fear as a binary 0 / 1 value.")
    return parser.parse_args()


def compute_risk(root_dir, fear_radius):
    total_fail = 0
    total_risk = 0
    for run in tqdm.tqdm(os.listdir(root_dir)):
        run_path = os.path.join(root_dir, run)
        for traj in tqdm.tqdm(os.listdir(run_path)):
            counter = 64
            traj_path = os.path.join(run_path, traj)
            for idx in reversed(range(0, len(os.listdir(os.path.join(traj_path, "info"))))):
                file_path = os.path.join(traj_path, "info", "%d.pkl"%idx)
                info = pickle.load(open(file_path, "rb"))
                if info["cost"] > 0:
                    total_fail +=1
                    total_risk +=1
                    risk = 1.0
                    counter = 0 
                else:
                    if counter <= fear_radius:
                        risk = 1.0
                        total_risk += 1
                    else:
                        risk = 0.0 
                    counter += 1 
                info["risk_%d"%fear_radius] = risk
                #print(idx, info["cost"], info["risk_%d"%fear_radius])
                pickle.dump(info, open(file_path, "wb"))
    print(total_fail, total_risk)





if __name__ == "__main__":
    args = parse_args()
    compute_risk(args.data_path, args.fear_radius)


