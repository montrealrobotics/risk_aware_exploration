import os
import sys
import pickle
import tqdm
import numpy as np


source_dir = sys.argv[1]
dest_dir = sys.argv[2]
train_test_ratio = 0.8



for label in ["safe", "unsafe"]:
    for mode in ["train", "test"]:
        os.system("rm -rf %s"%(os.path.join(dest_dir, mode, label)))
        os.makedirs(os.path.join(dest_dir, mode, label))

for i, folder in tqdm.tqdm(enumerate(os.listdir(source_dir))):
    folder_path = os.path.join(source_dir, "traj_%d"%i)
    for j, file_name in enumerate(os.listdir(os.path.join(folder_path, "lidar"))):
        info = pickle.load(open(os.path.join(folder_path, "info", "%d.pkl"%j), "rb"))

        mode = "train" if np.random.randn() <= train_test_ratio else "test"
        label = "safe" if info["cost"] == 0 else "unsafe"

        os.system("cp -r %s %s"%(os.path.join(folder_path, "lidar", "%d.pkl"%j), os.path.join(dest_dir, mode, label, "%d_%d.pkl"%(i,j))))
        # os.system("cp -r %s %s"%(os.path.join(folder_path, "info", "%d.pkl"%j), os.path.join(dest_dir, mode, label, "%d_%d.pkl"%(i,j))))

