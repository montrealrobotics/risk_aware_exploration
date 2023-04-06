import numpy as np
import os 
import sys 
import random 



root_dir = str(sys.argv[1])

train_path = os.path.join(root_dir, "train")
test_path = os.path.join(root_dir, "test")

for label in ["0", "1"]:
    os.makedirs(os.path.join(train_path, label))
    os.makedirs(os.path.join(test_path, label))

for label in ["0", "1"]:
    source_path = os.path.join(root_dir, label)
    files = os.listdir(source_path)
    for file_name in files:
        if np.random.randn() < 0.2:
            os.system("cp -r %s %s"%(os.path.join(source_path, file_name), os.path.join(test_path, label)))
        else:
            os.system("cp -r %s %s"%(os.path.join(source_path, file_name), os.path.join(train_path, label)))