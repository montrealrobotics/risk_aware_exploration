import hydra 
import sys
import os
from omegaconf import DictConfig


os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/.mujoco/mujoco200/bin")
os.chdir("/home/mila/k/kaustubh.mani/scratch/projects/risk_aware_exploration")
#sys.path.append("/home/mila/k/kaustubh.mani/scratch/projects/risk_aware_exploration")

@hydra.main(version_base=None, config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from models.ppo_continuous_action import train
#    print(os.listdir("./"))
#    return 1
    # Train model
    return train(cfg)


if __name__ == "__main__":
    main()
