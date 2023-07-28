import hydra
import sys
import os
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from pkgs.cleanrl.cleanrl.ppo_continuous_action import train
    return train(cfg)


if __name__ == "__main__":
    main()
