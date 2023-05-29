import hydra 
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from models.ppo_continuous_action import train

    # Train model
    return train(cfg)


if __name__ == "__main__":
    main()
