from ml_collections import ConfigDict

import wandb
from data.data_utils import unflatten
from run import run

hyperparameter_defaults = {}


if __name__ == '__main__':
    wandb.init(
        config=hyperparameter_defaults,
        mode="online",
    )
    args = ConfigDict(unflatten(wandb.config))
    run(wandb, args)
