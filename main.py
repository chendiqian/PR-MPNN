from ml_collections import ConfigDict

import wandb
from data.data_utils import unflatten, set_nonetype
from run import run

hyperparameter_defaults = {}


if __name__ == '__main__':
    wandb.init(
        config=hyperparameter_defaults,
        mode="online",
    )
    args = ConfigDict(set_nonetype(unflatten(wandb.config)))
    run(wandb, args)
