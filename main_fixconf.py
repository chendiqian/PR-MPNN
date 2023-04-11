from ml_collections import ConfigDict
from sacred import Experiment

import wandb
from run import run, naming

ex = Experiment()


@ex.automain
def main(fixed):
    args = ConfigDict(dict(fixed))
    hparams = naming(args)

    wandb_name = args.wandb_name if hasattr(args, "wandb_name") else "imle_ablate"
    wandb.init(project=wandb_name, mode="online" if args.use_wandb else "disabled",
               config=args.to_dict(),
               name=hparams,
               entity="mls-stuttgart")

    run(wandb, args)
