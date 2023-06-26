import math

import torch.optim as optim
from torch.optim import Optimizer


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1):
    """
    https://github.com/rampasek/GraphGPS/blob/95a17d57767b34387907f42a43f91a0354feac05/graphgps/optimizer/extra_optimizers.py#L158

    Args:
        optimizer:
        num_warmup_steps:
        num_training_steps:
        num_cycles:
        last_epoch:

    Returns:

    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class MyPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr


def make_get_embed_opt(args):
    def get_embed_opt(model):
        if model is None:
            return None, None

        if args.imle_configs.emb_optim == 'adam':
            optimizer_embd = optim.Adam(model.parameters(),
                                        lr=args.imle_configs.embd_lr,
                                        weight_decay=args.imle_configs.reg_embd)
        elif args.imle_configs.emb_optim == 'sgd':
            optimizer_embd = optim.SGD(model.parameters(),
                                       lr=args.imle_configs.embd_lr,
                                       weight_decay=args.imle_configs.reg_embd)
        elif args.imle_configs.emb_optim == 'adamw':
            optimizer_embd = optim.AdamW(model.parameters(),
                                         lr=args.imle_configs.embd_lr,
                                         weight_decay=args.imle_configs.reg_embd)
        else:
            raise ValueError

        # unless otherwise defined
        if args.imle_configs.emb_scheduler == 'step':
            scheduler_embd = optim.lr_scheduler.MultiStepLR(optimizer_embd,
                                                            eval(args.lr_steps),
                                                            gamma=0.1 ** 0.5)
        elif args.imle_configs.emb_scheduler == 'cosine':
            scheduler_embd = get_cosine_schedule_with_warmup(optimizer_embd, 50,
                                                             args.max_epochs)
        elif args.imle_configs.emb_scheduler == 'None' or args.imle_configs.emb_scheduler is None:
            scheduler_embd = optim.lr_scheduler.LambdaLR(optimizer_embd, lambda *args: 1.)
        else:
            raise ValueError
        return optimizer_embd, scheduler_embd

    return get_embed_opt


def make_get_opt(args):
    def get_opt(model):
        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg)
        elif args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        else:
            raise ValueError

        if args.lr_decay.scheduler == 'plateau':
            scheduler = MyPlateau(optimizer,
                                  mode=args.lr_decay.mode,
                                  factor=args.lr_decay.decay_rate,
                                  patience=args.lr_decay.patience,
                                  threshold_mode='abs',
                                  cooldown=0,
                                  min_lr=1.e-5)
            setattr(scheduler, 'lr_target', args.lr_decay.target)
        elif args.lr_decay.scheduler == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       eval(args.lr_decay.steps),
                                                       gamma=0.1 ** 0.5)
        elif args.lr_decay.scheduler == 'cyclic':
            assert isinstance(optimizer, optim.SGD), "CyclicLR only works with SGD"
            assert hasattr(args.lr_decay, "min_lr") and hasattr(args.lr_decay, "max_lr"), 'min_lr and max_lr must be defined'
            assert args.lr_decay.min_lr < args.lr_decay.max_lr, "min_lr must be smaller than max_lr"
            scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                                    base_lr=args.lr_decay.min_lr,
                                                    max_lr=args.lr_decay.max_lr)
        else:
            raise NotImplementedError

        return optimizer, scheduler
    return get_opt
