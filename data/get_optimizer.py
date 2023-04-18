import torch.optim as optim
from data.data_utils import get_cosine_schedule_with_warmup, MyPlateau


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
            scheduler_embd = None
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
        else:
            raise NotImplementedError

        return optimizer, scheduler
    return get_opt
