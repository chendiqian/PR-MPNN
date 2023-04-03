import torch.optim as optim
from data.data_utils import get_cosine_schedule_with_warmup


def make_get_embed_opt(args):
    def get_embed_opt(model):
        if model is None:
            return None, None

        # default
        optimizer_embd = optim.AdamW(model.parameters(),
                                     lr=args.imle_configs.embd_lr,
                                     weight_decay=args.imle_configs.reg_embd)

        # unless otherwise defined
        if hasattr(args.imle_configs, 'emb_optim'):
            if args.imle_configs.emb_optim == 'adam':
                optimizer_embd = optim.Adam(model.parameters(),
                                            lr=args.imle_configs.embd_lr,
                                            weight_decay=args.imle_configs.reg_embd)
            elif args.imle_configs.emb_optim == 'sgd':
                optimizer_embd = optim.SGD(model.parameters(),
                                           lr=args.imle_configs.embd_lr,
                                           weight_decay=args.imle_configs.reg_embd)
            else:
                raise ValueError

        # default
        scheduler_embd = get_cosine_schedule_with_warmup(optimizer_embd, 50, args.max_epochs)
        # unless otherwise defined
        if hasattr(args.imle_configs, 'emb_scheduler'):
            if args.imle_configs.emb_scheduler == 'step':
                scheduler_embd = optim.lr_scheduler.MultiStepLR(optimizer_embd,
                                                                args.lr_steps,
                                                                gamma=args.lr_decay_rate if hasattr(
                                                                    args,
                                                                    'lr_decay_rate')
                                                                else 0.1 ** 0.5)

        return optimizer_embd, scheduler_embd

    return get_embed_opt


def make_get_opt(args):
    def get_opt(model):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

        if hasattr(args, 'optim'):
            if args.optim == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg)
            else:
                raise ValueError

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   args.lr_steps,
                                                   gamma=args.lr_decay_rate if hasattr(
                                                       args, 'lr_decay_rate')
                                                   else 0.1 ** 0.5)
        return optimizer, scheduler
    return get_opt
