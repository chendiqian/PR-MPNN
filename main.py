from typing import Tuple
import logging
import os
import yaml
from ml_collections import ConfigDict
from sacred import Experiment
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from numpy import mean as np_mean
from numpy import std as np_std

from models.get_model import get_model
from training.trainer import Trainer
from data.get_data import get_data
from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.data_utils import SyncMeanTimer

ex = Experiment()


def get_logger(folder_path: str) -> logging.Logger:
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler(os.path.join(folder_path, 'training_logs.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def naming(args) -> str:
    name = f'{args.dataset}_{args.model}_'
    name += f'outlayer_{args.num_convlayers}_'
    name += f'innlayer_{args.sample_configs.inner_layer}_'

    if args.sample_configs.sample_policy is None:
        name += 'normal'
        return name

    if args.imle_configs is not None:
        name += f'IMLE_beta{args.imle_configs.beta}_'
        name += f'H{args.imle_configs.emb_hid_size}'
        name += f'L{args.imle_configs.emb_num_layer}'
        name += f'DP{args.imle_configs.dropout}'
        name += f'Beta{args.imle_configs.beta}_'
    else:
        name += 'OnTheFly_'

    name += f'policy_{args.sample_configs.sample_policy}_'
    name += f'samplek_{args.sample_configs.sample_k}_'
    name += f'sub2nAggr_{args.sample_configs.subgraph2node_aggr}_'
    return name


def prepare_exp(folder_name: str, num_run: int, num_fold: int) -> Tuple[SummaryWriter, str]:
    run_folder = os.path.join(folder_name, f'run{num_run}_fold{num_fold}_{"".join(str(datetime.now()).split(":"))}')
    os.mkdir(run_folder)
    writer = SummaryWriter(run_folder)
    return writer, run_folder


@ex.automain
def run(fixed):
    fixed = dict(fixed)
    root_dir = fixed['dataset'].lower()
    with open(f"./configs/{root_dir}/common_configs.yaml", 'r') as stream:
        try:
            common_configs = yaml.safe_load(stream)['common']
            default_configs = {k: v for k, v in common_configs.items() if k not in fixed}
            fixed.update(default_configs)
        except yaml.YAMLError as exc:
            print(exc)

    args = ConfigDict(fixed)
    hparams = naming(args)

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.isdir(os.path.join(args.log_path, hparams)):
        os.mkdir(os.path.join(args.log_path, hparams))
    folder_name = os.path.join(args.log_path, hparams)

    with open(os.path.join(folder_name, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    logger = get_logger(folder_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loaders, val_loaders, test_loaders = get_data(args, device)

    task_type = TASK_TYPE_DICT[args.dataset.lower()]
    criterion = CRITERION_DICT[args.dataset.lower()]

    model, emb_model, inner_model = get_model(args, device)

    trainer = Trainer(dataset=args.dataset.lower(),
                      task_type=task_type,
                      max_patience=args.patience,
                      criterion=criterion,
                      device=device,
                      imle_configs=args.imle_configs,
                      sample_configs=args.sample_configs)

    best_val_losses = [[] for _ in range(args.num_runs)]
    test_losses = [[] for _ in range(args.num_runs)]
    best_val_metrics = [[] for _ in range(args.num_runs)]
    test_metrics = [[] for _ in range(args.num_runs)]
    time_per_epoch = []

    for _run in range(args.num_runs):
        for _fold, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
            if emb_model is not None:
                emb_model.reset_parameters()
                optimizer_embd = torch.optim.Adam(emb_model.parameters(),
                                                  lr=args.imle_configs.embd_lr,
                                                  weight_decay=args.imle_configs.reg_embd)
                scheduler_embd = None
            else:
                optimizer_embd = None
                scheduler_embd = None
            model.reset_parameters()
            inner_model.reset_parameters()
            optimizer = torch.optim.Adam(list(model.parameters()) + list(inner_model.parameters()),
                                         lr=args.lr, weight_decay=args.reg)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             args.lr_steps,
                                                             gamma=args.lr_decay_rate if hasattr(args, 'lr_decay_rate')
                                                             else 0.1 ** 0.5)
            writer, run_folder = prepare_exp(folder_name, _run, _fold)

            best_epoch = 0
            epoch_timer = SyncMeanTimer()
            for epoch in range(args.max_epochs):
                train_loss, train_metric = trainer.train(train_loader,
                                                         emb_model,
                                                         model,
                                                         inner_model,
                                                         optimizer_embd,
                                                         optimizer)
                val_loss, val_metric, early_stop = trainer.inference(val_loader,
                                                                     emb_model,
                                                                     model,
                                                                     inner_model,
                                                                     scheduler_embd,
                                                                     scheduler,
                                                                     test=False)

                if epoch > args.min_epochs and early_stop:
                    logger.info('early stopping')
                    break

                logger.info(f'epoch: {epoch}, '
                            f'training loss: {train_loss}, '
                            f'val loss: {val_loss}, '
                            f'patience: {trainer.patience}, '
                            f'training metric: {train_metric}, '
                            f'val metric: {val_metric}, '
                            f'lr: {scheduler.optimizer.param_groups[0]["lr"]}')
                writer.add_scalar('loss/training loss', train_loss, epoch)
                writer.add_scalar('loss/val loss', val_loss, epoch)
                writer.add_scalar('metric/training metric', train_metric, epoch)
                writer.add_scalar('metric/val metric', val_metric, epoch)
                writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], epoch)

                if trainer.patience == 0:
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{run_folder}/model_best.pt')
                    torch.save(inner_model.state_dict(), f'{run_folder}/inner_model_best.pt')
                    if emb_model is not None:
                        torch.save(emb_model.state_dict(), f'{run_folder}/embd_model_best.pt')

            writer.flush()
            writer.close()

            model.load_state_dict(torch.load(f'{run_folder}/model_best.pt'))
            inner_model.load_state_dict(torch.load(f'{run_folder}/inner_model_best.pt'))
            logger.info(f'loaded best model at epoch {best_epoch}')
            if emb_model is not None:
                emb_model.load_state_dict(torch.load(f'{run_folder}/embd_model_best.pt'))

            start_time = epoch_timer.synctimer()
            test_loss, test_metric, _ = trainer.inference(test_loader, emb_model, model, inner_model, test=True)
            end_time = epoch_timer.synctimer()
            logger.info(f'Best val loss: {trainer.best_val_loss}')
            logger.info(f'Best val metric: {trainer.best_val_metric}')
            logger.info(f'test loss: {test_loss}')
            logger.info(f'test metric: {test_metric}')
            logger.info(f'max_memory_allocated: {torch.cuda.max_memory_allocated()}')
            logger.info(f'memory_allocated: {torch.cuda.memory_allocated()}')

            best_val_losses[_run].append(trainer.best_val_loss)
            test_losses[_run].append(test_loss)
            best_val_metrics[_run].append(trainer.best_val_metric)
            test_metrics[_run].append(test_metric)
            time_per_epoch.append(end_time - start_time)

            trainer.save_curve(run_folder)
            trainer.clear_stats()

    best_val_losses = [np_mean(_) for _ in best_val_losses]
    test_losses = [np_mean(_) for _ in test_losses]
    best_val_metrics = [np_mean(_) for _ in best_val_metrics]
    test_metrics = [np_mean(_) for _ in test_metrics]

    results = {'best_val_losses': best_val_losses,
               'test_losses': test_losses,
               'best_val_metrics': best_val_metrics,
               'test_metrics': test_metrics,
               'val_loss_stats': f'mean: {np_mean(best_val_losses)}, std: {np_std(best_val_losses)}',
               'test_loss_stats': f'mean: {np_mean(test_losses)}, std: {np_std(test_losses)}',
               'val_metrics_stats': f'mean: {np_mean(best_val_metrics)}, std: {np_std(best_val_metrics)}',
               'test_metrics_stats': f'mean: {np_mean(test_metrics)}, std: {np_std(test_metrics)}',
               'time_stats': f'mean: {np_mean(time_per_epoch)}, std: {np_std(time_per_epoch)}'}

    with open(os.path.join(folder_name, 'results.txt'), 'wt') as f:
        f.write(str(results))
