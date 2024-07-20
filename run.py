import os
import logging
import yaml

import numpy as np
import torch
import wandb
from ml_collections import ConfigDict
from sacred import Experiment

from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.get_data import get_data
from data.get_optimizer import make_get_embed_opt, make_get_opt
from data.utils.datatype_utils import SyncMeanTimer
from models.get_model import get_model
from training.trainer import Trainer

torch.multiprocessing.set_sharing_strategy('file_system')

ex = Experiment()


@ex.automain
def main(fixed):
    args = ConfigDict(dict(fixed))
    run_name = args.wandb_name
    logging.basicConfig(level=logging.DEBUG)

    wandb.init(project=args.wandb_project, mode="online" if args.use_wandb else "disabled",
               config=args.to_dict(),
               name=run_name,
               entity="mls-stuttgart")

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.isdir(os.path.join(args.log_path, run_name)):
        os.mkdir(os.path.join(args.log_path, run_name))
    folder_name = os.path.join(args.log_path, run_name)

    with open(os.path.join(folder_name, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loaders, val_loaders, test_loaders = get_data(args, device)

    trainer = Trainer(dataset=args.dataset.lower(),
                      task_type=TASK_TYPE_DICT[args.dataset.lower()],
                      max_patience=args.early_stop.patience,
                      patience_target=args.early_stop.target,
                      criterion=CRITERION_DICT[args.dataset.lower()],
                      device=device,
                      imle_configs=args.imle_configs,
                      sample_configs=args.sample_configs,
                      auxloss=args.imle_configs.auxloss if hasattr(args.imle_configs, 'auxloss') else None)

    best_train_metrics = [[] for _ in range(args.num_runs)]
    best_val_metrics = [[] for _ in range(args.num_runs)]
    test_metrics = [[] for _ in range(args.num_runs)]
    time_per_epoch = []

    get_embed_opt = make_get_embed_opt(args)
    get_opt = make_get_opt(args)

    for _run in range(args.num_runs):
        for _fold, (train_loader, val_loader, test_loader) in enumerate(
                zip(train_loaders, val_loaders, test_loaders)):
            model, emb_model = get_model(args, device)
            optimizer_embd, scheduler_embd = get_embed_opt(emb_model)
            optimizer, scheduler = get_opt(model)

            best_epoch = 0
            epoch_timer = SyncMeanTimer()
            for epoch in range(args.max_epochs):
                trainer.epoch = epoch
                train_loss, train_metric = trainer.train(train_loader,
                                                         emb_model,
                                                         model,
                                                         optimizer_embd,
                                                         optimizer)

                val_loss, val_metric, early_stop = trainer.inference(
                    val_loader,
                    emb_model,
                    model,
                    scheduler_embd,
                    scheduler,
                    test=False)

                if args.use_wandb:
                    log_dict = {"train_loss": train_loss,
                                "train_metric": train_metric,
                                "val_loss": val_loss,
                                "val_metric": val_metric,
                                "down_lr": scheduler.get_last_lr()[-1] if scheduler is not None else 0.,
                                "up_lr": scheduler_embd.get_last_lr()[-1] if scheduler_embd is not None else 0.}
                    wandb.log(log_dict)

                if epoch > args.min_epochs and early_stop:
                    logging.info('early stopping')
                    break

                logging.info(f'epoch: {epoch}, '
                            f'training loss: {round(train_loss, 5)}, '
                            f'val loss: {round(val_loss, 5)}, '
                            f'patience: {trainer.patience}, '
                            f'training metric: {round(train_metric, 5)}, '
                            f'val metric: {round(val_metric, 5)}')

                if trainer.patience == 0:
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{folder_name}/model_best_{_run}_{_fold}.pt')
                    if emb_model is not None:
                        torch.save(emb_model.state_dict(),
                                   f'{folder_name}/embd_model_best_{_run}_{_fold}.pt')

            # test inference
            model.load_state_dict(torch.load(f'{folder_name}/model_best_{_run}_{_fold}.pt'))
            logging.info(f'loaded best model at epoch {best_epoch}')
            if emb_model is not None:
                emb_model.load_state_dict(torch.load(f'{folder_name}/embd_model_best_{_run}_{_fold}.pt'))

            start_time = epoch_timer.synctimer()
            test_loss, test_metric, _ = trainer.inference(test_loader, emb_model, model, test=True)
            end_time = epoch_timer.synctimer()
            logging.info(f'Best val loss: {trainer.best_val_loss}')
            logging.info(f'Best val metric: {trainer.best_val_metric}')
            logging.info(f'test loss: {test_loss}')
            logging.info(f'test metric: {test_metric}')

            best_train_metrics[_run].append(trainer.best_train_metric)
            best_val_metrics[_run].append(trainer.best_val_metric)
            test_metrics[_run].append(test_metric)
            time_per_epoch.append(end_time - start_time)

            trainer.clear_stats()

    test_metrics = np.array(test_metrics)

    results = {'test_metrics_stats': f'mean: {np.mean(test_metrics)}, std: {np.std(test_metrics)}',
               'time_stats': f'mean: {np.mean(time_per_epoch)}, std: {np.std(time_per_epoch)}'}

    wandb.run.summary['test_metric'] = np.mean(test_metrics)
    wandb.run.summary['test_metric_std'] = np.std(test_metrics)

    wandb.run.summary['time_per_epoch'] = np.mean(time_per_epoch)
    wandb.run.summary['time_per_epoch_std'] = np.std(time_per_epoch)

    with open(os.path.join(folder_name, 'result.yaml'), 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)
