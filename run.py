import os

import torch
from numpy import mean as np_mean
from numpy import std as np_std

from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.data_utils import SyncMeanTimer
from data.get_data import get_data
from data.get_optimizer import make_get_embed_opt, make_get_opt
from models.get_model import get_model
from training.trainer import Trainer

torch.multiprocessing.set_sharing_strategy('file_system')
from datetime import datetime
import logging
import yaml


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
    name += f'layer_{args.num_convlayers}_'

    if hasattr(args, 'wandb_prefix'):
        name = f'{args.wandb_prefix}_' + name

    if args.sample_configs.sample_policy is None:
        name += 'normal'
        return name

    if hasattr(args.sample_configs, 'weight_edges'):
        name += f'weight_{args.sample_configs.weight_edges}_'

    if args.imle_configs is not None:
        name += f'sampler_{args.imle_configs.sampler}_'
        name += f'model_{args.imle_configs.model}_'
    else:
        name += 'OnTheFly_'

    name += f'ensemble_{args.sample_configs.ensemble}_'
    name += f'policy_{args.sample_configs.sample_policy}_'
    return name


def prepare_exp(folder_name: str, num_run: int, num_fold: int) -> str:
    run_folder = os.path.join(folder_name,
                              f'run{num_run}_fold{num_fold}_{"".join(str(datetime.now()).split(":"))}')
    os.mkdir(run_folder)
    return run_folder


def run(wandb, args):
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

    trainer = Trainer(dataset=args.dataset.lower(),
                      task_type=task_type,
                      max_patience=args.early_stop.patience,
                      patience_target=args.early_stop.target,
                      criterion=criterion,
                      device=device,
                      imle_configs=args.imle_configs,
                      sample_configs=args.sample_configs,
                      auxloss=args.imle_configs.auxloss if hasattr(args.imle_configs, 'auxloss') else None,
                      wandb=wandb,
                      use_wandb=args.use_wandb,
                      plot_args=args.plot_graphs if hasattr(args, 'plot_graphs') else None,
                      connectedness_metric_args=args.connectedness if hasattr(args, 'connectedness') else None)

    best_train_metrics = [[] for _ in range(args.num_runs)]
    best_val_metrics = [[] for _ in range(args.num_runs)]
    test_metrics = [[] for _ in range(args.num_runs)]
    test_metrics_ensemble = [[] for _ in range(args.num_runs)]
    time_per_epoch = []

    get_embed_opt = make_get_embed_opt(args)
    get_opt = make_get_opt(args)

    for _run in range(args.num_runs):
        for _fold, (train_loader, val_loader, test_loader) in enumerate(
                zip(train_loaders, val_loaders, test_loaders)):
            model, emb_model = get_model(args, device)
            optimizer_embd, scheduler_embd = get_embed_opt(emb_model)
            optimizer, scheduler = get_opt(model)

            # wandb.watch(model, log="all", log_freq=10)
            # if emb_model is not None:
            #     wandb.watch(emb_model, log="all", log_freq=10)

            run_folder = prepare_exp(folder_name, _run, _fold)

            best_epoch = 0
            epoch_timer = SyncMeanTimer()
            for epoch in range(args.max_epochs):
                trainer.epoch = epoch
                train_loss, train_metric = trainer.train(train_loader,
                                                         emb_model,
                                                         model,
                                                         optimizer_embd,
                                                         optimizer)
                
                val_loss, val_metric, val_loss_ensemble, val_metric_ensemble, early_stop = trainer.inference(
                    val_loader,
                    emb_model,
                    model,
                    scheduler_embd,
                    scheduler,
                    train_loss,
                    train_metric,
                    test=False)

                if epoch > args.min_epochs and early_stop:
                    logger.info('early stopping')
                    break

                logger.info(f'epoch: {epoch}, '
                            f'training loss: {round(train_loss, 5)}, '
                            f'val loss: {round(val_loss, 5)}, '
                            f'patience: {trainer.patience}, '
                            f'training metric: {round(train_metric, 5)}, '
                            f'val metric: {round(val_metric, 5)}, '
                            f'val loss ensemble: {round(val_loss_ensemble, 5)}, '
                            f'val metric ensemble: {round(val_metric_ensemble, 5)}')

                if epoch % 50 == 0:
                    torch.save(model.state_dict(), f'{run_folder}/model_{epoch}.pt')
                    if emb_model is not None:
                        torch.save(emb_model.state_dict(),
                                   f'{run_folder}/embd_model_{epoch}.pt')

                if trainer.patience == 0:
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{run_folder}/model_best.pt')
                    if emb_model is not None:
                        torch.save(emb_model.state_dict(),
                                   f'{run_folder}/embd_model_best.pt')

            # rm cached model
            used_model = os.listdir(run_folder)
            for modelname in used_model:
                if modelname.endswith('.pt') and not modelname.endswith('best.pt'):
                    os.remove(os.path.join(run_folder, modelname))
            # save last model
            torch.save(model.state_dict(), f'{run_folder}/model_final.pt')
            if emb_model is not None:
                torch.save(emb_model.state_dict(), f'{run_folder}/embd_model_final.pt')
            # test inference
            model.load_state_dict(torch.load(f'{run_folder}/model_best.pt'))
            logger.info(f'loaded best model at epoch {best_epoch}')
            if emb_model is not None:
                emb_model.load_state_dict(torch.load(f'{run_folder}/embd_model_best.pt'))

            start_time = epoch_timer.synctimer()
            test_loss, test_metric, test_loss_ensemble, test_metric_ensemble, _ = trainer.inference(test_loader, emb_model, model, test=True)
            end_time = epoch_timer.synctimer()
            logger.info(f'Best val loss: {trainer.best_val_loss}')
            logger.info(f'Best val metric: {trainer.best_val_metric}')
            logger.info(f'test loss: {test_loss}')
            logger.info(f'test metric: {test_metric}')
            logger.info(f'test loss ensemble: {test_loss_ensemble}')
            logger.info(f'test metric ensemble: {test_metric_ensemble}')

            best_train_metrics[_run].append(trainer.best_train_metric)
            best_val_metrics[_run].append(trainer.best_val_metric)
            test_metrics[_run].append(test_metric)
            test_metrics_ensemble[_run].append(test_metric_ensemble)
            time_per_epoch.append(end_time - start_time)

            trainer.clear_stats()

    if args.early_stop.target == 'train_metric':
        best_metrics = [np_mean(_) for _ in best_train_metrics]
    elif args.early_stop.target == 'val_metric':
        best_metrics = [np_mean(_) for _ in best_val_metrics]
    else:
        raise NotImplementedError
    test_metrics = [np_mean(_) for _ in test_metrics]
    test_metrics_ensemble = [np_mean(_) for _ in test_metrics_ensemble]

    results = {'best_metrics_type': args.early_stop.target,
               'best_metrics': str(best_metrics),
               'test_metrics': str(test_metrics),
               'test_metrics_ensemble': str(test_metrics_ensemble),
               'best_metrics_stats': f'mean: {np_mean(best_metrics)}, std: {np_std(best_metrics)}',
               'test_metrics_stats': f'mean: {np_mean(test_metrics)}, std: {np_std(test_metrics)}',
               'test_metrics_ensemble_stats': f'mean: {np_mean(test_metrics_ensemble)}, std: {np_std(test_metrics_ensemble)}',
               'time_stats': f'mean: {np_mean(time_per_epoch)}, std: {np_std(time_per_epoch)}'}

    with open(os.path.join(folder_name, 'result.yaml'), 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    wandb.run.summary['final_metric'] = np_mean(best_metrics)
    wandb.run.summary['final_metric_std'] = np_std(best_metrics)

    wandb.run.summary['final_test_metric'] = np_mean(test_metrics)
    wandb.run.summary['final_test_metric_std'] = np_std(test_metrics)

    wandb.run.summary['final_test_metric_ensemble'] = np_mean(test_metrics_ensemble)
    wandb.run.summary['final_test_metric_ensemble_std'] = np_std(test_metrics_ensemble)

    wandb.run.summary['time_per_epoch'] = np_mean(time_per_epoch)
    wandb.run.summary['time_per_epoch_std'] = np_std(time_per_epoch)
