import os

import numpy as np
import torch

from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.get_data import get_data
from data.get_optimizer import make_get_embed_opt, make_get_opt
from data.utils.datatype_utils import SyncMeanTimer
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

    if hasattr(args.sample_configs, 'sample_k'):
        name = f'add{args.sample_configs.sample_k}' + name
        
    if hasattr(args.sample_configs, 'sample_k2'):
        name = f'del{args.sample_configs.sample_k2}' + name

    if hasattr(args.sample_configs, 'candid_pool'):
        name = f'candid{args.sample_configs.candid_pool}_ens{args.sample_configs.ensemble}' + name
    else:
        name = f'ens{args.sample_configs.ensemble}' + name

    if hasattr(args.sample_configs, 'sample_k'):
        name = f'add{args.sample_configs.sample_k}_' + name

    if hasattr(args.sample_configs, 'sample_k2'):
        name = f'del{args.sample_configs.sample_k2}_' + name

    if hasattr(args, 'wandb_prefix'):
        name = f'{args.wandb_prefix}_' + name


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
    train_loaders, val_loaders, test_loaders, task_id = get_data(args, device)

    trainer = Trainer(dataset=args.dataset.lower(),
                      task_type=TASK_TYPE_DICT[args.dataset.lower()],
                      max_patience=args.early_stop.patience,
                      patience_target=args.early_stop.target,
                      criterion=CRITERION_DICT[args.dataset.lower()],
                      device=device,
                      num_layers=args.num_convlayers,
                      imle_configs=args.imle_configs,
                      sample_configs=args.sample_configs,
                      plot_graph_args=args.plot_graphs if hasattr(args, 'plot_graphs') else None,
                      plot_scores_args=args.plot_scores if hasattr(args, 'plot_scores') else None,
                      plot_heatmap_args=args.plot_heatmaps if hasattr(args, 'plot_heatmaps') else None,
                      auxloss=args.imle_configs.auxloss if hasattr(args.imle_configs, 'auxloss') else None,
                      wandb=wandb,
                      use_wandb=args.use_wandb,
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
            model, emb_model, surrogate_model = get_model(args, device)
            optimizer_embd, scheduler_embd = get_embed_opt(emb_model)
            optimizer, scheduler = get_opt(model, surrogate_model)

            # count the number of parameters for model and emb_model
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if emb_model is not None:
                num_params_emb = sum(p.numel() for p in emb_model.parameters() if p.requires_grad)
                total_params = num_params + num_params_emb
            else:
                num_params_emb = 0
                total_params = num_params
            #log to wandb
            wandb.run.summary['n_params_downstream'] = num_params
            wandb.run.summary['total_params'] = total_params
            wandb.run.summary['n_params_upstream'] = num_params_emb

            logger.info(f'Number of params: {total_params} (upstream: {num_params_emb}, downstream: {num_params})')

            # wandb.watch(model, log="all", log_freq=10)
            # if emb_model is not None:
            #     wandb.watch(emb_model, log="all", log_freq=10)

            run_folder = prepare_exp(folder_name, _run, _fold)

            best_epoch = 0
            epoch_timer = SyncMeanTimer()
            for epoch in range(args.max_epochs):

                if hasattr(args, 'reset_downstream') and args.reset_downstream == epoch:
                    logger.info('Resetting downstream model...')
                    model, _, _ = get_model(args, device)
                    optimizer, scheduler = get_opt(model, surrogate_model)
                    trainer.clear_stats()

                if hasattr(args, 'reset_upstream') and args.reset_upstream == epoch:
                    logger.info('Resetting upstream model...')
                    _, emb_model, _ = get_model(args, device)
                    optimizer_embd, scheduler_embd = get_embed_opt(emb_model)
                    trainer.clear_stats()

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
        best_metrics = np.array(best_train_metrics)
    elif args.early_stop.target in ['val_metric', 'val_metric_ensemble']:
        best_metrics = np.array(best_val_metrics)
    else:
        raise NotImplementedError
    test_metrics = np.array(test_metrics)
    test_metrics_ensemble = np.array(test_metrics_ensemble)

    if args.dataset.lower() != 'qm9':
        results = {'best_metrics_type': args.early_stop.target,
                   'best_metrics_stats': f'mean: {np.mean(best_metrics)}, std: {np.std(best_metrics)}',
                   'test_metrics_stats': f'mean: {np.mean(test_metrics)}, std: {np.std(test_metrics)}',
                   'test_metrics_ensemble_stats': f'mean: {np.mean(test_metrics_ensemble)}, std: {np.std(test_metrics_ensemble)}',
                   'time_stats': f'mean: {np.mean(time_per_epoch)}, std: {np.std(time_per_epoch)}'}

        wandb.run.summary['final_metric'] = np.mean(best_metrics)
        wandb.run.summary['final_metric_std'] = np.std(best_metrics)

        wandb.run.summary['test_metric'] = np.mean(test_metrics)
        wandb.run.summary['test_metric_std'] = np.std(test_metrics)

        wandb.run.summary['test_metric_ensemble'] = np.mean(test_metrics_ensemble)
        wandb.run.summary['test_metric_ensemble_std'] = np.std(test_metrics_ensemble)

        wandb.run.summary['time_per_epoch'] = np.mean(time_per_epoch)
        wandb.run.summary['time_per_epoch_std'] = np.std(time_per_epoch)
    else:
        results = {'best_metrics_type': args.early_stop.target,
                   'time_stats': f'mean: {np.mean(time_per_epoch)}, std: {np.std(time_per_epoch)}'}
        best_metrics_mean = np.mean(best_metrics, axis=0)
        best_metrics_std = np.std(best_metrics, axis=0)
        test_metrics_mean = np.mean(test_metrics, axis=0)
        test_metrics_std = np.std(test_metrics, axis=0)
        test_metrics_ensemble_mean = np.mean(test_metrics_ensemble, axis=0)
        test_metrics_ensemble_std = np.std(test_metrics_ensemble, axis=0)
        # https://github.com/radoslav11/SP-MPNN/blob/main/src/experiments/run_gr.py#L6C1-L20C2
        tasks = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]
        for i, id in enumerate(task_id):
            t = tasks[id]
            results[f'{t}_best_metrics_stats'] = f'mean: {best_metrics_mean[i]}, std: {best_metrics_std[i]}'
            results[f'{t}_test_metrics_stats'] = f'mean: {test_metrics_mean[i]}, std: {test_metrics_std[i]}'
            results[f'{t}_test_metrics_ensemble_stats'] = f'mean: {test_metrics_ensemble_mean[i]}, std: {test_metrics_ensemble_std[i]}'

            wandb.run.summary[f'{t}_final_metric'] = best_metrics_mean[i]
            wandb.run.summary[f'{t}_final_metric_std'] = best_metrics_std[i]

            wandb.run.summary[f'{t}_test_metric'] = test_metrics_mean[i]
            wandb.run.summary[f'{t}_test_metric_std'] = test_metrics_std[i]

            wandb.run.summary[f'{t}_test_metric_ensemble'] = test_metrics_ensemble_mean[i]
            wandb.run.summary[f'{t}_test_metric_ensemble_std'] = test_metrics_ensemble_std[i]

        wandb.run.summary['time_per_epoch'] = np.mean(time_per_epoch)
        wandb.run.summary['time_per_epoch_std'] = np.std(time_per_epoch)

    with open(os.path.join(folder_name, 'result.yaml'), 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)
