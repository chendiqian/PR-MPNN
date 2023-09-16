import os

import numpy as np
import torch
from ml_collections import ConfigDict
from sacred import Experiment

import wandb
from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.get_data import get_data
from models.get_model import get_model
from run import naming
from training.trainer import Trainer
import logging
torch.multiprocessing.set_sharing_strategy('file_system')

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

    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loaders, task_id = get_data(args, device)

    trainer = Trainer(dataset=args.dataset.lower(),
                      task_type=TASK_TYPE_DICT[args.dataset.lower()],
                      max_patience=0,
                      patience_target='val_metric',
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

    subfolder = os.listdir(args.model_path)
    subfolder = [f for f in subfolder if f.startswith('run')]

    test_metrics = [[] for _ in range(len(subfolder))]
    test_metrics_ensemble = [[] for _ in range(len(subfolder))]

    for _run in range(len(subfolder)):
        for _fold, test_loader in enumerate(test_loaders):
            model, emb_model, surrogate_model = get_model(args, device)

            # count the number of parameters for model and emb_model
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if emb_model is not None:
                num_params_emb = sum(p.numel() for p in emb_model.parameters() if p.requires_grad)
                total_params = num_params + num_params_emb
            else:
                num_params_emb = 0
                total_params = num_params
            # log to wandb
            wandb.run.summary['n_params_downstream'] = num_params
            wandb.run.summary['total_params'] = total_params
            wandb.run.summary['n_params_upstream'] = num_params_emb

            logging.info(f'Number of params: {total_params} (upstream: {num_params_emb}, downstream: {num_params})')

            model.load_state_dict(torch.load(f'{args.model_path}/{subfolder[_run]}/model_best.pt'))
            if emb_model is not None:
                emb_model.load_state_dict(torch.load(f'{args.model_path}/{subfolder[_run]}/embd_model_best.pt'))

            test_loss, test_metric, test_loss_ensemble, test_metric_ensemble, _ = trainer.inference(test_loader,
                                                                                                    emb_model, model,
                                                                                                    test=True)
            logging.info(f'test loss: {test_loss}')
            logging.info(f'test metric: {test_metric}')
            logging.info(f'test loss ensemble: {test_loss_ensemble}')
            logging.info(f'test metric ensemble: {test_metric_ensemble}')

            test_metrics[_run].append(test_metric)
            test_metrics_ensemble[_run].append(test_metric_ensemble)

    test_metrics = np.array(test_metrics)
    test_metrics_ensemble = np.array(test_metrics_ensemble)

    if args.dataset.lower() != 'qm9':
        results = {'test_metrics_stats': f'mean: {np.mean(test_metrics)}, std: {np.std(test_metrics)}',
                   'test_metrics_ensemble_stats': f'mean: {np.mean(test_metrics_ensemble)}, std: {np.std(test_metrics_ensemble)}'}

        wandb.run.summary['test_metric'] = np.mean(test_metrics)
        wandb.run.summary['test_metric_std'] = np.std(test_metrics)

        wandb.run.summary['test_metric_ensemble'] = np.mean(test_metrics_ensemble)
        wandb.run.summary['test_metric_ensemble_std'] = np.std(test_metrics_ensemble)
    else:
        results = {}
        test_metrics_mean = np.mean(test_metrics, axis=0)
        test_metrics_std = np.std(test_metrics, axis=0)
        test_metrics_ensemble_mean = np.mean(test_metrics_ensemble, axis=0)
        test_metrics_ensemble_std = np.std(test_metrics_ensemble, axis=0)
        # https://github.com/radoslav11/SP-MPNN/blob/main/src/experiments/run_gr.py#L6C1-L20C2
        tasks = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]
        for i, id in enumerate(task_id):
            t = tasks[id]
            results[f'{t}_test_metrics_ensemble_stats'] = f'mean: {test_metrics_ensemble_mean[i]}, std: {test_metrics_ensemble_std[i]}'

            wandb.run.summary[f'{t}_test_metric'] = test_metrics_mean[i]
            wandb.run.summary[f'{t}_test_metric_std'] = test_metrics_std[i]

            wandb.run.summary[f'{t}_test_metric_ensemble'] = test_metrics_ensemble_mean[i]
            wandb.run.summary[f'{t}_test_metric_ensemble_std'] = test_metrics_ensemble_std[i]
