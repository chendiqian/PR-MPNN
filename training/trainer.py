from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Data

from data.get_optimizer import MyPlateau
from data.get_sampler import get_sampler
from data.metrics.connectness_metrics import get_connectedness_metric
from data.metrics.metrics import get_eval
from data.utils.datatype_utils import (AttributedDataLoader,
                                       IsBetter,
                                       DuoDataStructure,
                                       BatchOriginalDataStructure)
from data.utils.plot_utils import plot_score, plot_rewired_graphs
from data.utils.args_utils import process_idx
from training.construct import (construct_from_edge_candidate,
                                construct_from_attention_mat)

LARGE_NUMBER = 1.e10


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 patience_target: str,
                 criterion: torch.nn.modules.loss,
                 device: Union[str, torch.device],
                 num_layers: int,
                 imle_configs: ConfigDict,
                 sample_configs: ConfigDict,
                 auxloss: ConfigDict,
                 wandb: Optional[Any] = None,
                 use_wandb: bool = False,
                 args: ConfigDict = None,
                 connectedness_metric_args: Optional[ConfigDict] = None,):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.task_type = task_type
        self.metric_comparator = IsBetter(self.task_type)
        self.criterion = criterion
        self.device = device

        self.max_patience = max_patience
        self.patience_target = patience_target

        # clear best scores
        self.clear_stats()

        self.wandb = wandb
        self.use_wandb = use_wandb

        if hasattr(sample_configs, 'merge_priors') and sample_configs.merge_priors:
            ensemble = 1
            merge_priors = True
        else:
            ensemble = sample_configs.ensemble
            merge_priors = False

        plot_args = args.plot_args if hasattr(args, 'plot_args') else None

        # plot functions
        if plot_args is None:
            self.plot = None
            self.plot_score = None
        else:
            self.plot = partial(plot_rewired_graphs,
                                wandb=wandb,
                                use_wandb=use_wandb,
                                plot_args=plot_args,
                                ensemble=ensemble,
                                num_train_ensemble=imle_configs.num_train_ensemble if imle_configs is not None else 1,
                                num_val_ensemble=imle_configs.num_val_ensemble if imle_configs is not None else 1,
                                include_original_graph=sample_configs.include_original_graph)
            self.plot_score = partial(plot_score,
                                      plot_args=plot_args,
                                      wandb=wandb,
                                      use_wandb=use_wandb)

        if connectedness_metric_args is not None:
            self.connectedness_metric = partial(get_connectedness_metric,
                                                metric=connectedness_metric_args.metric)
            self.connectedness_metric_cnt = connectedness_metric_args.every
        else:
            self.connectedness_metric = None

        train_forward, val_forward, sampler_class = get_sampler(imle_configs, sample_configs, self.device)

        if sample_configs.sample_policy is None:
            # normal training
            self.construct_duplicate_data = lambda x, *args: (x, None, 0.)
        elif imle_configs is None:
            # random sampling
            self.construct_duplicate_data = lambda x, *args: (x, None, 0.)
        elif imle_configs is not None:
            if imle_configs.model is None:
                # an exception, in this case we merge the sampling into the downstream model
                self.construct_duplicate_data = lambda x, *args: (x, None, 0.)
            else:
                if sample_configs.sample_policy == 'edge_candid':
                    construct_duplicate_data = partial(construct_from_edge_candidate,
                                                       ensemble=ensemble,
                                                       merge_priors=merge_priors,
                                                       samplek_dict={
                                                           'add_k': sample_configs.sample_k,
                                                           'del_k': sample_configs.sample_k2 if
                                                           hasattr(sample_configs,
                                                                   'sample_k2') else 0},
                                                       sampler_class=sampler_class,
                                                       train_forward=train_forward,
                                                       val_forward=val_forward,
                                                       weight_edges=imle_configs.weight_edges,
                                                       marginals_mask=imle_configs.marginals_mask,
                                                       include_original_graph=sample_configs.include_original_graph,
                                                       negative_sample=imle_configs.negative_sample,
                                                       separate=sample_configs.separate,
                                                       in_place=sample_configs.in_place,
                                                       directed_sampling=sample_configs.directed,
                                                       num_layers=num_layers,
                                                       rewire_layers=process_idx(
                                                           sample_configs.rewire_layers,
                                                           num_layers) if hasattr(
                                                           sample_configs,
                                                           'rewire_layers') else None,
                                                       auxloss_dict=auxloss,
                                                       wandb=wandb,
                                                       plot_heatmaps=args.plot_heatmaps if hasattr(args, 'plot_heatmaps') else None,)
                    def func(data, emb_model, batch_id, epoch, phase):
                        dat_batch, graphs = data.batch, data.list
                        data, scores, auxloss = construct_duplicate_data(dat_batch,
                                                                         graphs,
                                                                         emb_model.training,
                                                                         *emb_model(dat_batch),
                                                                         batch_id=batch_id,
                                                                         epoch=epoch,
                                                                         phase=phase)
                        return data, scores, auxloss
                    self.construct_duplicate_data = func
                elif sample_configs.sample_policy == 'global':
                    # learnable way with attention mask
                    policy = 'global_' + ('directed' if sample_configs.directed else 'undirected')
                    sampler_class.policy = policy
                    construct_duplicate_data = partial(construct_from_attention_mat,
                                                       ensemble=ensemble,
                                                       merge_priors=merge_priors,
                                                       sample_policy=policy,
                                                       samplek_dict={
                                                           'add_k': sample_configs.sample_k,
                                                           'del_k': sample_configs.sample_k2 if
                                                           hasattr(sample_configs, 'sample_k2') else 0},
                                                       directed_sampling=sample_configs.directed,
                                                       auxloss_dict=auxloss,
                                                       sampler_class=sampler_class,
                                                       train_forward=train_forward,
                                                       val_forward=val_forward,
                                                       weight_edges=imle_configs.weight_edges,
                                                       marginals_mask=imle_configs.marginals_mask,
                                                       device=self.device,
                                                       include_original_graph=sample_configs.include_original_graph,
                                                       negative_sample=imle_configs.negative_sample,
                                                       in_place=sample_configs.in_place,
                                                       separate=sample_configs.separate,
                                                       num_layers=num_layers,
                                                       rewire_layers=process_idx(
                                                           sample_configs.rewire_layers,
                                                           num_layers) if hasattr(
                                                           sample_configs,
                                                           'rewire_layers') else None)
                    def func(data, emb_model, batch_id, epoch, phase):
                        dat_batch, graphs = data.batch, data.list
                        data, scores, auxloss = construct_duplicate_data(dat_batch,
                                                                         graphs,
                                                                         emb_model.training,
                                                                         *emb_model(dat_batch),
                                                                         batch_id=batch_id,
                                                                         epoch=epoch,
                                                                         phase=phase)
                        return data, scores, auxloss
                    self.construct_duplicate_data = func
                else:
                    raise ValueError(f'unexpected policy {sample_configs.sample_policy}')

    def check_datatype(self, data, task_type):
        if isinstance(data, Data):
            if task_type == 'graph':
                num_preds = data.num_graphs
            elif task_type == 'node':
                num_preds = data.y.shape[0]
            else:
                raise NotImplementedError
            return data.to(self.device), num_preds
        elif type(data) == DuoDataStructure:
            # this must be before tuple type check, namedtuple is a subset of tuple
            if task_type == 'graph':
                num_preds = data.num_unique_graphs
            elif task_type == 'node':
                num_preds = data.y.shape[0] // (data.num_graphs // data.num_unique_graphs)
            else:
                raise NotImplementedError
            return DuoDataStructure(org=data.org.to(self.device) if data.org is not None else None,
                                    candidates=[g.to(self.device) for g in data.candidates],
                                    y=data.y.to(self.device),
                                    num_graphs=data.num_graphs,
                                    num_unique_graphs=data.num_unique_graphs), num_preds
        elif type(data) == BatchOriginalDataStructure:
            if task_type == 'graph':
                num_preds = data.num_graphs
            elif task_type == 'node':
                num_preds = data.y.shape[0]
            else:
                raise NotImplementedError
            return BatchOriginalDataStructure(batch=data.batch.to(self.device),
                                              list=[g.to(self.device) for g in data.list],
                                              y=data.y.to(self.device),
                                              num_graphs=data.num_graphs), num_preds
        else:
            raise TypeError(f"Unexpected dtype {type(data)}")

    def train(self,
              dataloader: AttributedDataLoader,
              emb_model,
              model,
              optimizer_embd,
              optimizer):

        if emb_model is not None:
            emb_model.train()
            optimizer_embd.zero_grad()

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        preds = []
        labels = []
        num_graphs = 0

        auxlosses = 0.
        for batch_id, data in enumerate(dataloader.loader):
            optimizer.zero_grad()
            data, _ = self.check_datatype(data, dataloader.task)
            data, scores, auxloss = self.construct_duplicate_data(data, emb_model, batch_id=batch_id, epoch=self.epoch, phase='train')

            if self.plot is not None:
                self.plot(data, True, self.epoch, batch_id)
            if self.plot_score is not None and scores is not None:
                self.plot_score(scores, self.epoch, batch_id)

            auxlosses = auxlosses + auxloss

            pred = model(data)
            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled])
            train_losses += loss.detach() * data.num_graphs

            loss = loss + auxlosses

            loss.backward()
            optimizer.step()
            if optimizer_embd is not None:
                # torch.nn.utils.clip_grad_value_(emb_model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(emb_model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                optimizer_embd.step()
                optimizer_embd.zero_grad()

            num_graphs += data.num_graphs
            preds.append(pred)
            labels.append(data.y)

        train_loss = train_losses.item() / num_graphs
        self.best_train_loss = min(self.best_train_loss, train_loss)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric = get_eval(self.task_type, labels, preds)

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model,
                  model,
                  scheduler_embd: Any = None,
                  scheduler: Any = None,
                  train_loss: float = 0.,
                  train_metric: float = 0.,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        preds = []
        labels = []

        preds_ensemble = []
        labels_ensemble = []
        connectedness_metrics = []

        preds_uncertainty = []

        for batch_id, data in enumerate(dataloader.loader):
            data, num_preds = self.check_datatype(data, dataloader.task)
            data, _, _ = self.construct_duplicate_data(data, emb_model, batch_id=batch_id, epoch=self.epoch, phase='test')

            pred = model(data)

            label = data.y
            label_ensemble = label[:num_preds]

            pred_ensemble = pred.reshape(*(-1, num_preds) + pred.shape[1:]).transpose(0, 1)

            # For classification, we should use something like the entropy of the mean prediction
            # i.e. we have (B_SZ, N_ENS, N_CLASSES)->(B_SZ, 1, N_CLASSES)->compute entropy over N_CLASSES
            pred_uncertainty = pred_ensemble.cpu().numpy().std(1)
            pred_ensemble = pred_ensemble.mean(dim=1)

            dataset_std = 1 if dataloader.std is None else dataloader.std.to(self.device)
            preds.append(pred * dataset_std)
            preds_ensemble.append(pred_ensemble * dataset_std)
            labels.append(label * dataset_std)
            labels_ensemble.append(label_ensemble * dataset_std)
            preds_uncertainty.append(pred_uncertainty)

            if not test and self.connectedness_metric is not None and self.epoch % self.connectedness_metric_cnt == 1:
                if isinstance(data, Data):
                    data = data
                elif type(data) == DuoDataStructure:
                    data = data.candidates[0]
                elif type(data) == BatchOriginalDataStructure:
                    data = data.batch
                else:
                    raise TypeError(f'{type(data)} unsupported')
                connectedness_metrics.append(self.connectedness_metric(data))

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        preds_ensemble = torch.cat(preds_ensemble, dim=0)
        labels_ensemble = torch.cat(labels_ensemble, dim=0)
        preds_uncertainty = np.concatenate(preds_uncertainty, axis=0)

        is_labeled = labels == labels
        is_labeled_ens = labels_ensemble == labels_ensemble
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()
        val_loss_ensemble = self.criterion(preds_ensemble[is_labeled_ens], labels_ensemble[is_labeled_ens]).item()

        val_metric = get_eval(self.task_type, labels, preds)
        val_metric_ensemble = get_eval(self.task_type, labels_ensemble, preds_ensemble)

        early_stop = False
        if not test:
            self.best_val_loss = min(self.best_val_loss, val_loss)

            if scheduler is not None:
                if isinstance(scheduler, MyPlateau):
                    if scheduler.lr_target == 'train_metric':
                        scheduler.step(train_metric)
                    elif scheduler.lr_target == 'train_loss':
                        scheduler.step(train_loss)
                    elif scheduler.lr_target == 'val_metric':
                        scheduler.step(val_metric)
                    elif scheduler.lr_target == 'val_loss':
                        scheduler.step(val_loss)
                else:
                    scheduler.step()
            if scheduler_embd is not None:
                if isinstance(scheduler_embd, MyPlateau):
                    raise NotImplementedError("Need to specify max or min plateau")
                else:
                    scheduler_embd.step()

            train_is_better, self.best_train_metric = self.metric_comparator(train_metric, self.best_train_metric)
            val_is_better, self.best_val_metric = self.metric_comparator(val_metric, self.best_val_metric)

            if self.patience_target == 'val_metric':
                is_better = val_is_better
            elif self.patience_target == 'train_metric':
                is_better = train_is_better
            else:
                raise NotImplementedError

            if is_better:
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

            if self.wandb is not None and self.use_wandb:
                log_dict = {"train_loss": train_loss,
                            "train_metric": train_metric,
                            "val_loss": val_loss,
                            "val_metric": val_metric,
                            "val_loss_ensemble": val_loss_ensemble,
                            "val_metric_ensemble": val_metric_ensemble,
                            "down_lr": scheduler.get_last_lr()[-1] if scheduler is not None else 0.,
                            "up_lr": scheduler_embd.get_last_lr()[-1] if scheduler_embd is not None else 0.,
                            "val_preds_uncertainty": self.wandb.Histogram(preds_uncertainty)}

                if len(connectedness_metrics) > 0:
                    connectedness_metrics_names = connectedness_metrics[0].keys()
                    for name in connectedness_metrics_names:
                        values = []
                        for metric in connectedness_metrics:
                            values.extend(metric[name])
                        log_dict[f'connectedness_{name}'] = self.wandb.Histogram(np.array(values, dtype=np.float32))

                self.wandb.log(log_dict)

        return val_loss, val_metric, val_loss_ensemble, val_metric_ensemble, early_stop

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.best_train_loss = 1e5
        self.best_train_metric = None
        self.patience = 0
        self.epoch = 0
