import os
import pickle
from functools import partial
from collections import defaultdict
from typing import Union, Optional, Any
from ml_collections import ConfigDict

import torch.linalg
from torch_geometric.data import Batch, Data

from data.data_utils import scale_grad, AttributedDataLoader, IsBetter
from data.metrics import eval_rocauc, eval_acc, eval_rmse
from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from subgraph.construct import (construct_imle_local_structure_subgraphs,
                                construct_random_local_structure_subgraphs,
                                construct_imle_subgraphs)
from training.imle_scheme import IMLEScheme
from training.soft_mask_scheme import softmax_all, softmax_topk


Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau,
                  torch.optim.lr_scheduler.MultiStepLR]
Emb_model = Any
Train_model = Any
Loss = torch.nn.modules.loss

POLICY_GRAPH_LEVEL = ['KMaxNeighbors', 'greedy_neighbors', 'graph_topk']
POLICY_NODE_LEVEL = ['topk']
POLICY_SOFT_MASK = ['soft_all', 'soft_topk']


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,
                 sample_configs: ConfigDict):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.task_type = task_type
        self.metric_comparator = IsBetter(self.task_type)
        self.criterion = criterion
        self.device = device

        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
        self.max_patience = max_patience

        self.curves = defaultdict(list)

        self.subgraph2node_aggr = sample_configs.subgraph2node_aggr
        self.sample_policy = sample_configs.sample_policy
        self.imle_scheduler = None
        if imle_configs is not None:
            # upstream may have micro batching
            self.micro_batch_embd = imle_configs.micro_batch_embd

            if self.sample_policy not in POLICY_SOFT_MASK:
                self.imle_scheduler = IMLEScheme(sample_configs.sample_policy,
                                                 None,
                                                 None,
                                                 None,
                                                 sample_configs.sample_k,
                                                 ensemble=sample_configs.ensemble
                                                 if hasattr(sample_configs, 'ensemble') else 1)

                @imle(target_distribution=TargetDistribution(alpha=1.0, beta=imle_configs.beta),
                      noise_distribution=GumbelDistribution(0., imle_configs.noise_scale, self.device),
                      input_noise_temperature=1.,
                      target_noise_temperature=1.,)
                def imle_sample_scheme(logits: torch.Tensor):
                    return self.imle_scheduler.torch_sample_scheme(logits)
                self.imle_sample_scheme = imle_sample_scheme

        if sample_configs.sample_policy is None:
            # normal training
            self.construct_duplicate_data = lambda x, *args: x
        elif imle_configs is not None:
            # N x N graph level pred, structure per node
            if self.sample_policy in POLICY_GRAPH_LEVEL:
                self.construct_duplicate_data = self.emb_model_graph_level_pred
            # already in subgraphs, O(N)
            elif self.sample_policy in POLICY_NODE_LEVEL:
                self.construct_duplicate_data = self.emb_model_node_level
            # no need IMLE, just backprop into soft masks
            elif self.sample_policy == 'soft_all':
                self.construct_duplicate_data = self.emb_model_node_soft_all
            elif self.sample_policy == 'soft_topk':
                self.construct_duplicate_data = partial(self.emb_model_node_soft_topk, k=sample_configs.sample_k)
            else:
                raise ValueError(f'{self.sample_policy} not supported')
        else:
            # N x N graph level pred, structure per node
            if self.sample_policy in POLICY_GRAPH_LEVEL:
                self.construct_duplicate_data = self.data_duplication
            # already in subgraphs, O(N)
            elif self.sample_policy in POLICY_NODE_LEVEL:
                self.construct_duplicate_data = lambda x, *args: x
            raise ValueError(f'{self.sample_policy} not supported')

    def data_duplication(self, data: Union[Data, Batch], *args):
        """
        For random sampling, graphs already have node_mask, but not other attributes

        :param data:
        :return:
        """
        graphs = Batch.to_data_list(data)
        new_batch = construct_random_local_structure_subgraphs(graphs,
                                                               data.node_mask,
                                                               data.nnodes,
                                                               data.batch,
                                                               data.y,
                                                               self.subgraph2node_aggr)
        return new_batch

    def emb_model_graph_level_pred(self,
                                   data: Union[Data, Batch],
                                   emb_model: Emb_model,
                                   device: Union[torch.device, str]):
        train = emb_model.training
        graphs = Batch.to_data_list(data)

        logits = emb_model(data.to(device))
        data = data.to('cpu')

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = tuple((data.nnodes ** 2).tolist())  # per node has a subg

        if train:
            node_mask, _ = self.imle_sample_scheme(logits)
        else:
            node_mask, _ = self.imle_scheduler.torch_sample_scheme(logits)

        new_batch = construct_imle_local_structure_subgraphs(graphs,
                                                             node_mask.cpu(),
                                                             data.nnodes,
                                                             data.batch,
                                                             data.y,
                                                             self.subgraph2node_aggr,
                                                             grad=train)

        return new_batch

    def emb_model_node_level(self,
                             data: Union[Data, Batch],
                             emb_model: Emb_model,
                             device: Union[torch.device, str]):
        train = emb_model.training
        graphs = Batch.to_data_list(data)

        logits = emb_model(data.to(device))
        data = data.cpu()

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = tuple(data.nnodes.tolist())
        self.imle_scheduler.seed_node_mask = data.target_mask

        if train:
            node_mask, _ = self.imle_sample_scheme(logits)
        else:
            node_mask, _ = self.imle_scheduler.torch_sample_scheme(logits)

        new_batch = construct_imle_subgraphs(graphs,
                                             node_mask,
                                             data.nnodes,
                                             data.batch,
                                             data.y,
                                             self.subgraph2node_aggr,
                                             grad=train)

        return new_batch

    def emb_model_node_soft_all(self,
                                data: Union[Data, Batch],
                                emb_model: Emb_model,
                                device: Union[torch.device, str]):
        train = emb_model.training
        graphs = Batch.to_data_list(data)

        logits = emb_model(data.to(device))

        # need to softmax
        logits = softmax_all(logits, data.nnodes, data.ptr if hasattr(data, 'ptr') else None)
        data = data.to('cpu')

        new_batch = construct_imle_local_structure_subgraphs(graphs,
                                                             logits.cpu(),
                                                             data.nnodes,
                                                             data.batch,
                                                             data.y,
                                                             self.subgraph2node_aggr,
                                                             grad=train)

        return new_batch

    def emb_model_node_soft_topk(self,
                                 data: Union[Data, Batch],
                                 emb_model: Emb_model,
                                 device: Union[torch.device, str],
                                 k: int = 1):
        train = emb_model.training
        graphs = Batch.to_data_list(data)

        logits = emb_model(data.to(device))

        # need to softmax
        logits = softmax_topk(logits, data.nnodes, k, training=train)
        data = data.to('cpu')

        new_batch = construct_imle_local_structure_subgraphs(graphs,
                                                             logits.cpu(),
                                                             data.nnodes,
                                                             data.batch,
                                                             data.y,
                                                             self.subgraph2node_aggr,
                                                             grad=train)

        return new_batch

    # def get_aux_loss(self, logits: torch.Tensor, split_idx: Tuple):
    #     """
    #     A KL divergence version
    #     """
    #     targets = torch.ones(logits.shape[0], device=logits.device, dtype=torch.float32)
    #     logits = torch.split(logits, split_idx, dim=0)
    #     targets = torch.split(targets, split_idx, dim=0)
    #     kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    #     loss = 0.
    #     for logit, target in zip(logits, targets):
    #         log_softmax_logits = torch.nn.LogSoftmax(dim=0)(logit.sum(1))
    #         target = target / logit.shape[0]
    #         loss += kl_loss(log_softmax_logits, target)
    #     return loss * self.aux_loss_weight

    def train(self,
              dataloader: AttributedDataLoader,
              emb_model: Emb_model,
              model: Train_model,
              optimizer_embd: Optional[Optimizer],
              optimizer: Optimizer):

        if emb_model is not None:
            emb_model.train()
            optimizer_embd.zero_grad()

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        if self.task_type != 'regression':
            preds = []
            labels = []
        else:
            preds, labels = None, None
        num_graphs = 0

        for batch_id, data in enumerate(dataloader.loader):
            optimizer.zero_grad()
            new_data = self.construct_duplicate_data(data, emb_model, self.device)

            data = data.to(self.device)
            new_data = new_data.to(self.device)

            pred = model(new_data, data)

            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled].to(torch.float))
            train_losses += loss.detach() * data.num_graphs

            loss.backward()
            optimizer.step()
            if optimizer_embd is not None:
                if (batch_id % self.micro_batch_embd == self.micro_batch_embd - 1) or (batch_id >= len(dataloader) - 1):
                    emb_model = scale_grad(emb_model, (batch_id % self.micro_batch_embd) + 1)
                    optimizer_embd.step()
                    optimizer_embd.zero_grad()

            num_graphs += data.num_graphs
            if isinstance(preds, list):
                preds.append(pred)
                labels.append(data.y)

        train_loss = train_losses.item() / num_graphs
        self.curves['train_loss'].append(train_loss)

        if isinstance(preds, list):
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.task_type == 'rocauc':
                train_metric = eval_rocauc(labels, preds)
            elif self.task_type == 'rmse':
                train_metric = eval_rmse(labels, preds)
            elif self.task_type == 'acc':
                if preds.shape[1] == 1:
                    preds = (preds > 0.).to(torch.int)
                else:
                    preds = torch.argmax(preds, dim=1)
                train_metric = eval_acc(labels, preds)
            else:
                raise NotImplementedError
        else:
            train_metric = train_loss
        self.curves['train_metric'].append(train_metric)

        if self.imle_scheduler is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model: Emb_model,
                  model: Train_model,
                  scheduler_embd: Optional[Scheduler] = None,
                  scheduler: Optional[Scheduler] = None,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        preds = []
        labels = []

        for data in dataloader.loader:
            new_data = self.construct_duplicate_data(data, emb_model, self.device)
            data = data.to(self.device)
            new_data = new_data.to(self.device)

            pred = model(new_data, data)

            if dataloader.std is not None:
                preds.append(pred * dataloader.std)
                labels.append(data.y.to(torch.float) * dataloader.std)
            else:
                preds.append(pred)
                labels.append(data.y.to(torch.float))

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        is_labeled = labels == labels
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()
        if self.task_type == 'rocauc':
            val_metric = eval_rocauc(labels, preds)
        elif self.task_type == 'regression':  # not specified regression loss type
            val_metric = val_loss
        elif self.task_type == 'rmse':
            val_metric = eval_rmse(labels, preds)
        elif self.task_type == 'acc':
            if preds.shape[1] == 1:
                preds = (preds > 0.).to(torch.int)
            else:
                preds = torch.argmax(preds, dim=1)
            val_metric = eval_acc(labels, preds)
        else:
            raise NotImplementedError

        early_stop = False
        if not test:
            self.curves['val_metric'].append(val_metric)
            self.curves['val_loss'].append(val_loss)

            self.best_val_loss = min(self.best_val_loss, val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise NotImplementedError("Need to specify max or min plateau")
            else:
                scheduler.step()
            if scheduler_embd is not None:
                if isinstance(scheduler_embd, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    raise NotImplementedError("Need to specify max or min plateau")
                else:
                    scheduler_embd.step()

            if self.metric_comparator(val_metric, self.best_val_metric):
                self.best_val_metric = val_metric
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

        if self.imle_scheduler is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return val_loss, val_metric, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))

    def clear_stats(self):
        self.curves = defaultdict(list)
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
