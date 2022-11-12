import os
import pickle
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
from subgraph.construct import construct_imle_local_structure_subgraphs, construct_random_local_structure_subgraphs
from training.imle_scheme import IMLEScheme

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau,
                  torch.optim.lr_scheduler.MultiStepLR]
Emb_model = Any
Train_model = Any
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,

                 sample_policy: str = None,
                 subgraph2node_aggr: str = 'add',
                 sample_k: int = 3,
                 **kwargs):
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

        self.subgraph2node_aggr = subgraph2node_aggr
        if imle_configs is not None:  # need to cache some configs, otherwise everything's in the dataloader already
            self.micro_batch_embd = imle_configs.micro_batch_embd
            self.temp = 1.
            self.target_distribution = TargetDistribution(alpha=1.0, beta=imle_configs.beta)
            self.noise_distribution = GumbelDistribution(0., imle_configs.noise_scale, self.device)
            self.imle_scheduler = IMLEScheme(sample_policy,
                                             None,
                                             None,
                                             sample_k,
                                             ensemble=imle_configs.ensemble if hasattr(imle_configs, 'ensemble') else 1)

        # from original data batch to duplicated data batches
        # [g1, g2, ..., gn] -> [g1_1, g1_1, g1_3, ...]
        if sample_policy is None:
            self.construct_duplicate_data = lambda x, _: x
        elif imle_configs is not None:
            self.construct_duplicate_data = self.emb_model_forward
        else:
            self.construct_duplicate_data = self.data_duplication

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

    def emb_model_forward(self, data: Union[Data, Batch], emb_model: Emb_model):
        """
        Common forward propagation for train and val, only called when embedding model is trained.

        :param data:
        :param emb_model:
        :return:
        """
        train = emb_model.training
        logits = emb_model(data)

        split_idx = tuple((data.nnodes ** 2).cpu().tolist())
        graphs = Batch.to_data_list(data)

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = split_idx

        if train:
            @imle(target_distribution=self.target_distribution,
                  noise_distribution=self.noise_distribution,
                  input_noise_temperature=self.temp,
                  target_noise_temperature=self.temp,
                  nb_samples=1)
            def imle_sample_scheme(logits: torch.Tensor):
                return self.imle_scheduler.torch_sample_scheme(logits)

            node_mask, _ = imle_sample_scheme(logits)
        else:
            node_mask, _ = self.imle_scheduler.torch_sample_scheme(logits)

        new_batch = construct_imle_local_structure_subgraphs(graphs,
                                                             node_mask,
                                                             data.nnodes,
                                                             data.batch,
                                                             data.y,
                                                             self.subgraph2node_aggr,
                                                             grad=train)

        return new_batch

    def train(self,
              dataloader: AttributedDataLoader,
              emb_model: Emb_model,
              model: Train_model,
              inner_model: Train_model,
              optimizer_embd: Optional[Optimizer],
              optimizer: Optimizer):

        if emb_model is not None:
            emb_model.train()
            optimizer_embd.zero_grad()

        model.train()
        inner_model.train()
        train_losses = torch.tensor(0., device=self.device)
        if self.task_type != 'regression':
            preds = []
            labels = []
        else:
            preds, labels = None, None
        num_graphs = 0

        for batch_id, data in enumerate(dataloader.loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            new_data = self.construct_duplicate_data(data, emb_model)

            intermediate_node_emb = inner_model(new_data)
            pred = model(data, intermediate_node_emb)

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

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model: Emb_model,
                  model: Train_model,
                  inner_model: Train_model,
                  scheduler_embd: Optional[Scheduler] = None,
                  scheduler: Optional[Scheduler] = None,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        inner_model.eval()
        preds = []
        labels = []

        for data in dataloader.loader:
            data = data.to(self.device)
            new_data = self.construct_duplicate_data(data, emb_model)

            intermediate_node_emb = inner_model(new_data)
            pred = model(data, intermediate_node_emb)

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

        if emb_model is not None:
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
