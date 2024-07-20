from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch
from ml_collections import ConfigDict

from data.get_optimizer import MyPlateau
from data.metrics import get_eval
from data.utils.datatype_utils import (AttributedDataLoader,
                                       IsBetter,
                                       BatchOriginalDataStructure)
from training.construct import construct_from_edge_candidate
from simple.simple_scheme import EdgeSIMPLEBatched

LARGE_NUMBER = 1.e10


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 patience_target: str,
                 criterion: torch.nn.modules.loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,
                 sample_configs: ConfigDict,
                 auxloss: ConfigDict):
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

        simple_sampler = EdgeSIMPLEBatched(sample_configs.sample_k,
                                           device,
                                           val_ensemble=imle_configs.num_val_ensemble,
                                           train_ensemble=imle_configs.num_train_ensemble)
        train_forward = simple_sampler
        val_forward = simple_sampler.validation
        construct_duplicate_data = partial(construct_from_edge_candidate,
                                           samplek_dict={
                                               'add_k': sample_configs.sample_k,
                                               'del_k': sample_configs.sample_k2 if
                                               hasattr(sample_configs,
                                                       'sample_k2') else 0},
                                           sampler_class=simple_sampler,
                                           train_forward=train_forward,
                                           val_forward=val_forward,
                                           include_original_graph=sample_configs.include_original_graph,
                                           separate=sample_configs.separate,
                                           directed_sampling=sample_configs.directed,
                                           auxloss_dict=auxloss)

        def func(data, emb_model):
            dat_batch, graphs = data.batch, data.list
            data, auxloss = construct_duplicate_data(dat_batch,
                                                             graphs,
                                                             emb_model.training,
                                                             *emb_model(dat_batch))
            return data, auxloss

        self.construct_duplicate_data = func

    def to_device(self, data):
        if type(data) == BatchOriginalDataStructure:
            return BatchOriginalDataStructure(batch=data.batch.to(self.device),
                                              list=[g.to(self.device) for g in data.list],
                                              y=data.y.to(self.device),
                                              num_graphs=data.num_graphs)
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

        for data in dataloader.loader:
            optimizer.zero_grad()
            if optimizer_embd is not None:
                optimizer_embd.zero_grad()
            data = self.to_device(data)
            data, auxloss = self.construct_duplicate_data(data, emb_model)

            pred = model(data)
            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled])
            train_losses += loss.detach() * data.num_graphs

            loss = loss + auxloss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if optimizer_embd is not None:
                torch.nn.utils.clip_grad_norm_(emb_model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer_embd.step()

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
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()

        model.eval()
        preds = []
        labels = []

        for batch_id, data in enumerate(dataloader.loader):
            data = self.to_device(data)
            data, _ = self.construct_duplicate_data(data, emb_model)
            pred = model(data)
            label = data.y
            dataset_std = 1 if dataloader.std is None else dataloader.std.to(self.device)
            preds.append(pred * dataset_std)
            labels.append(label * dataset_std)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        is_labeled = labels == labels
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()

        val_metric = get_eval(self.task_type, labels, preds)

        early_stop = False
        if scheduler is not None:
            if isinstance(scheduler, MyPlateau):
                scheduler.step(val_metric)
            else:
                scheduler.step()
        if scheduler_embd is not None:
            if isinstance(scheduler_embd, MyPlateau):
                raise NotImplementedError("Need to specify max or min plateau")
            else:
                scheduler_embd.step()

        if not test:
            is_better, self.best_val_metric = self.metric_comparator(val_metric, self.best_val_metric)
            self.best_val_loss = min(self.best_val_loss, val_loss)

            if is_better:
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

        return val_loss, val_metric, early_stop

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.best_train_loss = 1e5
        self.best_train_metric = None
        self.patience = 0
        self.epoch = 0
