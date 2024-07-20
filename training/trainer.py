from typing import Any, Union

import torch

from data.metrics import get_eval
from data.utils.datatype_utils import AttributedDataLoader, IsBetter


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 max_patience: int,
                 criterion: torch.nn.modules.loss,
                 device: Union[str, torch.device]):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.task_type = task_type
        self.metric_comparator = IsBetter(self.task_type)
        self.criterion = criterion
        self.device = device
        self.max_patience = max_patience

        # clear best scores
        self.clear_stats()

    def train(self,
              dataloader: AttributedDataLoader,
              model,
              optimizer):
        model.train()
        train_losses = torch.tensor(0., device=self.device)
        preds = []
        labels = []
        num_graphs = 0

        for data in dataloader.loader:
            optimizer.zero_grad()
            data = data.to(self.device)
            pred, new_data, auxloss = model(data)

            is_labeled = new_data.y == new_data.y
            loss = self.criterion(pred[is_labeled], new_data.y[is_labeled])
            train_losses += loss.detach() * new_data.num_graphs

            loss = loss + auxloss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            num_graphs += new_data.num_graphs
            preds.append(pred)
            labels.append(new_data.y)
        train_loss = train_losses.item() / num_graphs
        self.best_train_loss = min(self.best_train_loss, train_loss)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric = get_eval(self.task_type, labels, preds)

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  model,
                  scheduler: Any = None,
                  test: bool = False):
        model.eval()
        preds = []
        labels = []

        for data in dataloader.loader:
            data = data.to(self.device)
            pred, new_data, _ = model(data)
            label = new_data.y
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
            scheduler.step(val_metric)

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
