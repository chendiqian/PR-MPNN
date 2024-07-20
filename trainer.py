from typing import Any, Union, Optional, Tuple

import torch
from data.metrics import get_eval
from torch_geometric.loader import DataLoader


class IsBetter:
    """
    A comparator for different metrics, to unify >= and <=

    """
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self, val1: float, val2: Optional[float]) -> Tuple[bool, float]:
        if val2 is None:
            return True, val1

        if self.task_type in ['regression', 'rmse', 'mae']:
            better = val1 < val2
            the_better = val1 if better else val2
            return better, the_better
        elif self.task_type in ['rocauc', 'acc', 'f1_macro', 'ap']:
            better = val1 > val2
            the_better = val1 if better else val2
            return better, the_better
        else:
            raise ValueError


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
              dataloader: DataLoader,
              model,
              optimizer):
        model.train()
        train_losses = torch.tensor(0., device=self.device)
        preds = []
        labels = []
        num_graphs = 0

        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(self.device)
            pred, new_data, auxloss = model(data)
            label = new_data[0].y
            ngraphs = new_data[0].num_graphs

            is_labeled = label == label
            loss = self.criterion(pred[is_labeled], label[is_labeled])
            train_losses += loss.detach() * ngraphs

            loss = loss + auxloss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            num_graphs += ngraphs
            preds.append(pred)
            labels.append(label)
        train_loss = train_losses.item() / num_graphs
        self.best_train_loss = min(self.best_train_loss, train_loss)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric = get_eval(self.task_type, labels, preds)

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: DataLoader,
                  model,
                  dataset_std: float,
                  scheduler: Any = None,
                  test: bool = False):
        model.eval()
        preds = []
        labels = []

        for data in dataloader:
            data = data.to(self.device)
            pred, new_data, _ = model(data)
            label = new_data[0].y
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
