import math
import time
from collections import namedtuple
from typing import Union, Optional, Dict, Any, Tuple

import torch
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import index_sort

AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'mean',
        'std',
        'task',
    ])

DuoDataStructure = namedtuple(
    'DuoDataStructure', [
        'org',
        'candidates',
        'y',
        'num_graphs',
    ]
)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1):
    """
    https://github.com/rampasek/GraphGPS/blob/95a17d57767b34387907f42a43f91a0354feac05/graphgps/optimizer/extra_optimizers.py#L158

    Args:
        optimizer:
        num_warmup_steps:
        num_training_steps:
        num_cycles:
        last_epoch:

    Returns:

    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def scale_grad(model: torch.nn.Module, scalar: Union[int, float]) -> torch.nn.Module:
    """
    Scale down the grad, typically for accumulated grad while micro batching

    :param model:
    :param scalar:
    :return:
    """
    assert scalar > 0

    if scalar == 1:
        return model

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= scalar

    return model


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


class SyncMeanTimer:
    def __init__(self):
        self.count = 0
        self.mean_time = 0.
        self.last_start_time = 0.
        self.last_end_time = 0.

    @classmethod
    def synctimer(cls):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def __call__(self, start: bool):
        if start:
            self.last_start_time = self.synctimer()
            return self.last_start_time
        else:
            self.last_end_time = self.synctimer()
            self.mean_time = (self.mean_time * self.count + self.last_end_time - self.last_start_time) / (self.count + 1)
            self.count += 1
            return self.last_end_time


def batched_edge_index_to_batched_adj(data: Data, target_dtype: torch.dtype=torch.float):
    """

    Args:
        data: should be the original batch, i.e. without ensembles
        target_dtype:

    Returns:

    """
    device = data.x.device
    B = data.num_graphs
    N = (data._slice_dict['x'][1:] - data._slice_dict['x'][:-1]).max().item()
    num_edges = (data._slice_dict['edge_index'][1:] - data._slice_dict['edge_index'][:-1]).to(device)
    graph_idx_mask = torch.repeat_interleave(torch.arange(B, device=device), num_edges)
    edge_index_rel = torch.repeat_interleave(data._inc_dict['edge_index'].to(device), num_edges)
    local_edge_index = data.edge_index - edge_index_rel
    adj = torch.zeros(B, N, N, 1, dtype=target_dtype, device=device)
    adj[graph_idx_mask, local_edge_index[0], local_edge_index[1]] = 1
    return adj


def self_defined_softmax(scores, mask):
    """
    A specific function

    Args:
        scores: B, N, N, E
        mask: same shape as scores

    Returns:

    """
    scores = scores - scores.detach().max()  # for numerical stability
    exp_scores = torch.exp(scores)
    exp_scores = exp_scores * mask
    softmax_scores = exp_scores / exp_scores.sum()
    return softmax_scores

def multiply_with_mask(scores, mask):
    """
    A specific function

    Args:
        scores: B, N, N, E
        mask: same shape as scores

    Returns:

    """
    scores = scores * mask
    return scores


def unflatten(
    d: Dict[str, Any],
    base: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Convert any keys containing dotted paths to nested dicts

    >>> unflatten({'a': 12, 'b': 13, 'c': 14})  # no expansion
    {'a': 12, 'b': 13, 'c': 14}

    >>> unflatten({'a.b.c': 12})  # dotted path expansion
    {'a': {'b': {'c': 12}}}

    >>> unflatten({'a.b.c': 12, 'a': {'b.d': 13}})  # merging
    {'a': {'b': {'c': 12, 'd': 13}}}

    >>> unflatten({'a.b': 12, 'a': {'b': 13}})  # insertion-order overwrites
    {'a': {'b': 13}}

    >>> unflatten({'a': {}})  # insertion-order overwrites
    {'a': {}}
    """
    if base is None:
        base = {}

    for key, value in d.items():
        root = base

        ###
        # If a dotted path is encountered, create nested dicts for all but
        # the last level, then change root to that last level, and key to
        # the final key in the path.
        #
        # This allows one final setitem at the bottom of the loop.
        #
        if '.' in key:
            *parts, key = key.split('.')

            for part in parts:
                root.setdefault(part, {})
                root = root[part]

        if isinstance(value, dict):
            value = unflatten(value, root.get(key, {}))

        root[key] = value

    return base


def set_nonetype(dic: Dict):
    for k, v in dic.items():
        if isinstance(v, str) and v.lower() == 'none':
            dic[k] = None
        elif isinstance(v, dict):
            dic[k] = set_nonetype(v)
    return dic


class MyPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    https://github.com/rampasek/GraphGPS/blob/main/graphgps/loss/weighted_cross_entropy.py
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.shape[1] > 2 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.shape[1] > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(), weight=weight[true])
        return loss


def non_merge_coalesce(
    edge_index,
    edge_attr,
    edge_weight,
    num_nodes,
    is_sorted: bool = False,
    sort_by_row: bool = True,
):
    nnz = edge_index.size(1)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        edge_index = edge_index[:, perm]
        if isinstance(edge_attr, torch.Tensor):
            edge_attr = edge_attr[perm]
        if isinstance(edge_weight, torch.Tensor):
            edge_weight = edge_weight[perm]

    return edge_index, edge_attr, edge_weight
