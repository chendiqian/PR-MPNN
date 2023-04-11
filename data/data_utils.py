import math
import time
from collections import namedtuple
from typing import Union, Optional, Dict, Any

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch_geometric.data import Data

AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'mean',
        'std'
    ])


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

    def __call__(self, val1: float, val2: Optional[float]) -> bool:
        if val2 is None:
            return True

        if self.task_type in ['regression', 'rmse']:
            return val1 <= val2
        elif self.task_type in ['rocauc', 'acc']:
            return val1 >= val2
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
