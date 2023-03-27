from collections import namedtuple
from typing import Union, Optional
import time

import torch


AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'mean',
        'std'
    ])


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
