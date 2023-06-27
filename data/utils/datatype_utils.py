import time
from collections import namedtuple
from typing import Optional, Tuple

import torch

AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'std',
        'task',
    ])

DuoDataStructure = namedtuple(
    'DuoDataStructure', [
        'org',
        'candidates',
        'y',
        'num_graphs',
        'num_unique_graphs',
    ],
)


BatchOriginalDataStructure = namedtuple(
    'DuoDataStructure', [
        'batch',
        'list',
        'y',
        'num_graphs',
    ],
)


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
