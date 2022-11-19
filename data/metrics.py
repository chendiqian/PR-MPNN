# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py

from typing import Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def pre_proc(y1, y2):
    if len(y1.shape) == 1:
        y1 = y1[:, None]
    if len(y2.shape) == 1:
        y2 = y2[:, None]
    if isinstance(y1, torch.Tensor):
        y1 = y1.detach().cpu().numpy()
    if isinstance(y2, torch.Tensor):
        y2 = y2.detach().cpu().numpy()
    return y1, y2


def eval_rocauc(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
        compute ROC-AUC averaged across tasks
    """

    y_true, y_pred = pre_proc(y_true, y_pred)

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.any(y_true[:, i] == 1) and np.any(y_true[:, i] == 0):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def eval_acc(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    eval accuracy (potentially multi task)

    :param y_true:
    :param y_pred:
    :return:
    """

    y_true, y_pred = pre_proc(y_true, y_pred)

    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rmse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    y_true, y_pred = pre_proc(y_true, y_pred)

    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)
