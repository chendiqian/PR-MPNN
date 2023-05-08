# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

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


def eval_rocauc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        compute ROC-AUC averaged across tasks
    """
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


def eval_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    eval accuracy (potentially multi task)

    :param y_true:
    :param y_pred:
    :return:
    """
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)


def eval_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mae_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        mae_list.append(np.abs(y_true[is_labeled, i] - y_pred[is_labeled, i]).mean())

    return sum(mae_list) / len(mae_list)

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
        From:
        https://github.com/XiaoxinHe/Graph-MLPMixer/blob/48cd68f9e92a7ecbf15aea0baf22f6f338b2030e/train/peptides_func.py
    '''

    ap_list = []
    # check if y_true and y_pred are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


def get_eval(task_type: str, y_true: torch.Tensor, y_pred: torch.Tensor):
    if task_type == 'rocauc':
        func = eval_rocauc
    elif task_type == 'rmse':
        func = eval_rmse
    elif task_type == 'acc':
        if y_pred.shape[1] == 1:
            y_pred = (y_pred > 0.).to(torch.int)
        else:
            y_pred = torch.argmax(y_pred, dim=1)
        func = eval_acc
    elif task_type == 'mae':
        func = eval_mae
    elif task_type == 'ap':
        func = eval_ap
    else:
        raise NotImplementedError

    y_true, y_pred = pre_proc(y_true, y_pred)
    metric = func(y_true, y_pred)
    return metric
