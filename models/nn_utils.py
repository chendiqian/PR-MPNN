from typing import List

import torch
from torch import nn as nn


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


def reset_sequential_parameters(seq: torch.nn.Sequential) -> None:
    lst = torch.nn.ModuleList(seq)
    for l in lst:
        if not isinstance(l, (torch.nn.ReLU, torch.nn.Dropout, torch.nn.GELU)):
            l.reset_parameters()
    # return torch.nn.Sequential(**lst)


def reset_modulelist_parameters(seq: torch.nn.ModuleList) -> None:
    for l in seq:
        if not isinstance(l, torch.nn.ReLU):
            l.reset_parameters()


class OneHot(torch.nn.Module):
    def __init__(self, num_classes: int, target_dtype: torch.dtype = torch.float):
        super(OneHot, self).__init__()
        self.num_classes = num_classes
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.one_hot(x, self.num_classes).to(self.target_dtype)

    def reset_parameters(self):
        pass


class MLP(torch.nn.Module):
    def __init__(self, hidden_dims: List,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 dropout: float = 0.5,
                 activate_last: bool = False):
        super(MLP, self).__init__()

        assert not (batch_norm and layer_norm)   # cannot be both true

        num_layers = len(hidden_dims) - 1
        modules = []
        for i in range(num_layers):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=i < num_layers - 1))
            if batch_norm and i < num_layers - 1:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))
            if layer_norm and i < num_layers - 1:
                modules.append(torch.nn.LayerNorm(hidden_dims[i + 1]))
            if i < num_layers - 1 or activate_last:
                modules.append(torch.nn.ReLU())
                modules.append(torch.nn.Dropout(p=dropout))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

    def reset_parameters(self):
        reset_sequential_parameters(self.mlp)


def cat_pooling(x, graph_idx):
    """

    Args:
        x:
        graph_idx: should be like (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3) in case of 4 graphs, 3 ensembles

    Returns:

    """
    unique, ensemble = torch.unique(graph_idx, return_counts=True)
    assert len(torch.unique(ensemble)) == 1, "each graph should have the same number of ensemble!"
    n_ensemble = ensemble[0]
    n_graphs = len(unique)

    new_x = x.new_empty(n_graphs, n_ensemble, x.shape[-1])
    ensemble_idx = torch.arange(n_ensemble, device=x.device).repeat_interleave(n_graphs)
    new_x[graph_idx, ensemble_idx, :] = x
    return new_x.reshape(n_graphs, -1)


class BiEmbedding(torch.nn.Module):
    def __init__(self,
                 dim_in,
                 hidden,):
        super(BiEmbedding, self).__init__()
        self.layer0_keys = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)
        self.layer0_values = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)

    def forward(self, x):
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed
        return x

    def reset_parameters(self):
        self.layer0_keys.reset_parameters()
        self.layer0_values.reset_parameters()
