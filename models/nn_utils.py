from typing import List

import torch


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


def reset_sequential_parameters(seq: torch.nn.Sequential) -> None:
    lst = torch.nn.ModuleList(seq)
    for l in lst:
        if not isinstance(l, (torch.nn.ReLU, torch.nn.Dropout)):
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
    def __init__(self, hidden_dims: List, norm: bool, dropout: float = 0.5):
        super(MLP, self).__init__()

        num_layers = len(hidden_dims) - 1
        modules = []
        for i in range(num_layers):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if norm and i < num_layers - 1:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))
            if i < num_layers - 1:
                modules.append(torch.nn.ReLU())
                modules.append(torch.nn.Dropout(p=dropout))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

    def reset_parameters(self):
        reset_sequential_parameters(self.mlp)
