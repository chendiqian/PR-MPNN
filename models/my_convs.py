from typing import Union

import torch
from torch.nn import GELU, Sequential, BatchNorm1d as BN, functional as F
from torch_geometric.nn import MessagePassing, MLP, Linear


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


class GINEConv(MessagePassing):
    def __init__(self,
                 mlp: Union[MLP, torch.nn.Sequential],
                 bond_encoder: Union[MLP, torch.nn.Sequential]):
        super(GINEConv, self).__init__(aggr="add")

        self.mlp = mlp
        self.bond_encoder = bond_encoder
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]
        elif edge_weight is None:
            edge_weight = x.new_ones(edge_index.shape[1], 1)

        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else None
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight

    def update(self, aggr_out):
        return aggr_out


class BaseGINE(torch.nn.Module):
    def __init__(self, num_layers, hidden, out_feature, use_bn, dropout, use_residual, edge_encoder=None):
        super(BaseGINE, self).__init__()

        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = use_residual

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GINEConv(
                Sequential(
                    Linear(-1, hidden),
                    GELU(),
                    Linear(-1, out_feature if i == num_layers - 1 else hidden),
                ),
                edge_encoder,
            ))
            if use_bn:
                self.bns.append(BN(out_feature if i == num_layers - 1 else hidden))

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr, edge_weight)
            if self.use_bn:
                x_new = self.bns[i](x_new)
            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

        return x


class GNN_Placeholder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GNN_Placeholder, self).__init__()
        pass

    def forward(self, x, *args):
        return x
