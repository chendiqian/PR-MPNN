from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Linear, GELU, Sequential, BatchNorm1d as BN
from torch_geometric.nn import MessagePassing, MLP
from torch_geometric.utils import degree

from .my_encoder import BondEncoder


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int = 64, mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,):
        super(GINConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], batch_norm=True, dropout=0.)

        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        m = x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out


class BaseGIN(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden, out_feature, use_bn, dropout, use_residual, edge_encoder=None):
        super(BaseGIN, self).__init__()

        del edge_encoder
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = use_residual

        if num_layers == 0:
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()
        elif num_layers == 1:
            self.convs = torch.nn.ModuleList([GINConv(
                in_features,
                Sequential(
                    Linear(in_features, hidden),
                    GELU(),
                    Linear(hidden, out_feature),
                ),
            )])
            if use_bn:
                self.bns = torch.nn.ModuleList([BN(out_feature)])
        else:
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            # first layer
            self.convs.append(
                GINConv(
                    in_features,
                    Sequential(
                        Linear(in_features, hidden),
                        GELU(),
                        Linear(hidden, hidden),
                    ),
                )
            )
            if use_bn:
                self.bns.append(BN(hidden))

            # middle layer
            for i in range(num_layers - 2):
                self.convs.append(
                    GINConv(
                        hidden,
                        Sequential(
                            Linear(hidden, hidden),
                            GELU(),
                            Linear(hidden, hidden),
                        ))
                )
                if use_bn:
                    self.bns.append(BN(hidden))

            # last layer
            self.convs.append(
                GINConv(
                    hidden,
                    Sequential(
                        Linear(hidden, hidden),
                        GELU(),
                        Linear(hidden, out_feature),
                    ),
                )
            )
            if use_bn:
                self.bns.append(BN(out_feature))

    def forward(self, x, edge_index, edge_attr, edge_weight=None):

        for i, conv in enumerate(self.convs):
            x_new = conv(x,
                         edge_index[i] if isinstance(edge_index, (list, tuple)) else edge_index,
                         edge_attr,
                         edge_weight[i] if isinstance(edge_weight, (list, tuple)) else edge_weight)
            if self.use_bn:
                x_new = self.bns[i](x_new)
            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

        return x


class GINEConv(MessagePassing):
    def __init__(self, emb_dim: int = 64,
                 mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 bond_encoder: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 dropout: float = 0.):

        super(GINEConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], batch_norm=True, dropout=dropout)

        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

        if bond_encoder is None:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        else:
            self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else None
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out


class BaseGINE(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden, out_feature, use_bn, dropout, use_residual, edge_encoder=None):
        super(BaseGINE, self).__init__()

        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = use_residual

        if num_layers == 0:
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()
        elif num_layers == 1:
            self.convs = torch.nn.ModuleList([GINEConv(
                in_features,
                Sequential(
                    Linear(in_features, hidden),
                    GELU(),
                    Linear(hidden, out_feature),
                ),
                edge_encoder,
                dropout
            )])
            if use_bn:
                self.bns = torch.nn.ModuleList([BN(out_feature)])
        else:
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            # first layer
            self.convs.append(
                GINEConv(
                    in_features,
                    Sequential(
                        Linear(in_features, hidden),
                        GELU(),
                        Linear(hidden, hidden),
                    ),
                    edge_encoder,
                    dropout
                )
            )
            if use_bn:
                self.bns.append(BN(hidden))

            # middle layer
            for i in range(num_layers - 2):
                self.convs.append(
                    GINEConv(
                        hidden,
                        Sequential(
                            Linear(hidden, hidden),
                            GELU(),
                            Linear(hidden, hidden),
                        ),
                        edge_encoder,
                        dropout
                    )
                )
                if use_bn:
                    self.bns.append(BN(hidden))

            # last layer
            self.convs.append(
                GINEConv(
                    hidden,
                    Sequential(
                        Linear(hidden, hidden),
                        GELU(),
                        Linear(hidden, out_feature),
                    ),
                    edge_encoder,
                    dropout
                )
            )
            if use_bn:
                self.bns.append(BN(out_feature))

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x_new = conv(x,
                         edge_index[i] if isinstance(edge_index, (list, tuple)) else edge_index,
                         edge_attr[i] if isinstance(edge_attr, (list, tuple)) else edge_attr,
                         edge_weight[i] if isinstance(edge_weight, (list, tuple)) else edge_weight)
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


# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py#L37
class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, bond_encoder):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        """"""
        if edge_weight is not None and edge_weight.ndim == 1:
            edge_weight = edge_weight[:, None]

        if edge_attr is not None and self.bond_encoder is not None:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = None

        row, col = edge_index.detach()
        deg = degree(row, x.shape[0], dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_embedding, edge_weight=edge_weight)

    def message(self, x_j, norm, edge_attr, edge_weight):
        m = norm.view(-1, 1) * (torch.relu(x_j + edge_attr) if edge_attr is not None else x_j)
        return m * edge_weight if edge_weight is not None else m
