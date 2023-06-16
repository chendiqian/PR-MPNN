from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import MessagePassing

from .nn_modules import MLP
from .nn_utils import reset_sequential_parameters, residual
from .my_encoder import BondEncoder


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int = 64, mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,):
        super(GINConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], batch_norm=True, dropout=0.)

        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, edge_index, edge_weight):
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

    def reset_parameters(self):
        self.eps = torch.nn.Parameter(torch.Tensor([0.]).to(self.eps.device))
        if isinstance(self.mlp, (MLP, torch.nn.Linear)):
            self.mlp.reset_parameters()
        elif isinstance(self.mlp, torch.nn.Sequential):
            reset_sequential_parameters(self.mlp)
        else:
            raise TypeError


class BaseGIN(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden, out_feature, use_bn, dropout, use_residual, edge_encoder=None):
        super(BaseGIN, self).__init__()

        del edge_encoder
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = use_residual

        if num_layers == 1:
            self.convs = torch.nn.ModuleList([GINConv(
                in_features,
                Sequential(
                    Linear(in_features, hidden),
                    ReLU(),
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
                        ReLU(),
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
                            ReLU(),
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
                        ReLU(),
                        Linear(hidden, out_feature),
                    ),
                )
            )
            if use_bn:
                self.bns.append(BN(out_feature))

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        del edge_attr

        for i, conv in enumerate(self.convs):
            x_new = conv(x,
                         edge_index[i] if isinstance(edge_index, (list, tuple)) else edge_index,
                         edge_weight[i] if isinstance(edge_weight, (list, tuple)) else edge_weight)
            if self.use_bn:
                x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

        return x


class GINEConv(MessagePassing):
    def __init__(self, emb_dim: int = 64,
                 mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 bond_encoder: Optional[Union[MLP, torch.nn.Sequential]] = None):

        super(GINEConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], batch_norm=True, dropout=0.)

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
        m = torch.relu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        raise NotImplementedError


class BaseGINE(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden, out_feature, use_bn, dropout, use_residual, edge_encoder=None):
        super(BaseGINE, self).__init__()

        self.use_bn = use_bn
        self.dropout = dropout
        self.use_residual = use_residual

        if num_layers == 1:
            self.convs = torch.nn.ModuleList([GINEConv(
                in_features,
                Sequential(
                    Linear(in_features, hidden),
                    ReLU(),
                    Linear(hidden, out_feature),
                ),
                edge_encoder,
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
                        ReLU(),
                        Linear(hidden, hidden),
                    ),
                    edge_encoder,
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
                            ReLU(),
                            Linear(hidden, hidden),
                        ),
                        edge_encoder,
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
                        ReLU(),
                        Linear(hidden, out_feature),
                    ),
                    edge_encoder,
                )
            )
            if use_bn:
                self.bns.append(BN(out_feature))

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x_new = conv(x,
                         edge_index[i] if isinstance(edge_index, (list, tuple)) else edge_index,
                         edge_attr[i] if isinstance(edge_attr, (list, tuple)) else edge_attr,
                         edge_weight[i] if isinstance(edge_weight, (list, tuple)) else edge_weight)
            if self.use_bn:
                x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

        return x


# class SAGEConv(MessagePassing):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         project: bool = False,
#     ):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.project = project
#
#         super(SAGEConv, self).__init__()
#
#         if self.project:
#             self.lin = torch.nn.Linear(in_channels, in_channels, bias=True)
#
#         self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=True)
#         self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.project:
#             self.lin.reset_parameters()
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()
#
#     def forward(self, x, edge_index, edge_weight):
#         if edge_weight is not None and edge_weight.ndim < 2:
#             edge_weight = edge_weight[:, None]
#
#         x = (x, x)
#
#         if self.project and hasattr(self, 'lin'):
#             x = (self.lin(x[0]).relu(), x[1])
#
#         # propagate_type: (x: OptPairTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
#         out = self.lin_l(out)
#
#         x_r = x[1]
#         out = out + self.lin_r(x_r)
#
#         return out
#
#     def message(self, x_j, edge_weight):
#         return x_j * edge_weight if edge_weight is not None else x_j
#
#
class GNN_Placeholder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GNN_Placeholder, self).__init__()
        pass

    def forward(self, x, *args):
        return x

    def reset_parameters(self):
        pass
#
#
# class GCNConv(MessagePassing):
#
#     def __init__(self, in_channels: int, out_channels: int):
#         super(GCNConv, self).__init__(aggr='add')
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#
#     def forward(self, x, edge_index, edge_weight):
#         """"""
#         if edge_weight is not None and edge_weight.ndim == 1:
#             edge_weight = edge_weight[:, None]
#
#         row, col = edge_index
#         deg = degree(row, x.shape[0], dtype=torch.float)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#         x = self.lin(x)
#         return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)
#
#     def message(self, x_j, norm, edge_weight):
#         m = norm.view(-1, 1) * x_j
#         return m * edge_weight if edge_weight is not None else m
