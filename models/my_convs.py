from typing import Optional, Union

import torch
from torch.nn import Linear, GELU, Sequential, BatchNorm1d as BN, BatchNorm1d, ReLU, ModuleList, functional as F
from torch_geometric.nn import MessagePassing, MLP, global_add_pool, global_max_pool, global_mean_pool

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
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], norm='batch_norm', dropout=0.)

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
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], norm='batch_norm', dropout=dropout)

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


class QM9_Net(torch.nn.Module):
    def __init__(
        self,
        encoder,
        gnn_type,
        edge_encoder,
        num_classes,
        emb_sizes,
        num_layers,
        drpt_prob=0.5,
        graph_pooling="max",
    ):
        super(QM9_Net, self).__init__()
        self.drpt_prob = drpt_prob
        self.graph_pooling = graph_pooling

        self.initial_mlp = encoder
        self.initial_linear = Linear(emb_sizes, num_classes)

        gnn_layers = []
        linears = []
        # mlps = []
        for i in range(num_layers):
            if gnn_type in ['gin', 'gine']:
                mlp = Sequential(
                        Linear(emb_sizes, emb_sizes),
                        BatchNorm1d(emb_sizes),
                        ReLU(),
                        Linear(emb_sizes, emb_sizes),
                        BatchNorm1d(emb_sizes),
                        ReLU(),
                )
            # mlps.append(mlp)
                if gnn_type == 'gin':
                    gnn_layer = GINConv(emb_sizes, mlp)
                else:
                    gnn_layer = GINEConv(emb_sizes, mlp, edge_encoder)
            else:
                raise NotImplementedError
            gnn_layers.append(gnn_layer)
            linears.append(Linear(emb_sizes, num_classes))

        self.gnn_modules = ModuleList(gnn_layers)
        self.linear_modules = ModuleList(linears)

        if graph_pooling == "sum":
            self.pooling = global_add_pool
        elif graph_pooling == 'max':
            self.pooling = global_max_pool
        elif graph_pooling == "mean":
            self.pooling = global_mean_pool

    def forward(self, data):
        x_feat = self.initial_mlp(data)  # Otherwise by an MLP
        out = F.dropout(
            self.pooling(self.initial_linear(x_feat), data.batch), p=self.drpt_prob
        )

        for gin_layer, linear_layer in zip(self.gnn_modules, self.linear_modules):
            x_feat = gin_layer(x_feat, data.edge_index, data.edge_attr, data.edge_weight)

            out += F.dropout(
                linear_layer(self.pooling(x_feat, data.batch)),
                p=self.drpt_prob,
                training=self.training,
            )

        return out
