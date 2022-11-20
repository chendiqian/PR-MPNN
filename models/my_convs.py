from typing import Optional, Union

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import degree

from .my_encoder import BondEncoder
from .nn_utils import reset_sequential_parameters, MLP


class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True):

        super(GCNConv, self).__init__(aggr='add')

        self.add_self_loops = add_self_loops
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        """"""
        if self.add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index, None, 1., x.shape[0])

        row, col = edge_index
        deg = degree(row, x.shape[0], dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, norm, edge_weight):
        m = norm.view(-1, 1) * x_j
        return m * edge_weight if edge_weight is not None else m


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int = 64, mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,):
        super(GINConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], norm=True, dropout=0.)

        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, edge_index, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        m = torch.relu(x_j)
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


class GINEConv(MessagePassing):
    def __init__(self, emb_dim: int = 64,
                 mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 bond_encoder: Optional[Union[MLP, torch.nn.Sequential]] = None):

        super(GINEConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MLP([emb_dim, emb_dim, emb_dim], norm=True, dropout=0.)

        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

        if bond_encoder is None:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        else:
            self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = torch.relu(x_j + edge_attr)
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

        if isinstance(self.bond_encoder, (BondEncoder, MLP, torch.nn.Linear)):
            self.bond_encoder.reset_parameters()
        elif isinstance(self.bond_encoder, torch.nn.Sequential):
            reset_sequential_parameters(self.bond_encoder)
        else:
            raise TypeError


class GNN_Placeholder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GNN_Placeholder, self).__init__()
        pass

    def forward(self, data):
        return data.x

    def reset_parameters(self):
        pass
