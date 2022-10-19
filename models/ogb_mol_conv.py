# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from .encoder import AtomEncoder, BondEncoder, AtomEncoderPlaceholder


# GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
            emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

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
        lst = torch.nn.ModuleList(self.mlp)
        for l in lst:
            if not isinstance(l, torch.nn.ReLU):
                l.reset_parameters()
        self.eps = torch.nn.Parameter(torch.Tensor([0.]).to(self.eps.device))
        self.bond_encoder.reset_parameters()


# GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin', atom_embed=True):
        """
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        """

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom_embed:
            self.atom_encoder = AtomEncoder(emb_dim)
        else:
            self.atom_encoder = AtomEncoderPlaceholder()

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = data.edge_weight

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_weight)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        else:
            raise ValueError

        return node_representation

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        for c in self.convs:
            c.reset_parameters()
        for b in self.batch_norms:
            b.reset_parameters()


class GNN_node_Placeholder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GNN_node_Placeholder, self).__init__()
        pass

    def forward(self, data):
        return data.x

    def reset_parameters(self):
        pass
