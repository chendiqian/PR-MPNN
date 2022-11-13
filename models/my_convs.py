import torch
from torch_geometric.nn import MessagePassing

from models.encoder import BondEncoder


class GINConv(MessagePassing):
    def __init__(self, emb_dim=64, mlp=None):
        """
            emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

        if mlp is not None:
            self.mlp = mlp
        else:
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


class GNN_Placeholder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GNN_Placeholder, self).__init__()
        pass

    def forward(self, data):
        return data.x

    def reset_parameters(self):
        pass
