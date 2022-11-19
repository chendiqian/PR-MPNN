# adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
import torch
import torch.nn.functional as F
from .my_convs import GCNConv


class CoraGCN(torch.nn.Module):
    def __init__(self, num_convlayers, in_features, hid, num_classes, dropout, aggr=None):
        super(CoraGCN, self).__init__()

        self.gc = torch.nn.ModuleList([GCNConv(in_features, hid)])
        for l in range(num_convlayers - 2):
            self.gc.append(GCNConv(hid, hid))
        self.gc.append(GCNConv(hid, num_classes))

        self.dropout = dropout
        self.aggr = aggr

    def forward(self, data, *args):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, l in enumerate(self.gc):
            x = l(x, edge_index, edge_weight)
            if i < len(self.gc) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x[data.target_mask]

    def reset_parameters(self):
        for g in self.gc:
            g.reset_parameters()
