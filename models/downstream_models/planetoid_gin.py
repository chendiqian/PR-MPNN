import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from models.my_convs import GINConv
from models.nn_utils import MLP


class PlanetoidGIN(torch.nn.Module):
    def __init__(self, num_convlayers, in_features, hid, num_classes, dropout, aggr=None):
        super(PlanetoidGIN, self).__init__()

        self.gc = torch.nn.ModuleList([GINConv(hid, MLP([in_features, hid], norm=False, dropout=0.))])
        for l in range(num_convlayers - 2):
            self.gc.append(GINConv(hid, MLP([hid, hid], norm=False, dropout=0.)))
        self.gc.append(GINConv(hid, MLP([hid, num_classes], norm=False, dropout=0.)))

        self.dropout = dropout
        self.aggr = aggr

    def forward(self, data, *args):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, l in enumerate(self.gc):
            x = l(x, edge_index, edge_weight)
            if i < len(self.gc) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # return global_add_pool(x, data.batch)
        return x[data.target_mask]

    def reset_parameters(self):
        for g in self.gc:
            g.reset_parameters()