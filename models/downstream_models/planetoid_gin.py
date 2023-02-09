import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter

from models.my_convs import GINConv
from models.nn_utils import MLP


class PlanetoidGIN(torch.nn.Module):
    def __init__(self, num_convlayers, in_features, hid, num_classes, dropout, aggr=None):
        super(PlanetoidGIN, self).__init__()

        self.gc = torch.nn.ModuleList([GINConv(hid, MLP([in_features, hid], dropout=0.))])
        for l in range(num_convlayers - 2):
            self.gc.append(GINConv(hid, MLP([hid, hid], dropout=0.)))
        self.gc.append(GINConv(hid, MLP([hid, num_classes], dropout=0.)))

        self.dropout = dropout

        assert aggr in ['center', 'mean', 'add', 'sum']
        self.aggr_type = aggr
        if aggr == 'center':
            self.pool = lambda x, target_mask, *args: x[target_mask]
        elif aggr == 'mean':
            self.pool = lambda x, batch, node_mask: global_add_pool(x, batch) / scatter(
                node_mask.to(torch.float).detach(), batch, dim=0, reduce="sum")[:, None] \
                if node_mask is not None else global_mean_pool(x, batch)
        elif aggr in ['add', 'sum']:
            self.pool = lambda x, batch, *args: global_add_pool(x, batch)

    def forward(self, data, *args):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, l in enumerate(self.gc):
            x = l(x, edge_index, edge_weight)
            if i < len(self.gc) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if hasattr(data, 'node_mask') and data.node_mask.dtype == torch.float:
            x = x * data.node_mask[:, None]

        if self.aggr_type == 'center':
            return self.pool(x, data.target_mask)
        else:
            node_mask = data.node_mask if hasattr(data, 'node_mask') else None
            return self.pool(x, data.batch, node_mask)

    def reset_parameters(self):
        for g in self.gc:
            g.reset_parameters()
