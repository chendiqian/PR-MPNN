# adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter
from models.my_convs import GCNConv


class PlanetoidGCN(torch.nn.Module):
    def __init__(self, num_convlayers, in_features, hid, num_classes, dropout, aggr=None):
        super(PlanetoidGCN, self).__init__()

        self.gc = torch.nn.ModuleList([GCNConv(in_features, hid)])
        for l in range(num_convlayers - 2):
            self.gc.append(GCNConv(hid, hid))
        self.gc.append(GCNConv(hid, num_classes))

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
