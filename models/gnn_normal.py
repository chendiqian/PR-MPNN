import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from models.my_convs import BaseGINE
from torch_geometric.nn import MLP


class GNN_Normal(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
                 num_layers,
                 hidden,
                 num_classes,
                 use_bn,
                 dropout,
                 residual,
                 mlp_layers_intragraph,
                 graph_pooling='mean'):
        super(GNN_Normal, self).__init__()

        self.encoder = encoder

        if base_gnn == 'gine':
            self.gnn = BaseGINE(num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        else:
            raise NotImplementedError

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling =graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        if mlp_layers_intragraph > 0:
            self.mlp = MLP([hidden] * mlp_layers_intragraph + [num_classes], dropout=0., norm=None)
        else:
            self.mlp = lambda x: x

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        if self.encoder is not None:
            data.x = self.encoder(data)

        h_node = self.gnn(data.x, data.edge_index, data.edge_attr, data.edge_weight)
        h_graph = self.pool(h_node, getattr(data, self.graph_pool_idx))
        h_graph = self.mlp(h_graph)
        return h_graph, [data], 0.
