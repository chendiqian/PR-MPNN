import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set, global_max_pool

from models.my_convs import BaseGIN, BaseGINE, BasePNA, BaseGCN
from models.nn_modules import MLP


class GNN_Normal(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
                 deg_hist,
                 in_features,
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

        if base_gnn == 'gin':
            self.gnn = BaseGIN(in_features, num_layers, hidden, hidden, use_bn,
                                   dropout, residual, edge_encoder)
        elif base_gnn == 'gine':
            self.gnn = BaseGINE(in_features, num_layers, hidden, hidden, use_bn,
                               dropout, residual, edge_encoder)
        elif base_gnn == 'pna':
            assert deg_hist is not None
            self.gnn = BasePNA(in_features, num_layers, hidden, hidden, use_bn,
                                dropout, residual, deg_hist, edge_encoder)
        elif base_gnn == 'gcn':
            self.gnn = BaseGCN(in_features, num_layers, hidden, hidden, use_bn,
                                dropout, residual, edge_encoder)
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
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=6)
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
            self.mlp = MLP([hidden if graph_pooling != 'set2set' else hidden * 2] * mlp_layers_intragraph + [num_classes], dropout=0.)
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
        return h_graph
