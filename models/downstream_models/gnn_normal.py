import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set

from models.my_convs import BaseGIN, BaseGINE
from models.nn_modules import MLP


class GNN_Normal(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
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
            model_class = BaseGIN
        elif base_gnn == 'gine':
            model_class = BaseGINE
        else:
            raise NotImplementedError

        self.gnn = model_class(in_features, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        self.graph_pooling =graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=6)
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.candid_pool = self.pool
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.candid_pool = self.pool
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hidden if graph_pooling != 'set2set' else hidden * 2] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        if self.encoder is not None:
            data.x = self.encoder(data)

        h_node = self.gnn(data.x, data.edge_index, data.edge_attr, data.edge_weight)
        h_graph = self.pool(h_node, getattr(data, self.graph_pool_idx))
        h_graph = self.mlp(h_graph)
        return h_graph
