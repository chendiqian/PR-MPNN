import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set

from models.my_convs import GNN_Placeholder, BaseGIN
from models.nn_utils import cat_pooling
from models.nn_modules import MLP


class ZINC_GIN(torch.nn.Module):
    def __init__(self,
                 ensemble,
                 in_features,
                 num_layers,
                 hidden,
                 num_classes,
                 mlp_layers_intragraph,
                 mlp_layers_intergraph,
                 graph_pooling='mean',
                 inter_graph_pooling=None):
        super(ZINC_GIN, self).__init__()

        if num_layers > 0:
            self.gnn = BaseGIN(in_features, num_layers, hidden)
        else:
            self.gnn = GNN_Placeholder()

        # intra-graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        self.inter_graph_pooling = inter_graph_pooling
        # inter-graph pooling
        if self.inter_graph_pooling is None or inter_graph_pooling == 'None':
            self.mlp1 = MLP([hidden] * mlp_layers_intragraph + [num_classes], dropout=0.)
            self.inter_pool = None
            self.mlp2 = None
        elif inter_graph_pooling == 'mean':
            assert mlp_layers_intergraph > 0
            self.mlp1 = MLP([hidden] * (mlp_layers_intragraph + 1), dropout=0.)
            self.inter_pool = global_mean_pool
            self.mlp2 = MLP([hidden] * mlp_layers_intergraph + [num_classes], dropout=0.)
        elif inter_graph_pooling == 'cat':
            assert mlp_layers_intergraph > 0
            self.mlp1 = MLP([hidden] * (mlp_layers_intragraph + 1), dropout=0.)
            self.inter_pool = cat_pooling
            self.mlp2 = MLP([hidden * ensemble] + [hidden] * (mlp_layers_intergraph - 1) + [num_classes], dropout=0.)
        elif inter_graph_pooling == 'set2set':
            assert mlp_layers_intergraph > 0
            self.mlp1 = MLP([hidden] * (mlp_layers_intragraph + 1), dropout=0.)
            self.inter_pool = Set2Set(hidden, processing_steps=6)
            self.mlp2 = MLP([hidden * 2] + [hidden] * (mlp_layers_intergraph - 1) + [num_classes], dropout=0.)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp1.reset_parameters()
        if self.mlp2 is not None:
            self.mlp2.reset_parameters()

    def forward(self, data):
        h_node = self.gnn(data)

        if self.inter_graph_pooling is None or self.inter_graph_pooling == 'None':
            h_graph = self.pool(h_node, data.batch)
            if hasattr(data, 'inter_graph_idx'):
                h_graph = global_mean_pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp1(h_graph)
        elif self.inter_graph_pooling in ['mean', 'cat', 'set2set']:
            h_graph = self.pool(h_node, data.batch)
            h_graph = self.mlp1(h_graph)
            # inter graph pooling
            if hasattr(data, 'inter_graph_idx'):
                h_graph = self.inter_pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp2(h_graph)
        else:
            raise NotImplementedError

        return h_graph
