import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, global_add_pool

from models.my_convs import GINConv, GNN_Placeholder
from models.nn_utils import residual, MLP, cat_pooling


class BaseGIN(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden):
        super(BaseGIN, self).__init__()

        assert num_layers > 0
        self.conv1 = GINConv(
            hidden,
            Sequential(
                Linear(in_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
        )

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    hidden,
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        ReLU(),
                    ))
                )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight

        x = self.conv1(x, edge_index, edge_weight)
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_weight)
            x = residual(x, x_new)

        return x


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
        if inter_graph_pooling == 'None':
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
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp1.reset_parameters()
        if self.mlp2 is not None:
            self.mlp2.reset_parameters()

    def forward(self, data):
        h_node = self.gnn(data)

        if self.inter_graph_pooling == 'None':
            h_graph = self.pool(h_node, data.batch)
            if hasattr(data, 'inter_graph_idx'):
                h_graph = global_mean_pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp1(h_graph)
        elif self.inter_graph_pooling in ['mean', 'cat']:
            h_graph = self.pool(h_node, data.batch)
            h_graph = self.mlp1(h_graph)
            # inter graph pooling
            if hasattr(data, 'inter_graph_idx'):
                h_graph = self.inter_pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp2(h_graph)
        else:
            raise NotImplementedError

        return h_graph
