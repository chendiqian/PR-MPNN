import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, global_add_pool

from models.my_convs import GINConv, GNN_Placeholder
from models.nn_utils import residual, MLP


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
                 in_features,
                 num_layers,
                 hidden,
                 num_classes,
                 mlp_layers_intragraph,
                 mlp_layers_intergraph,
                 graph_pooling='mean', ):
        super(ZINC_GIN, self).__init__()

        if num_layers > 0:
            self.gnn = BaseGIN(in_features, num_layers, hidden)
        else:
            self.gnn = GNN_Placeholder()

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        assert mlp_layers_intragraph > 0
        if mlp_layers_intergraph > 0:
            self.mlp1 = MLP([hidden] * (mlp_layers_intragraph + 1), dropout=0.)
            self.mlp2 = MLP([hidden] * mlp_layers_intergraph + [num_classes], dropout=0.)
        else:
            self.mlp1 = MLP([hidden] * mlp_layers_intragraph + [num_classes], dropout=0.)
            self.mlp2 = None

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp1.reset_parameters()
        if self.mlp2 is not None:
            self.mlp2.reset_parameters()

    def forward(self, data):
        h_node = self.gnn(data)

        if self.mlp2 is None:
            # intra graph pooling
            h_graph = self.pool(h_node, data.batch)
            # inter graph pooling
            if hasattr(data, 'inter_graph_idx'):
                h_graph = self.pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp1(h_graph)
        else:
            # intra graph pooling
            h_graph = self.pool(h_node, data.batch)
            h_graph = self.mlp1(h_graph)
            # inter graph pooling
            if hasattr(data, 'inter_graph_idx'):
                h_graph = self.pool(h_graph, data.inter_graph_idx)
            h_graph = self.mlp2(h_graph)

        return h_graph
