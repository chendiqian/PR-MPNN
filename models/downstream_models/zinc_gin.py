import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter

from models.my_convs import GINEConv, GNN_Placeholder
from models.nn_utils import residual, MLP


class BaseGIN(torch.nn.Module):
    def __init__(self, in_features, edge_features, num_layers, hidden):
        super(BaseGIN, self).__init__()

        assert num_layers > 0
        self.conv1 = GINEConv(
            hidden,
            Sequential(
                Linear(in_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            bond_encoder=MLP([edge_features, in_features, in_features], dropout=0., norm=False),
        )

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    hidden,
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        ReLU(),
                    ),
                    bond_encoder=MLP([edge_features, hidden, hidden], norm=False, dropout=0.),)
                )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = data.edge_weight

        x = self.conv1(x, edge_index, edge_attr, edge_weight)
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_attr, edge_weight)
            x = residual(x, x_new)

        return x


class ZINC_GIN_Outer(torch.nn.Module):
    def __init__(self, in_features, edge_features, num_layers, hidden, num_classes, extra_dim, graph_pooling='mean'):
        super(ZINC_GIN_Outer, self).__init__()

        # map data.x to dim0
        self.atom_encoder = Linear(in_features, extra_dim[0]) if extra_dim[0] > 0 else None
        # map the extra feature to dim1
        self.extra_emb_layer = Linear(hidden, extra_dim[1]) if extra_dim[1] > 0 else None
        # merge 2 features
        self.merge_layer = Linear(sum(extra_dim), hidden) if min(extra_dim) > 0 else None

        if num_layers > 0:
            self.gnn = BaseGIN(hidden, edge_features, num_layers, hidden)
        else:
            self.gnn = GNN_Placeholder()

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        self.mlp = MLP([hidden, hidden, hidden, num_classes], norm=False, dropout=0.)

    def reset_parameters(self):
        if self.atom_encoder is not None:
            self.atom_encoder.reset_parameters()
        if self.extra_emb_layer is not None:
            self.extra_emb_layer.reset_parameters()
        if self.merge_layer is not None:
            self.merge_layer.reset_parameters()
        self.gnn.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data, intermediate_node_emb):
        if self.atom_encoder is not None and self.extra_emb_layer is not None:
            x = self.atom_encoder(data.x)
            extra = self.extra_emb_layer(intermediate_node_emb)
            input_data = torch.relu(
                self.merge_layer(
                    torch.cat([x, extra], dim=1)
                )
            )
        elif self.atom_encoder is not None and self.extra_emb_layer is None:
            input_data = self.atom_encoder(data.x)
        elif self.atom_encoder is None and self.extra_emb_layer is not None:
            input_data = self.extra_emb_layer(intermediate_node_emb)
        else:
            raise ValueError

        data.x = input_data
        h_node = self.gnn(data)
        h_graph = self.pool(h_node, data.batch)
        h_graph = self.mlp(h_graph)
        return h_graph


class ZINC_GIN_Inner(torch.nn.Module):
    def __init__(self, in_features, edge_features, num_layers, hidden, subgraph2node_aggr='add'):
        super(ZINC_GIN_Inner, self).__init__()
        self.gnn = BaseGIN(in_features, edge_features, num_layers, hidden)

        self.graph_pooling = subgraph2node_aggr

        if self.graph_pooling in ['center', 'add']:
            self.inner_pool = lambda x, subgraphs2nodes, *args: global_add_pool(x, subgraphs2nodes)
        elif self.graph_pooling == 'mean':
            self.inner_pool = \
                lambda x, subgraphs2nodes, node_mask: global_add_pool(x, subgraphs2nodes) / \
                                                      scatter(node_mask.to(torch.float).detach(),
                                                              subgraphs2nodes, dim=0, reduce="sum")[:, None]
        elif self.graph_pooling is None:
            self.inner_pool = None
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data):
        h_node = self.gnn(data)

        if self.inner_pool is not None:
            assert hasattr(data, 'node_mask') and hasattr(data, 'subgraphs2nodes')
            if data.node_mask.dtype == torch.float:
                h_node = h_node * data.node_mask[:, None]
            else:
                h_node = h_node[data.node_mask]
            h_node = self.inner_pool(h_node, data.subgraphs2nodes, data.node_mask)

        return h_node
