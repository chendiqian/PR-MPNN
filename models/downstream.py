from typing import List
import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MLP
from torch_geometric.data import Data
from models.my_convs import BaseGINE


class GNN_Duo(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 num_candidates,
                 num_layers,
                 hidden,
                 num_classes,
                 use_bn,
                 dropout,
                 residual,
                 mlp_layers_intragraph,
                 mlp_layers_intergraph,
                 graph_pooling,
                 inter_graph_pooling):
        super(GNN_Duo, self).__init__()

        self.encoder = encoder

        self.gnn = BaseGINE(num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        self.candid_gnns = torch.nn.ModuleList(
            [BaseGINE(num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
             for _ in range(num_candidates)]
        )

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        if graph_pooling == "sum":
            self.pool = global_add_pool
            self.candid_pool = self.pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
            self.candid_pool = self.pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
            self.candid_pool = self.pool
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
            self.candid_pool = self.pool
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

        if mlp_layers_intragraph == 0:
            self.mlp = lambda x: x
        else:
            self.mlp = MLP([hidden] + [hidden] * mlp_layers_intragraph, dropout=0., plain_last=False)
        if mlp_layers_intragraph == 0:
            self.candid_mlps = lambda x: x
        else:
            self.candid_mlps = torch.nn.ModuleList([MLP([hidden] + [hidden] * mlp_layers_intragraph,
                                                        dropout=0.,
                                                        plain_last=False) for _ in range(num_candidates)])

        # inter-graph pooling
        self.inter_graph_pooling = inter_graph_pooling
        if inter_graph_pooling == 'mean':
            self.final_mlp = MLP([hidden] * max(mlp_layers_intergraph, 1) + [num_classes], dropout=0.)
        elif inter_graph_pooling == 'cat':
            self.final_mlp = MLP([-1] + [hidden] * max(mlp_layers_intergraph - 1, 0) + [num_classes], dropout=0.)
        else:
            raise NotImplementedError

    def forward(self, original_data: Data, rewired_data: List[Data]):
        if self.encoder is not None:
            if original_data is not None:
                original_data.x = self.encoder(original_data)
            for c in rewired_data:
                c.x = self.encoder(c)

        if original_data is not None:
            h_node_org = self.gnn(original_data.x,
                                  original_data.edge_index,
                                  original_data.edge_attr,
                                  original_data.edge_weight)
            h_graph_org = self.pool(h_node_org, getattr(original_data, self.graph_pool_idx))
            h_graph_org = self.mlp(h_graph_org)
        else:
            h_graph_org = None

        h_graphs = []
        for i, c in enumerate(rewired_data):
            gnn = self.candid_gnns[i] if isinstance(self.candid_gnns, torch.nn.ModuleList) else self.candid_gnns
            h_node = gnn(c.x, c.edge_index, c.edge_attr, c.edge_weight)
            pool = self.candid_pool[i] if isinstance(self.candid_pool, torch.nn.ModuleList) else self.candid_pool
            h_graph = pool(h_node, getattr(c, self.graph_pool_idx))
            mlp = self.candid_mlps[i] if isinstance(self.candid_mlps, torch.nn.ModuleList) else self.candid_mlps
            h_graphs.append(mlp(h_graph))

        if h_graph_org is not None:
            assert not h_graphs[0].shape[0] % h_graph_org.shape[0]
            h_graph_org = h_graph_org.repeat((h_graphs[0].shape[0] // h_graph_org.shape[0], 1))
            h_graphs = [h_graph_org] + h_graphs

        # inter graph pooling
        if self.inter_graph_pooling == 'mean':
            h_graphs = torch.stack(h_graphs, dim=0).mean(0)
        elif self.inter_graph_pooling == 'cat':
            h_graphs = torch.cat(h_graphs, dim=1)
        else:
            raise ValueError

        h_graphs = self.final_mlp(h_graphs)
        return h_graphs
