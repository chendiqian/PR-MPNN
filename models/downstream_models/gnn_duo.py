import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set

from models.my_convs import BaseGIN, BaseGINE
from models.nn_modules import MLP

from data.utils.datatype_utils import DuoDataStructure


class GNN_Duo(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
                 share_weights,
                 include_org,
                 num_candidates,
                 in_features,
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

        if base_gnn == 'gin':
            model_class = BaseGIN
        elif base_gnn == 'gine':
            model_class = BaseGINE
        else:
            raise NotImplementedError

        self.gnn = model_class(in_features, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        if share_weights:
            self.candid_gnns = self.gnn
        else:
            self.candid_gnns = torch.nn.ModuleList([model_class(in_features, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder) for _ in range(num_candidates)])

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        if graph_pooling == "sum":
            self.pool = global_add_pool
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
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=6)
            if share_weights:
                self.candid_pool = self.pool
            else:
                self.candid_pool = torch.nn.ModuleList([Set2Set(hidden, processing_steps=6) for _ in range(num_candidates)])
        else:
            raise NotImplementedError

        self.inter_graph_pooling = inter_graph_pooling
        in_mlp = hidden if graph_pooling != 'set2set' else hidden * 2
        self.mlp = MLP([in_mlp] + [hidden] * mlp_layers_intragraph, dropout=0.)
        if share_weights:
            self.candid_mlps = self.mlp
        else:
            self.candid_mlps = torch.nn.ModuleList([MLP([in_mlp] + [hidden] * mlp_layers_intragraph, dropout=0.) for _ in range(num_candidates)])

        # inter-graph pooling
        if inter_graph_pooling == 'mean':
            self.final_mlp = MLP([hidden] * max(mlp_layers_intergraph, 1) + [num_classes], dropout=0.)
        elif inter_graph_pooling == 'cat':
            self.final_mlp = MLP([hidden * (num_candidates + int(include_org))] + [hidden] * max(mlp_layers_intergraph - 1, 0) + [num_classes], dropout=0.)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data: DuoDataStructure):
        org = data.org
        candidates = data.candidates

        if self.encoder is not None:
            if org is not None:
                org.x = self.encoder(org)
            for c in candidates:
                c.x = self.encoder(c)

        if org is not None:
            h_node_org = self.gnn(org.x, org.edge_index, org.edge_attr, org.edge_weight)
            h_graph_org = self.pool(h_node_org, getattr(org, self.graph_pool_idx))
            h_graph_org = self.mlp(h_graph_org)
        else:
            h_graph_org = None

        h_graphs = []
        for i, c in enumerate(candidates):
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
