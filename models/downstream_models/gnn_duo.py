import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, Set2Set

from models.my_convs import BaseGIN, BaseGINE, BasePNA, BaseGCN
from models.downstream_models.qm9_gnn import QM9_Net
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
                 deg_hist,
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
        self.base_gnn = base_gnn

        if base_gnn == 'gin':
            self.gnn = BaseGIN(hidden, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        elif base_gnn == 'gine':
            self.gnn = BaseGINE(hidden, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        elif base_gnn == 'gcn':
            self.gnn = BaseGCN(hidden, num_layers, hidden, hidden, use_bn, dropout, residual)
        elif base_gnn == 'qm9gine':
            self.encoder = None  # no encoder, qm9 model has one
            graph_pooling, qm9_graph_pooling = None, graph_pooling
            self.gnn = QM9_Net(encoder, 'gine', edge_encoder, hidden, hidden, num_layers, dropout, qm9_graph_pooling)
        elif base_gnn == 'qm9gin':
            self.encoder = None  # no encoder, qm9 model has one
            graph_pooling, qm9_graph_pooling = None, graph_pooling
            self.gnn = QM9_Net(encoder, 'gin', edge_encoder, hidden, hidden, num_layers, dropout, qm9_graph_pooling)
        elif base_gnn == 'pna':
            assert deg_hist is not None
            self.gnn = BasePNA(hidden, num_layers, hidden, hidden, use_bn,
                               dropout, residual, deg_hist, edge_encoder)
        else:
            raise NotImplementedError

        if share_weights:
            self.candid_gnns = self.gnn
        else:
            if base_gnn == 'gin':
                self.candid_gnns = torch.nn.ModuleList(
                    [BaseGIN(hidden, num_layers, hidden, hidden,
                             use_bn, dropout, residual, edge_encoder)
                     for _ in range(num_candidates)]
                )
            elif base_gnn == 'gine':
                self.candid_gnns = torch.nn.ModuleList(
                    [BaseGINE(hidden, num_layers, hidden, hidden,
                              use_bn, dropout, residual, edge_encoder)
                     for _ in range(num_candidates)]
                )
            elif base_gnn == 'gcn':
                self.candid_gnns = torch.nn.ModuleList(
                    [BaseGCN(hidden, num_layers, hidden, hidden,
                              use_bn, dropout, residual)
                     for _ in range(num_candidates)]
                )
            elif base_gnn == 'qm9gine':
                self.candid_gnns = torch.nn.ModuleList(
                    [QM9_Net(encoder, 'gine', edge_encoder, hidden, hidden,
                             num_layers, dropout, qm9_graph_pooling)
                     for _ in range(num_candidates)]
                )
            elif base_gnn == 'qm9gin':
                self.candid_gnns = torch.nn.ModuleList(
                    [QM9_Net(encoder, 'gin', edge_encoder, hidden, hidden,
                             num_layers, dropout, qm9_graph_pooling)
                     for _ in range(num_candidates)]
                )
            elif base_gnn == 'pna':
                assert deg_hist is not None
                self.candid_gnns = torch.nn.ModuleList(
                    [BasePNA(hidden, num_layers, hidden, hidden, use_bn,
                                   dropout, residual, deg_hist, edge_encoder)
                     for _ in range(num_candidates)])
            else:
                raise NotImplementedError

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
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=6)
            if share_weights:
                self.candid_pool = self.pool
            else:
                self.candid_pool = torch.nn.ModuleList([Set2Set(hidden, processing_steps=6) for _ in range(num_candidates)])
        else:
            raise NotImplementedError

        in_mlp = hidden if graph_pooling != 'set2set' else hidden * 2
        if mlp_layers_intragraph == 0:
            self.mlp = lambda x: x
        else:
            self.mlp = MLP([in_mlp] + [hidden] * mlp_layers_intragraph, dropout=0.)
        if share_weights:
            self.candid_mlps = self.mlp
        else:
            if mlp_layers_intragraph == 0:
                self.candid_mlps = lambda x: x
            else:
                self.candid_mlps = torch.nn.ModuleList([MLP([in_mlp] + [hidden] * mlp_layers_intragraph, dropout=0.) for _ in range(num_candidates)])

        # inter-graph pooling
        self.inter_graph_pooling = inter_graph_pooling
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
            if self.base_gnn.startswith('qm9'):
                h_node_org = self.gnn(org)
            else:
                h_node_org = self.gnn(org.x, org.edge_index, org.edge_attr, org.edge_weight)
            h_graph_org = self.pool(h_node_org, getattr(org, self.graph_pool_idx))
            h_graph_org = self.mlp(h_graph_org)
        else:
            h_graph_org = None

        h_graphs = []
        for i, c in enumerate(candidates):
            gnn = self.candid_gnns[i] if isinstance(self.candid_gnns, torch.nn.ModuleList) else self.candid_gnns
            if self.base_gnn.startswith('qm9'):
                h_node = gnn(c)
            else:
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
