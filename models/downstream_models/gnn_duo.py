import torch

from models.my_convs import BaseGIN, BaseGINE, BasePNA, BaseGCN
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
                 num_layers,
                 hidden,
                 num_classes,
                 use_bn,
                 dropout,
                 residual,
                 mlp_layers_intragraph,
                 mlp_layers_intergraph,
                 inter_graph_pooling):
        super(GNN_Duo, self).__init__()

        self.encoder = encoder
        self.base_gnn = base_gnn

        if base_gnn == 'gin':
            self.gnn = BaseGIN(hidden, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        elif base_gnn == 'gine':
            self.gnn = BaseGINE(hidden, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
        elif base_gnn == 'gcn':
            self.gnn = BaseGCN(hidden, num_layers, hidden, hidden, use_bn, dropout, residual, edge_encoder)
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
                              use_bn, dropout, residual, edge_encoder)
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
        self.pool = lambda x, *args: x
        self.candid_pool = self.pool

        if mlp_layers_intragraph == 0:
            self.mlp = lambda x: x
        else:
            self.mlp = MLP([hidden] + [hidden] * mlp_layers_intragraph, dropout=0., activate_last=True)
        if share_weights:
            self.candid_mlps = self.mlp
        else:
            if mlp_layers_intragraph == 0:
                self.candid_mlps = lambda x: x
            else:
                self.candid_mlps = torch.nn.ModuleList([MLP([hidden] + [hidden] * mlp_layers_intragraph, dropout=0., activate_last=True) for _ in range(num_candidates)])

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
