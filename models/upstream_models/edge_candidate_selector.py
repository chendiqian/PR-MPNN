from typing import Union

import torch
from torch_geometric.data import Data, Batch

from models.nn_modules import MLP
from models.my_convs import BaseGINE, BasePNA, GNN_Placeholder


class EdgeSelector(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 in_dim,
                 hid_size,
                 gnn_layer,
                 mlp_layer,
                 use_deletion_head,
                 directed_sampling=False,
                 dropout=0.,
                 ensemble=1,
                 use_bn=False,
                 deg_hist=None,
                 upstream_model=None):
        super(EdgeSelector, self).__init__()

        if isinstance(mlp_layer, int):
            self.use_bilinear = False
        elif isinstance(mlp_layer, str) and mlp_layer == 'bilinear':
            self.use_bilinear = True
        else:
            raise ValueError(f'{mlp_layer} not supported as mlp_layer arg')

        if gnn_layer == 0:
            self.gnn = GNN_Placeholder()
        else:
            if upstream_model == 'pna':
                assert deg_hist is not None
                self.gnn = BasePNA(in_dim, gnn_layer, hid_size, hid_size, True, dropout, True, deg_hist, edge_encoder)
            else:
                # default to GIN
                self.gnn = BaseGINE(in_dim, gnn_layer, hid_size, hid_size, True, dropout, True, edge_encoder)
            in_dim = hid_size

        self.atom_encoder = encoder
        self.projector1 = torch.nn.Bilinear(in_dim, in_dim, ensemble, bias=True) if self.use_bilinear \
            else MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                     batch_norm=use_bn, dropout=dropout)

        self.use_deletion_head = use_deletion_head
        self.directed_sampling =directed_sampling
        if use_deletion_head:
            self.projector2 = torch.nn.Bilinear(in_dim, in_dim, ensemble, bias=True) if self.use_bilinear \
                else MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                         batch_norm=use_bn, dropout=dropout)

    def forward(self, data: Union[Data, Batch]):
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)
        x = self.gnn(x, data.edge_index, data.edge_attr)

        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device), torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        if self.use_bilinear:
            select_edge_candidates = self.projector1(x[edge_candidate_idx[:, 0]], x[edge_candidate_idx[:, 1]])
        else:
            edge_candidates = torch.hstack([x[edge_candidate_idx[:, 0]], x[edge_candidate_idx[:, 1]]])
            select_edge_candidates = self.projector1(edge_candidates)
        if self.use_deletion_head:
            edge_index = data.edge_index
            if not self.directed_sampling:
                edge_index = edge_index[:, edge_index[0] <= edge_index[1]]  # self loops included
            if self.use_bilinear:
                delete_edge_candidates = self.projector2(x[edge_index[0]], x[edge_index[1]])
            else:
                cur_edges = torch.hstack([x[edge_index[0]], x[edge_index[1]]])
                delete_edge_candidates = self.projector2(cur_edges)
        else:
            delete_edge_candidates = None

        return select_edge_candidates, delete_edge_candidates, edge_candidate_idx

    def reset_parameters(self):
        raise NotImplementedError
