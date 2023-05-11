from typing import Union

import torch
from torch_geometric.data import Data, Batch

from models.nn_modules import MLP

from models.my_convs import BaseGINE, GNN_Placeholder


class EdgeSelector(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 in_dim,
                 hid_size,
                 gnn_layer,
                 mlp_layer,
                 use_deletion_head,
                 dropout=0.,
                 ensemble=1,
                 use_bn=False):
        super(EdgeSelector, self).__init__()

        if gnn_layer == 0:
            self.gnn = GNN_Placeholder()
        else:
            self.gnn = BaseGINE(in_dim, gnn_layer, hid_size, hid_size, edge_encoder, True, dropout, True)
            in_dim = hid_size

        self.atom_encoder = encoder
        self.mlp1 = MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       batch_norm=use_bn, dropout=dropout)

        self.use_deletion_head = use_deletion_head
        if use_deletion_head:
            self.mlp2 = MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                           batch_norm=use_bn, dropout=dropout)

    def forward(self, data: Union[Data, Batch]):
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)
        x = self.gnn(x, data.edge_index, data.edge_attr)

        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device), torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        edge_candidates = torch.hstack([x[edge_candidate_idx[:, 0]], x[edge_candidate_idx[:, 1]]])

        select_edge_candidates = self.mlp1(edge_candidates)
        if self.use_deletion_head:
            delete_edge_candidates = self.mlp2(edge_candidates)
        else:
            delete_edge_candidates = None

        return select_edge_candidates, delete_edge_candidates, edge_candidate_idx

    def reset_parameters(self):
        raise NotImplementedError
