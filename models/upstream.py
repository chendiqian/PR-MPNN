from typing import Union

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MLP

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
                 directed_sampling=False,
                 dropout=0.,
                 ensemble=1,
                 use_bn=False):
        super(EdgeSelector, self).__init__()

        if gnn_layer == 0:
            self.gnn = GNN_Placeholder()
        else:
            self.gnn = BaseGINE(in_dim, gnn_layer, hid_size, hid_size, True, dropout, True, edge_encoder)
            in_dim = hid_size

        self.atom_encoder = encoder
        self.projector1 = MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                              norm='batch_norm' if use_bn else None, dropout=dropout)

        self.use_deletion_head = use_deletion_head
        self.directed_sampling = directed_sampling
        if use_deletion_head:
            self.projector2 = MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                                  norm='batch_norm' if use_bn else None, dropout=dropout)

    def forward(self, data: Union[Data, Batch]):
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)
        x = self.gnn(x, data.edge_index, data.edge_attr)

        edge_rel = data._slice_dict['x'].to(x.device)[:-1]
        edge_candidate_idx = data.edge_candidate + edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        edge_candidates = torch.hstack([x[edge_candidate_idx[:, 0]], x[edge_candidate_idx[:, 1]]])
        select_edge_candidates = self.projector1(edge_candidates)
        if self.use_deletion_head:
            edge_index = data.edge_index
            if not self.directed_sampling:
                edge_index = edge_index[:, edge_index[0] <= edge_index[1]]  # self loops included
            cur_edges = torch.hstack([x[edge_index[0]], x[edge_index[1]]])
            delete_edge_candidates = self.projector2(cur_edges)
        else:
            delete_edge_candidates = None

        return select_edge_candidates, delete_edge_candidates, edge_candidate_idx
