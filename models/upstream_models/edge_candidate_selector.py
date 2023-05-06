from typing import Union

import torch
from torch_geometric.data import Data, Batch

from models.nn_modules import MLP


class EdgeSelector(torch.nn.Module):
    def __init__(self,
                 encoder,
                 in_dim,
                 hid_size,
                 mlp_layer,
                 dropout=0.,
                 ensemble=1,
                 use_bn=False):
        super(EdgeSelector, self).__init__()

        self.atom_encoder = encoder
        self.mlp = MLP([in_dim * 2] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       batch_norm=use_bn, dropout=dropout)

    def forward(self, data: Union[Data, Batch]):
        assert hasattr(data, 'edge_candidate') and hasattr(data, 'num_edge_candidate')
        x = self.atom_encoder(data)
        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device), torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]
        edge_candidates = torch.hstack([x[edge_candidate_idx[:, 0]], x[edge_candidate_idx[:, 1]]])

        edge_candidates = self.mlp(edge_candidates)

        return edge_candidates, edge_candidate_idx

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.mlp.reset_parameters()
