from typing import Callable

import torch


class HybridModel(torch.nn.Module):
    def __init__(self,
                 upstream: torch.nn.Module,
                 downstream: torch.nn.Module,
                 rewiring: Callable):
        super(HybridModel, self).__init__()
        self.upstream = upstream
        self.downstream = downstream
        self.rewiring = rewiring

    def forward(self, data):
        select_edge_candidates, delete_edge_candidates, edge_candidate_idx = self.upstream(data)
        new_data, auxloss = self.rewiring(data,
                                          select_edge_candidates,
                                          delete_edge_candidates,
                                          edge_candidate_idx)

        pred = self.downstream(data, new_data)
        return pred, new_data, auxloss
