import torch
from training.deterministic_scheme import (rewire_global_directed,
                                           rewire_global_undirected,
                                           select_from_edge_candidates)

LARGE_NUMBER = 1.e10

class IMLEScheme:
    def __init__(self, imle_sample_policy, sample_k):
        self.policy = imle_sample_policy
        self.k = sample_k
        self.adj = None  # for potential usage

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        if self.policy == 'global_topk_directed':
            mask = rewire_global_directed(local_logits, self.k, self.adj)
        elif self.policy == 'global_topk_undirected':
            mask = rewire_global_undirected(local_logits, self.k, self.adj)
        elif self.policy == 'edge_candid':
            mask = select_from_edge_candidates(local_logits, self.k)
        else:
            raise NotImplementedError

        return mask, None
