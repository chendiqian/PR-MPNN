import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.deterministic_scheme import rewire_global_undirected, rewire_global_directed
from simple.simple import Layer

LARGE_NUMBER = 1.e10

def logsigmoid(x):
    return -F.softplus(-x) + 1.e-7


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self, k, device, policy):
        super(EdgeSIMPLEBatched, self).__init__()
        self.k = k
        self.device = device
        self.policy = policy
        self.layer_configs = dict()

    def forward(self, scores):
        bsz, Nmax, _, ensemble = scores.shape
        if self.policy == 'global_topk_directed':
            target_size = Nmax ** 2
            local_k = min(self.k, target_size)
            flat_scores = scores.permute((0, 3, 1, 2)).reshape(bsz * ensemble, target_size)
        elif self.policy == 'global_topk_undirected':
            target_size = (Nmax * (Nmax - 1)) // 2
            local_k = min(self.k, target_size)
            triu_idx = np.triu_indices(Nmax, k=1)
            scores = scores + scores.transpose(1, 2)
            flat_scores = scores[:, triu_idx[0], triu_idx[1], :].permute((0, 2, 1)).reshape(bsz * ensemble, -1)
        else:
            raise NotImplementedError

        N = 2 ** math.ceil(math.log2(target_size))
        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, self.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        flat_scores = torch.cat(
            [flat_scores,
             torch.full((flat_scores.shape[0], N - flat_scores.shape[1]),
                        fill_value=-LARGE_NUMBER,
                        dtype=flat_scores.dtype,
                        device=flat_scores.device)],
            dim=1)

        flat_scores = logsigmoid(flat_scores)
        samples = layer(flat_scores, local_k)
        # unpadding
        samples = samples[:, :target_size]
        if self.policy == 'global_topk_directed':
            new_mask = samples.reshape(bsz, ensemble, Nmax, Nmax).permute((0, 2, 3, 1))
        elif self.policy == 'global_topk_undirected':
            samples = samples.reshape(bsz, ensemble, -1).permute((0, 2, 1))
            new_mask = scores.new_zeros(scores.shape)
            new_mask[:, triu_idx[0], triu_idx[1], :] = samples
            new_mask = new_mask + new_mask.transpose(1, 2)
        return new_mask, None

    @torch.no_grad()
    def validation(self, scores):
        if self.policy == 'global_topk_directed':
            mask = rewire_global_directed(scores, self.k)
        elif self.policy == 'global_topk_undirected':
            # make symmetric
            scores = scores + scores.transpose(1, 2)
            mask = rewire_global_undirected(scores, self.k)
        else:
            raise NotImplementedError
        return mask
