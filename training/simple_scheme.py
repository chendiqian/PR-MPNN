import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple.simple import Layer

LARGE_NUMBER = 1.e10

def logsigmoid(x):
    return -F.softplus(-x) + 1.e-7


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self, k, device):
        super(EdgeSIMPLEBatched, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()

    def forward(self, scores):
        bsz, Nmax, _, ensemble = scores.shape
        local_k = min(self.k, Nmax ** 2)

        scores = scores.permute((0, 3, 1, 2)).reshape(bsz * ensemble, Nmax ** 2)

        # need to create or load some pkl files, which takes some time, so I cache them here
        N = 2 ** math.ceil(math.log2(Nmax ** 2))
        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, self.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        scores = torch.cat([scores, torch.full((scores.shape[0],  N - scores.shape[1]), fill_value=-LARGE_NUMBER, dtype=scores.dtype, device=scores.device)], dim=1)

        scores = logsigmoid(scores)
        samples = layer(scores, local_k)
        # unpadding
        samples = samples[:, :Nmax**2]
        samples = samples.reshape(bsz, ensemble, Nmax, Nmax).permute((0, 2, 3, 1))
        return samples, None

    @torch.no_grad()
    def validation(self, scores):
        Batch, Nmax, _, ensemble = scores.shape
        if self.k >= Nmax ** 2:
            return scores.new_ones(scores.shape)

        scores = scores.reshape(Batch, -1, ensemble)
        thresh = torch.topk(scores, self.k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
        mask = (scores >= thresh).to(torch.float)
        mask = mask.reshape(Batch, Nmax, Nmax, ensemble)
        return mask, None