import math

import torch
import torch.nn as nn

from simple.simple import Layer

LARGE_NUMBER = 1.e10


def select_from_edge_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self):
        super(EdgeSIMPLEBatched, self).__init__()
        self.layer_configs = dict()

    def forward(self, scores, k, times_sampled):
        bsz, Nmax, ensemble = scores.shape
        flat_scores = scores.permute((0, 2, 1)).reshape(bsz * ensemble, Nmax)
        target_size = Nmax
        local_k = min(k, Nmax)
        N = 2 ** math.ceil(math.log2(target_size))

        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, scores.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        flat_scores = torch.cat(
            [flat_scores,
             torch.full((flat_scores.shape[0], N - flat_scores.shape[1]),
                        fill_value=-LARGE_NUMBER,
                        dtype=flat_scores.dtype,
                        device=flat_scores.device)],
            dim=1)

        # we potentially need to sample multiple times
        marginals = layer.log_pr(flat_scores).exp().permute(1, 0)
        # (times_sampled) x (B x E) x (N x N)
        samples = layer.sample(flat_scores, local_k, times_sampled)
        samples = (samples - marginals[None]).detach() + marginals[None]

        # unpadding
        samples = samples[..., :target_size]
        marginals = marginals[:, :target_size]

        # VE x (B x E) x Nmax -> VE x B x Nmax x E
        new_mask = samples.reshape(times_sampled, bsz, ensemble, Nmax).permute((0, 1, 3, 2))
        # (B x E) x Nmax -> B x Nmax x E
        new_marginals = marginals.reshape(bsz, ensemble, Nmax).permute((0, 2, 1))

        return new_mask, new_marginals

    @torch.no_grad()
    def validation(self, scores, k, times_sampled):
        """
        during the inference we need to margin-out the stochasticity
        thus we do top-k once or sample multiple times

        Args:
            scores: shape B x N x N x E
            k: int, sample k
            times_sampled: int, sample times from each ensemble

        Returns:
            mask: shape B x N x N x (E x VE)

        """
        if times_sampled == 1:
            _, marginals = self.forward(scores, k, 1)
            # do deterministic top-k
            mask = select_from_edge_candidates(scores, k)
            return mask[None], marginals
        else:
            return self.forward(scores, k, times_sampled)
