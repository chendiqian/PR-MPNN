import torch
import numpy as np

EPSILON = np.finfo(np.float32).tiny

class GumbelSampler(torch.nn.Module):
    def __init__(self, k, tau=0.1, hard=True):
        super(GumbelSampler, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        bsz, Nmax, _, ensemble = scores.shape
        local_k = min(self.k, Nmax ** 2)

        scores = scores.permute((0, 3, 1, 2)).reshape(bsz * ensemble, Nmax ** 2)

        m = torch.distributions.gumbel.Gumbel(scores.new_zeros(scores.shape), scores.new_ones(scores.shape))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = scores.new_zeros(scores.shape)
        onehot_approx = scores.new_zeros(scores.shape)
        for i in range(local_k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = khot.new_zeros(khot.shape)
            val, ind = torch.topk(khot, local_k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        res = res.reshape(bsz, ensemble, Nmax, Nmax).permute((0, 2, 3, 1))
        return res, None

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
