import torch
import numpy as np
from training.deterministic_scheme import (rewire_global_undirected,
                                           rewire_global_directed,
                                           rewire_global_semi,
                                           select_from_edge_candidates)

EPSILON = np.finfo(np.float32).tiny
LARGE_NUMBER = 1.e10

class GumbelSampler(torch.nn.Module):
    def __init__(self, k, tau=0.1, hard=True, policy=None):
        super(GumbelSampler, self).__init__()
        self.policy = policy
        self.k = k
        self.hard = hard
        self.tau = tau
        self.adj = None   # for potential usage

    def forward(self, scores, train_ensemble):
        if self.policy == 'global_topk_directed':
            bsz, Nmax, _, ensemble = scores.shape
            local_k = min(self.k, Nmax ** 2)
            flat_scores = scores.permute((0, 3, 1, 2)).reshape(bsz * ensemble, Nmax ** 2)
        elif self.policy == 'global_topk_undirected':
            bsz, Nmax, _, ensemble = scores.shape
            local_k = min(self.k, (Nmax * (Nmax - 1)) // 2)
            triu_idx = np.triu_indices(Nmax, k=1)
            # make symmetric
            scores = scores + scores.transpose(1, 2)
            flat_scores = scores[:, triu_idx[0], triu_idx[1], :].permute((0, 2, 1)).reshape(bsz * ensemble, -1)
        elif self.policy == 'global_topk_semi':
            bsz, Nmax, _, ensemble = scores.shape
            triu_idx = np.triu_indices(Nmax, k=1)
            uniques_num_edges = self.adj[:, triu_idx[0], triu_idx[1], :].sum(dim=(1, 2))
            max_num_edges = uniques_num_edges.max().long().item()
            target_size = (Nmax * (Nmax - 1)) // 2 - max_num_edges
            if self.k > target_size:
                raise ValueError(f"k = {self.k} too large!")
            local_k = self.k
            scores = scores + scores.transpose(1, 2)
            scores = scores - self.adj * LARGE_NUMBER  # do not sample existing edges
            flat_scores = scores[:, triu_idx[0], triu_idx[1], :].permute((0, 2, 1)).reshape(bsz * ensemble, -1)
        elif self.policy == 'edge_candid':
            bsz, Nmax, ensemble = scores.shape
            flat_scores = scores.permute((0, 2, 1)).reshape(bsz * ensemble, Nmax)
            local_k = min(self.k, Nmax)
        else:
            raise NotImplementedError

        # sample several times with
        flat_scores = flat_scores.repeat(train_ensemble, 1)

        m = torch.distributions.gumbel.Gumbel(flat_scores.new_zeros(flat_scores.shape),
                                              flat_scores.new_ones(flat_scores.shape))
        g = m.sample()
        flat_scores = flat_scores + g

        # continuous top k
        khot = flat_scores.new_zeros(flat_scores.shape)
        onehot_approx = flat_scores.new_zeros(flat_scores.shape)
        for i in range(local_k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=flat_scores.device))
            flat_scores = flat_scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(flat_scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = khot.new_zeros(khot.shape)
            val, ind = torch.topk(khot, local_k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        if self.policy == 'global_topk_directed':
            new_mask = res.reshape(train_ensemble, bsz, ensemble, Nmax, Nmax).permute((0, 1, 3, 4, 2))
        elif self.policy == 'global_topk_undirected':
            res = res.reshape(train_ensemble, bsz, ensemble, -1).permute((0, 1, 3, 2))
            new_mask = scores.new_zeros((train_ensemble,) + scores.shape)
            new_mask[:, :, triu_idx[0], triu_idx[1], :] = res
            new_mask = new_mask + new_mask.transpose(2, 3)
        elif self.policy == 'global_topk_semi':
            res = res.reshape(train_ensemble, bsz, ensemble, -1).permute((0, 1, 3, 2))
            new_mask = scores.new_zeros((train_ensemble,) + scores.shape)
            new_mask[:, :, triu_idx[0], triu_idx[1], :] = res
            new_mask = new_mask + new_mask.transpose(2, 3) + self.adj[None]
        elif self.policy == 'edge_candid':
            new_mask = res.reshape(train_ensemble, bsz, ensemble, Nmax).permute((0, 1, 3, 2))
        else:
            raise NotImplementedError
        return new_mask, None

    @torch.no_grad()
    def validation(self, scores, val_ensemble):
        if val_ensemble == 1:
            if self.policy == 'global_topk_directed':
                mask = rewire_global_directed(scores, self.k)
            elif self.policy == 'global_topk_undirected':
                # make symmetric
                scores = scores + scores.transpose(1, 2)
                mask = rewire_global_undirected(scores, self.k)
            elif self.policy == 'global_topk_semi':
                scores = scores + scores.transpose(1, 2)
                mask = rewire_global_semi(scores, self.k, self.adj)
            elif self.policy == 'edge_candid':
                mask = select_from_edge_candidates(scores, self.k)
            else:
                raise NotImplementedError

            return mask[None], None
        else:
            return self.forward(scores, val_ensemble)
