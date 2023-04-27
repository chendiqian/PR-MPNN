import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.deterministic_scheme import rewire_global_undirected, rewire_global_directed, rewire_global_semi
from simple.simple import Layer
from data.data_utils import self_defined_softmax

LARGE_NUMBER = 1.e10

def logsigmoid(x):
    return -F.softplus(-x) + 1.e-7


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self,
                 k,
                 device,
                 policy,
                 val_ensemble=1,
                 train_ensemble=1,
                 logits_activation=None):
        super(EdgeSIMPLEBatched, self).__init__()
        self.k = k
        self.device = device
        self.policy = policy
        self.layer_configs = dict()
        self.adj = None  # for potential usage
        assert val_ensemble > 0 and train_ensemble > 1
        self.val_ensemble = val_ensemble
        self.train_ensemble = train_ensemble
        self.logits_activation = logits_activation

    def forward(self, scores, times_sampled=None):
        if times_sampled is None:
            times_sampled = self.train_ensemble

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
        elif self.policy == 'global_topk_semi':
            triu_idx = np.triu_indices(Nmax, k=1)
            uniques_num_edges = self.adj[:, triu_idx[0], triu_idx[1], :].sum(dim=(1, 2))
            max_num_edges = uniques_num_edges.max().long().item()
            target_size = (Nmax * (Nmax - 1)) // 2
            if self.k > target_size - max_num_edges:
                raise ValueError(f"k = {self.k} too large!")
            local_k = self.k
            scores = scores + scores.transpose(1, 2)
            scores = scores - self.adj * LARGE_NUMBER   # do not sample existing edges
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

        #default logits activation is none
        if self.logits_activation == 'None' or self.logits_activation is None:
            pass
        elif self.logits_activation == 'logsoftmax':
            # todo: it is bad heuristic to detect the padding
            masks = (flat_scores.detach() > - LARGE_NUMBER / 2).float()
            flat_scores = torch.vmap(self_defined_softmax, in_dims=0, out_dims=0)(flat_scores, masks)
            flat_scores = torch.log(flat_scores + 1 / LARGE_NUMBER)
        elif self.logits_activation == 'logsigmoid':
            # todo: sigmoid is not good, it makes large scores too similar, i.e. close to 1.
            flat_scores = logsigmoid(flat_scores)
        else:
            raise NotImplementedError

        if times_sampled == 1:
            # this only samples once
            samples, marginals = layer(flat_scores, local_k)
        else:
            # we potentially need to sample multiple times
            marginals = layer.log_pr(flat_scores).exp().permute(1, 0)
            # (times_sampled) x (B x E) x (N x N)
            samples = torch.stack([layer.sample(flat_scores, local_k) for _ in range(times_sampled)], dim=0)
            samples = (samples - marginals[None]).detach() + marginals[None]

        # unpadding
        samples = samples[..., :target_size]
        marginals = marginals[:, :target_size]

        new_marginals = None
        if self.policy == 'global_topk_directed':
            new_mask = samples.reshape(times_sampled, bsz, ensemble, Nmax, Nmax).permute((0, 1, 3, 4, 2))
            new_marginals = marginals.reshape(bsz, ensemble, Nmax, Nmax).permute((0, 2, 3, 1))
        elif self.policy == 'global_topk_undirected':
            samples = samples.reshape(times_sampled, bsz, ensemble, -1).permute((0, 1, 3, 2))
            new_mask = scores.new_zeros((times_sampled,) + scores.shape)
            new_mask[:, :, triu_idx[0], triu_idx[1], :] = samples
            new_mask = new_mask + new_mask.transpose(2, 3)
            marginals = marginals.reshape(bsz, ensemble, -1).permute((0, 2, 1))
            new_marginals = scores.new_zeros(scores.shape)
            new_marginals[:, triu_idx[0], triu_idx[1], :] = marginals
            new_marginals = new_marginals + new_marginals.transpose(1, 2)
        elif self.policy == 'global_topk_semi':
            samples = samples.reshape(times_sampled, bsz, ensemble, -1).permute((0, 1, 3, 2))
            new_mask = scores.new_zeros((times_sampled,) + scores.shape)
            new_mask[:, :, triu_idx[0], triu_idx[1], :] = samples
            new_mask = new_mask + new_mask.transpose(2, 3) + self.adj[None]
            marginals = marginals.reshape(bsz, ensemble, -1).permute((0, 2, 1))
            new_marginals = scores.new_zeros(scores.shape)
            new_marginals[:, triu_idx[0], triu_idx[1], :] = marginals
            new_marginals = new_marginals + new_marginals.transpose(1, 2)
        else:
            raise ValueError
        return new_mask, new_marginals

    @torch.no_grad()
    def validation(self, scores):
        """
        during the inference we need to margin-out the stochasticity
        thus we do top-k once or sample multiple times

        Args:
            scores: shape B x N x N x E

        Returns:
            mask: shape B x N x N x (E x VE)

        """
        if self.val_ensemble == 1:
            _, marginals = self.forward(scores, times_sampled=1)

            # do deterministic top-k
            if self.policy == 'global_topk_directed':
                mask = rewire_global_directed(scores, self.k)
            elif self.policy == 'global_topk_undirected':
                # make symmetric
                scores = scores + scores.transpose(1, 2)
                mask = rewire_global_undirected(scores, self.k)
            elif self.policy == 'global_topk_semi':
                scores = scores + scores.transpose(1, 2)
                mask = rewire_global_semi(scores, self.k, self.adj)
            else:
                raise NotImplementedError
            return mask[None], marginals
        else:
            return self.forward(scores, times_sampled=self.val_ensemble)
