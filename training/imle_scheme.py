import torch
from training.deterministic_scheme import rewire_global_directed, rewire_global_undirected


class IMLEScheme:
    def __init__(self, imle_sample_policy, sample_k):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        if self.imle_sample_policy == 'global_topk_directed':
            mask = rewire_global_directed(local_logits, self.sample_k)
            return mask
        elif self.imle_sample_policy == 'global_topk_undirected':
            # make symmetric
            local_logits = local_logits + local_logits.transpose(1, 2)
            mask = rewire_global_undirected(local_logits, self.sample_k)
            return mask
        else:
            raise NotImplementedError
