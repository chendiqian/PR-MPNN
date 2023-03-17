import torch


class IMLEScheme:
    def __init__(self, imle_sample_policy, sample_k):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        if self.imle_sample_policy == 'graph_topk':
            # local_logits: (Batch, Nmax, Nmax, ensemble)
            Nmax = local_logits.shape[1]
            if self.sample_k >= Nmax:
                return local_logits.new_ones(local_logits.shape)

            thresh = torch.topk(local_logits, self.sample_k, dim=2, largest=True, sorted=True).values[:, :, -1, :][:, :, None, :]
            mask = (local_logits >= thresh).to(torch.float)
            return mask, None
        if self.imle_sample_policy == 'global_topk':
            Batch, Nmax, _, ensemble = local_logits.shape
            if self.sample_k >= Nmax ** 2:
                return local_logits.new_ones(local_logits.shape)

            local_logits = local_logits.reshape(Batch, -1, ensemble)
            thresh = torch.topk(local_logits, self.sample_k, dim=1, largest=True,
                                sorted=True).values[:, -1, :][:, None, :]
            mask = (local_logits >= thresh).to(torch.float)
            mask = mask.reshape(Batch, Nmax, Nmax, ensemble)
            return mask, None
        else:
            raise NotImplementedError
