import torch


class IMLEScheme:
    def __init__(self, imle_sample_policy, ptr, graphs, real_node_node_mask, sample_k):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self._ptr = ptr
        self._graphs = graphs
        self._real_node_node_mask = real_node_node_mask

    @property
    def ptr(self):
        return self._ptr

    @ptr.setter
    def ptr(self, value):
        assert value is None or value.dtype == torch.long
        self._ptr = value

    @ptr.deleter
    def ptr(self):
        del self._ptr

    @property
    def graphs(self):
        return self._graphs

    @graphs.setter
    def graphs(self, new_graphs):
        self._graphs = new_graphs

    @graphs.deleter
    def graphs(self):
        del self._graphs

    @property
    def seed_node_mask(self):
        return self._seed_node_mask

    @seed_node_mask.setter
    def seed_node_mask(self, node_mask):
        assert node_mask.dtype in [torch.bool, torch.long, torch.int64]
        self._seed_node_mask = node_mask

    @seed_node_mask.deleter
    def seed_node_mask(self):
        del self._seed_node_mask

    @property
    def real_node_node_mask(self):
        return self._real_node_node_mask

    @real_node_node_mask.setter
    def real_node_node_mask(self, value):
        assert value is None or value.dtype == torch.bool
        self._real_node_node_mask = value

    @real_node_node_mask.deleter
    def real_node_node_mask(self):
        del self._real_node_node_mask

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        if self.imle_sample_policy == 'graph_topk':
            # local_logits: (Batch, Nmax, Nmax, ensemble)
            Nmax = local_logits.shape[1]
            if self.sample_k >= Nmax:
                return local_logits.new_ones(local_logits.shape)

            local_logits[~self.real_node_node_mask] -= 1.e10
            thresh = torch.topk(local_logits, self.sample_k, dim=2, largest=True, sorted=True).values[:, :, -1, :][:, :, None, :]
            mask = (local_logits >= thresh).to(torch.float)
            return mask, None
        else:
            raise NotImplementedError
