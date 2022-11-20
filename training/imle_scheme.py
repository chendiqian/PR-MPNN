from math import ceil

import torch
from subgraph.or_based import get_or_optim_subgraphs
from subgraph.greedy_expand import greedy_grow_tree


class IMLEScheme:
    def __init__(self, imle_sample_policy, ptr, graphs, seed_node_mask, sample_k, ensemble=1):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self._ptr = ptr
        self._graphs = graphs
        self._seed_node_mask = seed_node_mask
        self.ensemble = ensemble

    @property
    def ptr(self):
        return self._ptr

    @ptr.setter
    def ptr(self, value):
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

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        local_logits = torch.split(local_logits, self.ptr, dim=0)

        sample_instance_idx = []
        if self.imle_sample_policy == 'KMaxNeighbors':
            for i, logit in enumerate(local_logits):
                logit = logit.reshape(self.graphs[i].num_nodes, self.graphs[i].num_nodes, self.ensemble)
                mask = get_or_optim_subgraphs(self.graphs[i].edge_index, logit, self.sample_k)
                mask.requires_grad = False
                sample_instance_idx.append(mask.reshape(self.graphs[i].num_nodes ** 2, self.ensemble))
        elif self.imle_sample_policy == 'greedy_neighbors':
            for i, logit in enumerate(local_logits):
                logit = logit.reshape(self.graphs[i].num_nodes, self.graphs[i].num_nodes, self.ensemble)
                mask = greedy_grow_tree(self.graphs[i], self.sample_k, logit)
                mask.requires_grad = False
                sample_instance_idx.append(mask.reshape(self.graphs[i].num_nodes ** 2, self.ensemble))
        elif self.imle_sample_policy == 'topk':
            for i, logit in enumerate(local_logits):
                if isinstance(self.sample_k, float):
                    k = int(ceil(self.sample_k * logit.shape[0]))
                elif isinstance(self.sample_k, int):
                    k = self.sample_k
                else:
                    raise TypeError

                if k >= logit.shape[0]:
                    sample_instance_idx.append(torch.ones(logit.shape, device=logit.device, dtype=torch.float))
                else:
                    thresh = torch.topk(logit, k, dim=0, largest=True, sorted=True).values[-1, :]
                    sample_instance_idx.append((logit >= thresh[None]).to(torch.float))
        else:
            raise NotImplementedError

        sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
        # seed node must be selected
        if self.seed_node_mask is not None:
            sample_instance_idx[self.seed_node_mask] = 1.
        sample_instance_idx.requires_grad = False

        return sample_instance_idx, None
