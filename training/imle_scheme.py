import torch
from subgraph.or_based import get_or_optim_subgraphs
from subgraph.greedy_expand import greedy_grow_tree


class IMLEScheme:
    def __init__(self, imle_sample_policy, ptr, graphs, sample_k):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self._ptr = ptr
        self._graphs = graphs

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

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):

        local_logits = logits.detach()
        local_logits = torch.split(local_logits, self.ptr, dim=0)

        sample_instance_idx = []
        if self.imle_sample_policy == 'KMaxNeighbors':
            for i, logit in enumerate(local_logits):
                logit = logit.reshape(self.graphs[i].num_nodes, self.graphs[i].num_nodes)
                mask = get_or_optim_subgraphs(self.graphs[i].edge_index, logit, self.sample_k)
                mask.requires_grad = False
                sample_instance_idx.append(mask.reshape(-1))
        elif self.imle_sample_policy == 'greedy_neighbors':
            for i, logit in enumerate(local_logits):
                logit = logit.reshape(self.graphs[i].num_nodes, self.graphs[i].num_nodes)
                mask = greedy_grow_tree(self.graphs[i], self.sample_k, logit)
                mask.requires_grad = False
                sample_instance_idx.append(mask.reshape(-1))
        else:
            raise NotImplementedError

        sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
        sample_instance_idx.requires_grad = False

        return sample_instance_idx, None
