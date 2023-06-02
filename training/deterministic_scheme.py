import torch
import numpy as np

LARGE_NUMBER = 1.e10

def rewire_global_directed(scores: torch.Tensor, k: int, adj: torch.Tensor):
    Batch, Nmax, _, ensemble = scores.shape

    k = min(k, Nmax ** 2 - torch.unique(adj[0], return_counts=True)[1].max().item())
    scores[adj] -= LARGE_NUMBER  # avoid selecting existing edges & self loops

    local_logits = scores.reshape(Batch, -1, ensemble)
    thresh = torch.topk(local_logits, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (local_logits >= thresh).to(torch.float)
    mask = mask.reshape(Batch, Nmax, Nmax, ensemble)
    return mask


def rewire_global_undirected(scores: torch.Tensor, k: int, adj: torch.Tensor):
    Batch, Nmax, _, ensemble = scores.shape

    k = min(k, (Nmax * (Nmax - 1)) // 2 - torch.unique(adj[0], return_counts=True)[1].max().item())

    scores[adj] -= LARGE_NUMBER  # avoid selecting existing edges & self loops
    scores = scores + scores.transpose(1, 2)
    triu_idx = np.triu_indices(Nmax, k=1)
    flat_local_logits = scores[:, triu_idx[0], triu_idx[1], :]
    thresh = torch.topk(flat_local_logits, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (flat_local_logits >= thresh).to(torch.float)
    new_mask = scores.new_zeros(scores.shape)
    new_mask[:, triu_idx[0], triu_idx[1], :] = mask
    new_mask = new_mask + new_mask.transpose(1, 2)
    return new_mask


def select_from_edge_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask
