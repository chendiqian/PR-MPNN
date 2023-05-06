import torch
import numpy as np

LARGE_NUMBER = 1.e10

def rewire_global_directed(scores: torch.Tensor, k: int):
    """

    Args:
        scores:
        k:

    Returns:

    """
    Batch, Nmax, _, ensemble = scores.shape
    if k >= Nmax ** 2:
        return scores.new_ones(scores.shape)

    local_logits = scores.reshape(Batch, -1, ensemble)
    thresh = torch.topk(local_logits, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (local_logits >= thresh).to(torch.float)
    mask = mask.reshape(Batch, Nmax, Nmax, ensemble)
    return mask


def rewire_global_undirected(scores: torch.Tensor, k: int):
    """
    scores should be symmetric

    Args:
        scores:
        k:

    Returns:

    """
    Batch, Nmax, _, ensemble = scores.shape
    if k >= (Nmax * (Nmax - 1)) // 2:
        mask = scores.new_ones(scores.shape)
        diag_idx = np.diag_indices(Nmax)
        mask[:, diag_idx[0], diag_idx[1], :] = 0.
        return mask

    triu_idx = np.triu_indices(Nmax, k=1)
    flat_local_logits = scores[:, triu_idx[0], triu_idx[1], :]
    thresh = torch.topk(flat_local_logits, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (flat_local_logits >= thresh).to(torch.float)
    new_mask = scores.new_zeros(scores.shape)
    new_mask[:, triu_idx[0], triu_idx[1], :] = mask
    new_mask = new_mask + new_mask.transpose(1, 2)
    return new_mask


def rewire_global_semi(scores: torch.Tensor, k: int, adj: torch.Tensor):
    Batch, Nmax, _, ensemble = scores.shape

    triu_idx = np.triu_indices(Nmax, k=1)
    uniques_num_edges = adj[:, triu_idx[0], triu_idx[1], :].sum(dim=(1, 2))
    max_num_edges = uniques_num_edges.max().item()
    target_size = (Nmax * (Nmax - 1)) // 2 - max_num_edges

    if k > target_size:
        raise ValueError(f"k = {k} too large!")

    scores = scores - adj * LARGE_NUMBER  # do not sample existing edges
    triu_idx = np.triu_indices(Nmax, k=1)
    flat_local_logits = scores[:, triu_idx[0], triu_idx[1], :]
    thresh = torch.topk(flat_local_logits, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (flat_local_logits >= thresh).to(torch.float)
    new_mask = scores.new_zeros(scores.shape)
    new_mask[:, triu_idx[0], triu_idx[1], :] = mask
    new_mask = new_mask + new_mask.transpose(1, 2) + adj
    return new_mask


def select_from_edge_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1,
             :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask
