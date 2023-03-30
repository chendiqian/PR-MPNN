import torch
import numpy as np


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
