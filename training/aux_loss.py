from typing import Tuple
from ml_collections import ConfigDict

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter

SMALL_EPS = 1.e-10


def entropy(scores, dim):
    """
    if scores are uniform, entropy maximized
    """
    return - (scores * torch.log2(scores + SMALL_EPS)).sum(dim=dim)


def get_degree_regularization(mask: torch.Tensor, auxloss: float, real_node_node_mask: torch.Tensor):
    mask = mask * real_node_node_mask.to(torch.float)[..., None]

    mask1 = mask.sum(1, keepdims=False)   # B x N x E
    mask1 = F.softmax(mask1, dim=1)
    loss1 = entropy(mask1, 1).mean()

    mask2 = mask.sum(2, keepdims=False)  # B x N x E
    mask2 = F.softmax(mask2, dim=1)
    loss2 = entropy(mask2, 1).mean()

    # we try to maximize the entropy, i.e., make mask uniform
    return - (loss1 + loss2) * auxloss


def get_variance_regularization(logits: torch.Tensor, auxloss: float, real_node_node_mask: torch.Tensor):
    B, N, _, E = logits.shape
    if E == 1:
        return 0.
    logits = logits / torch.linalg.norm(logits.detach(), dim=(1, 2), keepdim=True)  # detach so that the virtual nodes don't play a role
    logits = logits * real_node_node_mask.to(torch.float)[..., None]  # so that the virtual nodes don't play a role
    logits = torch.permute(logits, (0, 3, 1, 2)).reshape(B, E, N * N)
    dot = torch.einsum('bhm,bfm->bhf', logits, logits)
    eye = torch.eye(E, dtype=torch.float, device=logits.device)[None]
    loss = ((dot - eye) ** 2).mean()
    return loss * auxloss


def pairwise_KL_divergence(inputs, auxloss):
    """
    maximize the pair wise KL divergence
    https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html?highlight=kl_div#torch.nn.functional.kl_div
    We implement our own for pytorch does not batch the KL function

    Args:
        inputs:
        auxloss:

    Returns:

    """
    B, N, E = inputs.shape
    if E == 1:
        return 0.
    idx = np.triu_indices(N, k=1)
    inputs = torch.log_softmax(inputs, dim=-1)
    # func = torch.vmap(partial(F.kl_div, reduction='batchmean', log_target=True))
    src = inputs[:, idx[0]]
    dst = inputs[:, idx[1]]
    # KL is asymmetric
    # loss = func(src, dst).mean() + func(dst, src).mean()
    loss = (dst.exp() * (dst - src)).sum(-1).mean() + (src.exp() * (src - dst)).sum(-1).mean()
    return -loss * auxloss


def batch_kl_divergence(inputs, auxloss):
    """
    Sum the inputs across the batchsize dimension, and minimize KL between the result vec and 1s vector.

    Args:
        inputs:
        auxloss:

    Returns:

    """
    B, N, E = inputs.shape
    if E == 1:
        return 0.
    src = torch.log_softmax(inputs.sum(-1), dim=-1)
    dst = inputs.new_ones(B, N) / float(N)
    loss = F.kl_div(src, dst, reduction='batchmean', log_target=False)
    return loss * auxloss


def max_l2_distance_loss(inputs, auxloss):
    """
    Try to maximize the L2 distance of every pairs

    Args:
        inputs: shape (B, N, E)
        auxloss:

    Returns:

    """
    B, N, E = inputs.shape
    if E == 1:
        return 0.
    pair_wise_dist = (inputs[:, :, None, :] - inputs[:, None, :, :]).norm(2, dim=-1)
    # do not calculate the inverse pair
    idx = np.triu_indices(N, k=1)
    vals = pair_wise_dist[:, idx[0], idx[1]]
    loss = -vals.mean()
    return loss * auxloss


def max_min_l2_distance_loss(inputs, auxloss):
    """
    Try to maximize the minimum L2 distance

    Args:
        inputs: shape (B, N, E)
        auxloss:

    Returns:

    """
    B, N, E = inputs.shape
    if E == 1:
        return 0.
    pair_wise_dist = (inputs[:, :, None, :] - inputs[:, None, :, :]).norm(2, dim=-1)
    # mask = 1. - torch.eye(N).to(inputs.device)[None]
    # masked_pair_wise_dist = pair_wise_dist * mask
    idx = pair_wise_dist.nonzero().t()
    nonzero_vals = pair_wise_dist[idx[0], idx[1], idx[2]]
    # use scatter in case idx[0] for each graph is not uniform
    loss = -scatter(nonzero_vals, idx[0], dim=0, reduce='min').mean()
    return loss * auxloss


def cosine_similarity_loss(logits: torch.Tensor, auxloss: float):
    """
    Try to minimize the dot prod of different vecs dot(v, w)

    Args:
        logits:
        auxloss:

    Returns:

    """
    B, N, E = logits.shape
    if E == 1:
        return 0.
    logits = logits / torch.linalg.norm(logits.detach(), dim=1, keepdim=True)  # detach so that the virtual nodes don't play a role
    dot = torch.einsum('bmh,bmf->bhf', logits, logits)
    eye = torch.eye(E, dtype=torch.float, device=logits.device)[None]
    loss = ((dot - eye) ** 2).mean()
    return loss * auxloss


def get_original_bias(adj: Tuple[torch.Tensor], logits: torch.Tensor, auxloss: float):
    B, N, _, E = logits.shape
    # remove self loops, we don't want bias on self loops
    non_loop_idx = adj[1] != adj[2]
    adj = (adj[0][non_loop_idx], adj[1][non_loop_idx], adj[2][non_loop_idx])
    logits = F.softmax(logits, dim=2)  # make sure positive
    loss = torch.log((logits + SMALL_EPS)[adj])   # try to max this
    loss = loss.mean()
    return - loss * auxloss


def get_auxloss(auxloss_dict: ConfigDict, logits: torch.FloatTensor, node_mask: torch.FloatTensor):
    auxloss = 0.
    if auxloss_dict is None:
        return auxloss

    if hasattr(auxloss_dict, 'min_l2'):
        auxloss = auxloss + max_min_l2_distance_loss(logits, auxloss_dict.min_l2, )
    if hasattr(auxloss_dict, 'l2'):
        auxloss = auxloss + max_l2_distance_loss(logits, auxloss_dict.l2, )
    if hasattr(auxloss_dict, 'mask_l2'):
        auxloss = auxloss + max_l2_distance_loss(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_l2, )
    if hasattr(auxloss_dict, 'cos'):
        auxloss = auxloss + cosine_similarity_loss(logits, auxloss_dict.cos, )
    if hasattr(auxloss_dict, 'mask_cos'):
        auxloss = auxloss + cosine_similarity_loss(node_mask.reshape(-1, *node_mask.shape[-2:]),
                                                   auxloss_dict.mask_cos, )
    if hasattr(auxloss_dict, 'kl'):
        auxloss = auxloss + pairwise_KL_divergence(logits, auxloss_dict.kl, )
    if hasattr(auxloss_dict, 'mask_kl'):
        auxloss = auxloss + pairwise_KL_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_kl, )
    if hasattr(auxloss_dict, 'batch_kl'):
        # targeted at node mask
        auxloss = auxloss + batch_kl_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.batch_kl, )
    return auxloss
