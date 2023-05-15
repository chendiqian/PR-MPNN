import numpy as np
import torch
import torch.nn.functional as F

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


def get_variance_regularization_3d(logits: torch.Tensor, auxloss: float):
    B, N, E = logits.shape
    if E == 1:
        return 0.
    logits = logits / torch.linalg.norm(logits.detach(), dim=1, keepdim=True)  # detach so that the virtual nodes don't play a role
    dot = torch.einsum('bmh,bmf->bhf', logits, logits)
    eye = torch.eye(E, dtype=torch.float, device=logits.device)[None]
    loss = ((dot - eye) ** 2).mean()
    return loss * auxloss


def get_original_bias(adj: torch.Tensor, logits: torch.Tensor, auxloss: float, real_node_node_mask: torch.Tensor):
    B, N, _, E = logits.shape
    diag_idx = np.diag_indices(N)
    adj[:, diag_idx[0], diag_idx[1]] = 0.   # remove self loops
    logits = F.softmax(logits, dim=2)  # make sure positive
    loss = adj * torch.log(logits + SMALL_EPS)   # try to max this
    loss = loss * real_node_node_mask.to(torch.float)[..., None]  # so that the virtual nodes don't play a role
    loss = loss.sum() / adj.sum()
    return - loss * auxloss
