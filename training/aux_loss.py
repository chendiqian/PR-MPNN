from typing import Tuple
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

def cosine_similarity_loss(inputs, auxloss):
    loss = 0
    for sample in inputs:
        # Calculate pairwise cosine similarity
        cosine_similarity_matrix = F.cosine_similarity(sample.unsqueeze(0), sample.unsqueeze(1), dim=2)
        # Calculate mean cosine similarity
        mean_cosine_similarity = cosine_similarity_matrix.mean()
        # Sum the negative mean cosine similarity to the loss
        loss -= mean_cosine_similarity
    # Return average loss
    return (loss / inputs.size(0)) * auxloss

def l2_distance_loss(inputs, auxloss):
    loss = 0
    for sample in inputs:
        # Calculate pairwise L2 distance
        l2_distance_matrix = torch.cdist(sample, sample, p=2)
        # Create a mask for off-diagonal elements
        mask = 1 - torch.eye(l2_distance_matrix.size(0)).to(sample.device)
        # Apply mask to L2 distance matrix
        l2_distance_matrix_masked = l2_distance_matrix * mask
        # Calculate minimum L2 distance, ignoring zero distances
        min_l2_distance = l2_distance_matrix_masked[l2_distance_matrix_masked.nonzero()].min()
        # Sum the negative minimum L2 distance to the loss
        loss -= min_l2_distance
    # Return average loss
    return (loss / inputs.size(0)) * auxloss

def get_variance_regularization_3d(logits: torch.Tensor, auxloss: float):
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
