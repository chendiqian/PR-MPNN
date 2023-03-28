import torch
import torch.nn.functional as F


kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def get_kl_aux_loss(logits: torch.Tensor, auxloss: float,):
    """
    A KL divergence version
    """
    # Only when the input is flattened
    B, N, _, E = logits.shape
    logits = torch.permute(logits, (0, 3, 1, 2)).reshape(B, E, N * N)

    logits = logits.sum(1, keepdims=False)
    targets = logits.new_ones(logits.shape)

    log_softmax_logits = F.log_softmax(logits, dim=-1)
    log_softmax_target = F.log_softmax(targets, dim=-1)
    loss = kl_loss(log_softmax_logits, log_softmax_target)
    return loss * auxloss  # care the sign


def get_norm_aux_loss(logits: torch.Tensor, auxloss: float, real_node_node_mask: torch.Tensor):
    B, N, _, E = logits.shape
    logits = logits / torch.linalg.norm(logits.detach(), dim=(1, 2), keepdim=True)  # detach so that the virtual nodes don't play a role
    logits = logits * real_node_node_mask.to(torch.float)[..., None]  # so that the virtual nodes don't play a role
    logits = torch.permute(logits, (0, 3, 1, 2)).reshape(B, E, N * N)
    dot = torch.einsum('bhm,bfm->bhf', logits, logits)
    eye = torch.eye(E, dtype=torch.float, device=logits.device)[None]
    loss = ((dot - eye) ** 2).mean()
    return loss * auxloss
