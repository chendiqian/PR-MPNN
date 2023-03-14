import torch
import torch.nn.functional as F
import numpy as np


kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def get_batch_aux_loss(logits: torch.Tensor, auxloss: float,):
    """
    A KL divergence version
    """
    # Only when the input is flattened
    B, N, _, E = logits.shape
    logits = torch.permute(logits, (0, 1, 3, 2)).reshape(B, N, N * E)

    logits = logits.sum(1, keepdims=False)
    targets = logits.new_ones(logits.shape)

    log_softmax_logits = F.log_softmax(logits, dim=-1)
    log_softmax_target = F.log_softmax(targets, dim=-1)
    loss = kl_loss(log_softmax_logits, log_softmax_target)
    return loss * auxloss  # care the sign


def get_pair_aux_loss(logits: torch.Tensor, nnodes: torch.Tensor, auxloss: float):
    """

    Args:
        logits: Batch, Nmax, Nmax, ensemble
        nnodes:
        auxloss:

    Returns:

    """
    B, N, _, E = logits.shape
    logits = torch.permute(logits, (0, 1, 3, 2)).reshape(B, N, N * E)
    # so that every slice of batch looks like: e.g. each row is a hstack of ensemble, each row represents a node
    # tensor([[0.0100, 0.0200, -inf, 0.0100, 0.0201, -inf, 0.0101, 0.0201, -inf],
    #         [0.0200, 0.0400, -inf, 0.0201, 0.0401, -inf, 0.0201, 0.0401, -inf],
    #         [-inf,   -inf,   -inf, -inf,   -inf,   -inf, -inf,   -inf,   -inf]])
    idx_batch = torch.repeat_interleave(torch.arange(len(nnodes), device=logits.device),
                                        torch.div(nnodes * (nnodes - 1), 2,
                                                  rounding_mode='floor'), dim=0)
    idx = torch.from_numpy(np.concatenate([np.triu_indices(n, 1) for n in nnodes], axis=-1)).to(logits.device)
    log_softmax_logits = F.log_softmax(logits, dim=-1)
    source = log_softmax_logits[idx_batch, idx[0],]
    target = log_softmax_logits[idx_batch, idx[1],]
    loss = kl_loss(source, target)
    return - loss * auxloss