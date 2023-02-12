from typing import Optional

import torch
from torch_scatter import scatter


def softmax_all(emb: torch.Tensor,
                nnodes: torch.Tensor,
                ptr: Optional[torch.Tensor] = None):
    """
    apply softmax of the whole graph
    return (n ** 2, ensemble)
    if reshape to (n, n, ensemble), sum(1) should be all 1s

    Args:
        emb:
        nnodes:
        ptr:

    Returns:

    """
    if ptr is None:
        idx = torch.zeros(1, dtype=torch.long, device=emb.device)
    else:
        idx = torch.cat([torch.repeat_interleave(torch.arange(n, device=emb.device), n) + ptr[i] for i, n in enumerate(nnodes)], dim=0)

    exp_emb = torch.exp(emb)
    sum_exp_emb = scatter(src=exp_emb,
                          index=idx,
                          dim=0, reduce='sum')
    sft_emb = exp_emb / sum_exp_emb[idx]
    return sft_emb


def softmax_topk(emb: torch.Tensor,
                 nnodes: torch.Tensor,
                 k: int = 1,
                 training: bool = False):
    ensemble = emb.shape[-1]
    embs = torch.split(emb, (nnodes ** 2).cpu().tolist(), dim=0)   # n1 ^ 2 * ensemble, n2 ^ 2 * ensemble, ...

    softmax_embs = []

    for i, emb in enumerate(embs):
        emb = emb.reshape(nnodes[i], nnodes[i], ensemble)
        # if training:
        #     emb = emb + torch.randn(emb.shape, device=emb.device)
        thresh = torch.topk(emb, min(k, nnodes[i]), dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
        mask = emb >= thresh
        bias = torch.zeros(emb.shape, device=emb.device)
        bias[~mask] = 1.e10
        emb = emb - bias
        emb = torch.nn.functional.softmax(emb, dim=1)
        softmax_embs.append(emb.reshape(nnodes[i] * nnodes[i], ensemble))

    return torch.cat(softmax_embs, dim=0)
