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
