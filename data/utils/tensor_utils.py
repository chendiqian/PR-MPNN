import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import index_sort


def batched_edge_index_to_batched_adj(data: Data):
    """

    Args:
        data: should be the original batch, i.e. without ensembles

    Returns:
        batched edge_index ([0, 0, 0, 1, 1, 1, ...],
                            [0, 0, 1, 0, 1, 1, ...],
                            [1, 2, 1, 0, 0, 2, ...]) with self loops

    """
    device = data.x.device
    graph_idx_mask = torch.repeat_interleave(torch.arange(len(data.nedges), device=device), data.nedges)
    edge_index_rel = torch.repeat_interleave(data._inc_dict['edge_index'].to(device), data.nedges)
    local_edge_index = data.edge_index - edge_index_rel

    original_adj = (graph_idx_mask, local_edge_index[0], local_edge_index[1])

    # remove existing self loops
    non_loop_idx = local_edge_index[0] != local_edge_index[1]
    local_edge_index = local_edge_index[:, non_loop_idx]
    graph_idx_mask = graph_idx_mask[non_loop_idx]

    # add remaining self loops, because we don't want to sample from self loops
    self_loop_idx = torch.from_numpy(np.concatenate([np.arange(nn) for nn in data.nnodes.cpu().numpy()], axis=0)).to(device)
    local_edge_index = torch.hstack([local_edge_index, self_loop_idx[None].repeat(2, 1)])
    graph_idx_mask = torch.hstack([graph_idx_mask,
                                   torch.repeat_interleave(
                                       torch.arange(len(data.nnodes), device=device),
                                       data.nnodes
                                   )])
    ## a dense version, needs mem allocation
    # B = data.num_graphs
    # N = data.nnodes.max()
    # adj = torch.zeros(B, N, N, 1, dtype=target_dtype, device=device)
    # adj[graph_idx_mask, local_edge_index[0], local_edge_index[1]] = 1
    ## a sparse tensor index
    adj = (graph_idx_mask, local_edge_index[0], local_edge_index[1])
    return original_adj, adj


def self_defined_softmax(scores, mask):
    """
    A specific function

    Args:
        scores: B, N, N, E
        mask: same shape as scores

    Returns:

    """
    scores = scores - scores.detach().max()  # for numerical stability
    exp_scores = torch.exp(scores)
    exp_scores = exp_scores * mask
    softmax_scores = exp_scores / exp_scores.sum()
    return softmax_scores


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    https://github.com/rampasek/GraphGPS/blob/main/graphgps/loss/weighted_cross_entropy.py
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.shape[1] > 2 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.shape[1] > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(), weight=weight[true])
        return loss


def non_merge_coalesce(
    edge_index,
    edge_attr,
    edge_weight,
    num_nodes,
    is_sorted: bool = False,
    sort_by_row: bool = True,
):
    nnz = edge_index.size(1)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        edge_index = edge_index[:, perm]
        if isinstance(edge_attr, torch.Tensor):
            edge_attr = edge_attr[perm]
        if isinstance(edge_weight, torch.Tensor):
            edge_weight = edge_weight[perm]

    return edge_index, edge_attr, edge_weight


def batch_repeat_edge_index(edge_index: torch.Tensor, num_nodes: int, repeats: int):
    if repeats == 1:
        return edge_index

    edge_index_rel = torch.arange(repeats, device=edge_index.device).repeat_interleave(edge_index.shape[1]) * num_nodes
    edge_index = edge_index.repeat(1, repeats)
    edge_index += edge_index_rel
    return edge_index
