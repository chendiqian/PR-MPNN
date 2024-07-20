import torch
from torch_geometric.utils import index_sort
from torch_scatter import scatter


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


def sparsify_edge_weight(data, edge_weight, train):
    """
    a trick to sparsify the training weights

    Args:
        data: graph Batch data
        edge_weight: edge weight for weighting the message passing, (if train) require grad to train the upstream
        train: whether to mask out the 0 entries in the mask or not, if train, keep them for grad

    Returns:
        (sparsified) graph Batch data
    """
    device = edge_weight.device

    edge_ptr = data._slice_dict['edge_index'].to(device)
    nedges = edge_ptr[1:] - edge_ptr[:-1]

    if train:
        data.edge_weight = edge_weight
    else:
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        data.edge_index = data.edge_index[:, nonzero_idx]
        data.edge_weight = edge_weight[nonzero_idx]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[nonzero_idx]
        new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                      torch.arange(len(nedges), device=device).repeat_interleave(nedges), reduce='sum',
                                      dim_size=len(nedges))
        data._slice_dict['edge_index'] = torch.cumsum(torch.hstack([new_edges_per_graph.new_zeros(1),
                                                                    new_edges_per_graph]), dim=0)

    data._slice_dict['edge_weight'] = data._slice_dict['edge_index']
    data._inc_dict['edge_weight'] = data._inc_dict['edge_index'].new_zeros(data._inc_dict['edge_index'].shape)
    if data.edge_attr is not None:
        data._slice_dict['edge_attr'] = data._slice_dict['edge_index']
        data._inc_dict['edge_attr'] = data._inc_dict['edge_weight']

    return data
