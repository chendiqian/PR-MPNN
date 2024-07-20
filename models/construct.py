from typing import List, Callable, Dict, Any

import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from data.utils.datatype_utils import (DuoDataStructure)
from data.utils.tensor_utils import (non_merge_coalesce,
                                     batch_repeat_edge_index)
from models.aux_loss import get_auxloss

LARGE_NUMBER = 1.e10


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

    if train:
        data.edge_weight = edge_weight
    else:
        edge_ptr = data._slice_dict['edge_index'].to(device)
        nedges = edge_ptr[1:] - edge_ptr[:-1]
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        data.edge_index = data.edge_index[:, nonzero_idx]
        data.edge_weight = edge_weight[nonzero_idx]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[nonzero_idx]
        new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                      torch.arange(len(nedges), device=device).repeat_interleave(nedges), reduce='sum', dim_size=len(nedges))
        data._slice_dict['edge_index'] = torch.cumsum(torch.hstack([new_edges_per_graph.new_zeros(1),
                                                                    new_edges_per_graph]), dim=0)

    data._slice_dict['edge_weight'] = data._slice_dict['edge_index']
    data._inc_dict['edge_weight'] = data._inc_dict['edge_index'].new_zeros(data._inc_dict['edge_index'].shape)
    if data.edge_attr is not None:
        data._slice_dict['edge_attr'] = data._slice_dict['edge_index']
        data._inc_dict['edge_attr'] = data._inc_dict['edge_weight']

    return data


def sample4deletion(dat_batch: Data,
                    deletion_logits: torch.Tensor,
                    forward_func: Callable,
                    sampler_class: Any,
                    samplek_dict: Dict,
                    directed_sampling: bool,
                    auxloss: float,
                    auxloss_dict: ConfigDict,
                    device: torch.device,):
    batch_idx = torch.arange(len(dat_batch.nedges), device=device).repeat_interleave(dat_batch.nedges)
    if directed_sampling:
        deletion_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                         batch_idx,
                                                         max_num_nodes=dat_batch.nedges.max())
    else:
        direct_mask = dat_batch.edge_index[0] <= dat_batch.edge_index[1]
        directed_edge_index = dat_batch.edge_index[:, direct_mask]
        num_direct_edges = scatter(direct_mask.long(),
                                   batch_idx,
                                   reduce='sum',
                                   dim_size=len(dat_batch.nedges))
        deletion_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                         batch_idx[direct_mask],
                                                         max_num_nodes=num_direct_edges.max())

    # we select the least scores
    deletion_logits = -deletion_logits
    padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = deletion_logits - padding_bias

    # (#sampled, B, Nmax, E), (B, Nmax, E)
    sampler_class.k = samplek_dict['del_k']
    node_mask, _ = forward_func(logits)

    auxloss = auxloss + get_auxloss(auxloss_dict, deletion_logits, node_mask)
    return_logits = (deletion_logits.detach().clone(), node_mask.detach().clone())
    VE, B, N, E = node_mask.shape
    sampled_edge_weights = 1. - node_mask

    # num_edges x E x VE
    del_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
    if not directed_sampling:
        # reduce must be mean, otherwise the self loops have double weights
        _, del_edge_weight = to_undirected(directed_edge_index, del_edge_weight,
                                           num_nodes=dat_batch.num_nodes,
                                           reduce='mean')
    del_edge_weight = del_edge_weight.t().reshape(-1)

    return del_edge_weight, return_logits, auxloss


def construct_from_edge_candidate(dat_batch: Data,
                                  graphs: List[Data],
                                  train: bool,
                                  addition_logits: torch.Tensor,
                                  deletion_logits: torch.Tensor,
                                  edge_candidate_idx: torch.Tensor,

                                  samplek_dict: Dict,
                                  sampler_class,
                                  train_forward: Callable,
                                  val_forward: Callable,
                                  include_original_graph: bool,
                                  separate: bool = False,
                                  directed_sampling: bool = False,
                                  auxloss_dict: ConfigDict = None):
    """
    A super lengthy, cpmplicated and coupled function
    should find time to clean and comment it
    """

    VE = sampler_class.train_ensemble if train else sampler_class.val_ensemble
    E = addition_logits.shape[-1]
    B = len(dat_batch.nnodes)
    N = dat_batch.num_edge_candidate.max().item()

    # # ==================update del and add k for dynamic gnn=====================
    # Needed for strange behaviour in which samplek_dict remains modified after calling
    # the function, even if the old values are restored

    assert samplek_dict['del_k'] > 0 or samplek_dict['add_k'] > 0

    # ===============================edge addition======================================
    # (B x Nmax x E) (B x Nmax)

    auxloss = 0.
    if samplek_dict['add_k'] > 0:
        output_logits, real_node_mask = to_dense_batch(addition_logits,
                                                       torch.arange(len(dat_batch.nnodes),
                                                                    device=addition_logits.device).repeat_interleave(
                                                           dat_batch.num_edge_candidate),
                                                       max_num_nodes=dat_batch.num_edge_candidate.max())

        padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
        logits = output_logits - padding_bias

        # (#sampled, B, Nmax, E), (B, Nmax, E)
        sampler_class.k = samplek_dict['add_k']
        node_mask, marginals = train_forward(logits) if train else val_forward(logits)
        if train:
            auxloss = get_auxloss(auxloss_dict, output_logits, node_mask)

        sampled_edge_weights = torch.stack([marginals] * VE, dim=0)
        if not train:
            sampled_edge_weights = sampled_edge_weights * node_mask

        # num_edges x E x VE
        add_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
        add_edge_index = edge_candidate_idx.T

        if not directed_sampling:
            add_edge_index, add_edge_weight = to_undirected(add_edge_index,
                                                            add_edge_weight,
                                                            num_nodes=dat_batch.num_nodes)

        add_edge_index = batch_repeat_edge_index(add_edge_index, dat_batch.num_nodes, E * VE)
        add_edge_weight = add_edge_weight.t().reshape(-1)
    else:
        add_edge_weight = None
        add_edge_index = None

    # =============================edge deletion===================================
    if samplek_dict['del_k'] > 0:
        del_edge_weight, return_logits, auxloss = sample4deletion(dat_batch,
                                                   deletion_logits,
                                                   train_forward if train else val_forward,
                                                   sampler_class,
                                                   samplek_dict,
                                                   directed_sampling,
                                                   auxloss,
                                                   auxloss_dict if train else None,
                                                   deletion_logits.device)
    else:
        del_edge_weight = None

    new_graphs = graphs * (E * VE)
    dumb_repeat_batch = Batch.from_data_list(new_graphs)
    dumb_repeat_edge_index = dumb_repeat_batch.edge_index

    # del and add are modified on the same (ensembling) graph
    if not separate and del_edge_weight is not None and add_edge_weight is not None:
        rewired_batch = dumb_repeat_batch
        merged_edge_index = torch.cat([rewired_batch.edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat([del_edge_weight, add_edge_weight], dim=-1)
        if dat_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([rewired_batch.edge_attr,
                                          rewired_batch.edge_attr.new_zeros(add_edge_weight.shape[-1],
                                                                            dat_batch.edge_attr.shape[1])], dim=0)
        else:
            merged_edge_attr = None

        # pyg coalesce force to merge duplicate edges, which is in conflict with our _slice_dict calculation
        merged_edge_index, merged_edge_attr, merged_edge_weight = non_merge_coalesce(
            edge_index=merged_edge_index,
            edge_attr=merged_edge_attr,
            edge_weight=merged_edge_weight,
            num_nodes=dat_batch.num_nodes * VE * E)
        rewired_batch.edge_index = merged_edge_index
        rewired_batch.edge_attr = merged_edge_attr

        # inc dict
        original_num_edges = dat_batch.nedges.repeat(E * VE)
        new_num_edges = (dat_batch.num_edge_candidate * (2 if not directed_sampling else 1)).repeat(E * VE)
        rewired_batch._slice_dict['edge_index'] = torch.hstack([add_edge_index.new_zeros(1),
                                                                (original_num_edges + new_num_edges).cumsum(dim=0)])

        rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, train)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, auxloss
    else:
        # for the graph adding with rewired edges in-place
        if add_edge_weight is not None:
            merged_edge_index = torch.cat([dumb_repeat_edge_index, add_edge_index], dim=1)
            merged_edge_weight = torch.cat([add_edge_weight.new_ones(dumb_repeat_edge_index.shape[1]), add_edge_weight], dim=-1)
            if dat_batch.edge_attr is not None:
                merged_edge_attr = torch.cat([dat_batch.edge_attr.repeat(E * VE, 1),
                                              dat_batch.edge_attr.new_zeros(
                                                  add_edge_weight.shape[-1],
                                                  dat_batch.edge_attr.shape[1])], dim=0)
            else:
                merged_edge_attr = None

            # pyg coalesce force to merge duplicate edges, which is in conflict with our _slice_dict calculation
            merged_edge_index, merged_edge_attr, merged_edge_weight = non_merge_coalesce(
                edge_index=merged_edge_index,
                edge_attr=merged_edge_attr,
                edge_weight=merged_edge_weight,
                num_nodes=dat_batch.num_nodes * VE * E)
            add_rewired_batch = dumb_repeat_batch
            add_rewired_batch.edge_index = merged_edge_index
            add_rewired_batch.edge_attr = merged_edge_attr

            # inc dict
            original_num_edges = dat_batch.nedges.repeat(E * VE)
            new_num_edges = (dat_batch.num_edge_candidate * (2 if not directed_sampling else 1)).repeat(E * VE)
            add_rewired_batch._slice_dict['edge_index'] = torch.hstack([add_edge_index.new_zeros(1),
                 (original_num_edges + new_num_edges).cumsum(dim=0)])

            add_rewired_batch = sparsify_edge_weight(add_rewired_batch, merged_edge_weight, train)

            candidates = [add_rewired_batch]
        else:
            candidates = []

        if del_edge_weight is not None:
            del_rewired_batch = Batch.from_data_list(new_graphs)
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, train)
            candidates.append(del_rewired_batch)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=candidates,
                                     y=dumb_repeat_batch.y,
                                     num_graphs=dumb_repeat_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, auxloss
