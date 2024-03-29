import copy

from typing import List, Callable, Dict, Any

from math import ceil
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from data.utils.datatype_utils import (DuoDataStructure)
from data.utils.tensor_utils import (non_merge_coalesce,
                                     batch_repeat_edge_index)
from training.aux_loss import (cosine_similarity_loss,
                               max_min_l2_distance_loss,
                               max_l2_distance_loss,
                               pairwise_KL_divergence,
                               batch_kl_divergence)

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
    sampler_class.policy = 'edge_candid'  # because we are sampling from edge_index candidate
    sampler_class.k = samplek_dict['del_k']
    node_mask, _ = forward_func(logits)

    if auxloss_dict is not None:
        if hasattr(auxloss_dict, 'min_l2'):
            auxloss = auxloss + max_min_l2_distance_loss(deletion_logits, auxloss_dict.min_l2, )
        if hasattr(auxloss_dict, 'l2'):
            auxloss = auxloss + max_l2_distance_loss(deletion_logits, auxloss_dict.l2, )
        if hasattr(auxloss_dict, 'mask_l2'):
            auxloss = auxloss + max_l2_distance_loss(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_l2, )
        if hasattr(auxloss_dict, 'cos'):
            auxloss = auxloss + cosine_similarity_loss(deletion_logits, auxloss_dict.cos, )
        if hasattr(auxloss_dict, 'mask_cos'):
            auxloss = auxloss + cosine_similarity_loss(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_cos, )
        if hasattr(auxloss_dict, 'kl'):
            auxloss = auxloss + pairwise_KL_divergence(deletion_logits, auxloss_dict.kl, )
        if hasattr(auxloss_dict, 'mask_kl'):
            auxloss = auxloss + pairwise_KL_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_kl, )
        if hasattr(auxloss_dict, 'batch_kl'):
            # targeted at node mask
            auxloss = auxloss + batch_kl_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.batch_kl, )

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


def get_weighted_mask(weight_edges: str,
                      node_mask: torch.Tensor,
                      marginals: torch.Tensor,
                      output_logits: torch.Tensor,
                      mask_out: bool,
                      repeats: int):
    if weight_edges == 'None' or weight_edges is None:
        sampled_edge_weights = node_mask
    else:
        if weight_edges == 'marginals':
            # Maybe we should also try this with softmax?
            sampled_edge_weights = torch.stack([marginals] * repeats, dim=0)
        elif weight_edges == 'logits':
            sampled_edge_weights = torch.stack([output_logits] * repeats, dim=0)
        elif weight_edges == 'sigmoid_logits':
            sampled_edge_weights = torch.stack([torch.sigmoid(output_logits)] * repeats, dim=0)
        else:
            raise ValueError(f"{weight_edges} not supported")
        if mask_out:
            sampled_edge_weights = sampled_edge_weights * node_mask

    return sampled_edge_weights


def assign_layer_wise_info(rewire_layers: List,
                           num_layers: int,
                           rewired_batch: Batch,
                           dumb_repeat_edge_slice: torch.Tensor,
                           dumb_repeat_edge_index: torch.Tensor,
                           dumb_repeat_edge_attr: torch.Tensor,
                           ):
    per_layer_slice_dict = [rewired_batch._slice_dict['edge_index'] if idx_l in rewire_layers
                            else dumb_repeat_edge_slice for idx_l in range(num_layers)]
    per_layer_edge_index = [rewired_batch.edge_index if idx_l in rewire_layers
                            else dumb_repeat_edge_index for idx_l in range(num_layers)]
    per_layer_edge_attr = [rewired_batch.edge_attr if idx_l in rewire_layers
                           else dumb_repeat_edge_attr for idx_l in range(num_layers)] \
        if dumb_repeat_edge_attr is not None else [None] * num_layers
    pad = torch.ones(dumb_repeat_edge_index.shape[1], dtype=torch.float,
                     device=dumb_repeat_edge_index.device)
    per_layer_edge_weight = [rewired_batch.edge_weight if idx_l in rewire_layers
                             else pad for idx_l in range(num_layers)]

    rewired_batch._slice_dict['edge_index'] = per_layer_slice_dict
    rewired_batch.edge_index = per_layer_edge_index
    rewired_batch.edge_attr = per_layer_edge_attr
    rewired_batch.edge_weight = per_layer_edge_weight

    rewired_batch._slice_dict['edge_weight'] = rewired_batch._slice_dict['edge_index']
    if rewired_batch.edge_attr is not None:
        rewired_batch._slice_dict['edge_attr'] = rewired_batch._slice_dict['edge_index']

    return rewired_batch


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
                                  weight_edges: str,
                                  marginals_mask: bool,
                                  include_original_graph: bool,
                                  separate: bool = False,
                                  in_place: bool = True,
                                  directed_sampling: bool = False,
                                  num_layers: int = None,
                                  rewire_layers: List = None,
                                  auxloss_dict: ConfigDict = None,
                                  sample_ratio: float = 1.0):
    """
    A super lengthy, cpmplicated and coupled function
    should find time to clean and comment it
    """
    assert in_place

    auxloss = 0.
    plot_scores = dict()

    VE = sampler_class.train_ensemble if train else sampler_class.val_ensemble
    E = addition_logits.shape[-1]
    B = len(dat_batch.nnodes)
    N = dat_batch.num_edge_candidate.max().item()

    new_samplek_dict = copy.deepcopy(samplek_dict)

    # # ==================update del and add k for dynamic gnn=====================
    # Needed for strange behaviour in which samplek_dict remains modified after calling
    # the function, even if the old values are restored

    if sample_ratio != 1.0:
        new_samplek_dict['del_k'] = ceil(new_samplek_dict['del_k'] * sample_ratio)
        new_samplek_dict['add_k'] = ceil(new_samplek_dict['add_k'] * sample_ratio)

    assert new_samplek_dict['del_k'] > 0 or new_samplek_dict['add_k'] > 0

    # ===============================edge addition======================================
    # (B x Nmax x E) (B x Nmax)

    if new_samplek_dict['add_k'] > 0:
        output_logits, real_node_mask = to_dense_batch(addition_logits,
                                                       torch.arange(len(dat_batch.nnodes),
                                                                    device=addition_logits.device).repeat_interleave(
                                                           dat_batch.num_edge_candidate),
                                                       max_num_nodes=dat_batch.num_edge_candidate.max())

        padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
        logits = output_logits - padding_bias

        # (#sampled, B, Nmax, E), (B, Nmax, E)
        sampler_class.k = new_samplek_dict['add_k']
        node_mask, marginals = train_forward(logits) if train else val_forward(logits)

        if train and auxloss_dict is not None:
            if hasattr(auxloss_dict, 'min_l2'):
                auxloss = auxloss + max_min_l2_distance_loss(output_logits, auxloss_dict.min_l2, )
            if hasattr(auxloss_dict, 'l2'):
                auxloss = auxloss + max_l2_distance_loss(output_logits, auxloss_dict.l2, )
            if hasattr(auxloss_dict, 'mask_l2'):
                auxloss = auxloss + max_l2_distance_loss(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_l2, )
            if hasattr(auxloss_dict, 'cos'):
                auxloss = auxloss + cosine_similarity_loss(output_logits, auxloss_dict.cos, )
            if hasattr(auxloss_dict, 'mask_cos'):
                auxloss = auxloss + cosine_similarity_loss(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_cos, )
            if hasattr(auxloss_dict, 'kl'):
                auxloss = auxloss + pairwise_KL_divergence(output_logits, auxloss_dict.kl, )
            if hasattr(auxloss_dict, 'mask_kl'):
                auxloss = auxloss + pairwise_KL_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.mask_kl, )
            if hasattr(auxloss_dict, 'batch_kl'):
                # targeted at node mask
                auxloss = auxloss + batch_kl_divergence(node_mask.reshape(-1, *node_mask.shape[-2:]), auxloss_dict.batch_kl, )

        plot_scores['add'] = (output_logits.detach().clone(), node_mask.detach().clone())

        sampled_edge_weights = get_weighted_mask(weight_edges,
                                                 node_mask,
                                                 marginals,
                                                 output_logits,
                                                 marginals_mask or not train,
                                                 VE)

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
    if new_samplek_dict['del_k'] > 0:
        del_edge_weight, return_logits, auxloss = sample4deletion(dat_batch,
                                                   deletion_logits,
                                                   train_forward if train else val_forward,
                                                   sampler_class,
                                                   new_samplek_dict,
                                                   directed_sampling,
                                                   auxloss,
                                                   auxloss_dict if train else None,
                                                   deletion_logits.device)
        plot_scores['del'] = return_logits
    else:
        del_edge_weight = None

    new_graphs = graphs * (E * VE)
    dumb_repeat_batch = Batch.from_data_list(new_graphs)
    dumb_repeat_edge_index = dumb_repeat_batch.edge_index
    dumb_repeat_edge_slice = dumb_repeat_batch._slice_dict['edge_index'].to(dumb_repeat_edge_index.device)
    dumb_repeat_edge_attr = dumb_repeat_batch.edge_attr if dumb_repeat_batch.edge_attr is not None else None

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

        if rewire_layers is not None:
            rewired_batch = assign_layer_wise_info(rewire_layers,
                                                   num_layers,
                                                   rewired_batch,
                                                   dumb_repeat_edge_slice,
                                                   dumb_repeat_edge_index,
                                                   dumb_repeat_edge_attr)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, plot_scores, auxloss
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

            if rewire_layers is not None:
                add_rewired_batch = assign_layer_wise_info(rewire_layers,
                                                           num_layers,
                                                           add_rewired_batch,
                                                           dumb_repeat_edge_slice,
                                                           dumb_repeat_edge_index,
                                                           dumb_repeat_edge_attr)

            candidates = [add_rewired_batch]
        else:
            candidates = []

        if del_edge_weight is not None:
            del_rewired_batch = Batch.from_data_list(new_graphs)
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, train)

            if rewire_layers is not None:
                del_rewired_batch = assign_layer_wise_info(rewire_layers,
                                                           num_layers,
                                                           del_rewired_batch,
                                                           dumb_repeat_edge_slice,
                                                           dumb_repeat_edge_index,
                                                           dumb_repeat_edge_attr)

            candidates.append(del_rewired_batch)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=candidates,
                                     y=dumb_repeat_batch.y,
                                     num_graphs=dumb_repeat_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, plot_scores, auxloss
