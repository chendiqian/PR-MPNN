from typing import Tuple, List, Callable, Dict

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from data.data_utils import DuoDataStructure, batched_edge_index_to_batched_adj, non_merge_coalesce
from training.aux_loss import get_variance_regularization, get_original_bias, get_variance_regularization_3d

LARGE_NUMBER = 1.e10


def sparsify_edge_weight(data, edge_weight, negative_sample):
    """
    a trick to sparsify the training weights

    Args:
        data: graph Batch data
        edge_weight: edge weight for weighting the message passing, (if train) require grad to train the upstream
        negative_sample: whether to mask out the 0 entries in the mask or not

    Returns:
        (sparsified) graph Batch data
    """
    device = edge_weight.device
    if negative_sample == 'zero':
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
    elif negative_sample == 'full':
        data.edge_weight = edge_weight
    else:
        raise ValueError

    data._slice_dict['edge_weight'] = data._slice_dict['edge_index']
    data._inc_dict['edge_weight'] = data._inc_dict['edge_index'].new_zeros(data._inc_dict['edge_index'].shape)
    if data.edge_attr is not None:
        data._slice_dict['edge_attr'] = data._slice_dict['edge_index']
        data._inc_dict['edge_attr'] = data._inc_dict['edge_weight']

    return data


def construct_from_edge_candidate(collate_data: Tuple[Data, List[Data]],
                                  emb_model: Callable,
                                  samplek_dict: Dict,
                                  sampler_class,
                                  train_forward: Callable,
                                  val_forward: Callable,
                                  weight_edges: str,
                                  marginals_mask: bool,
                                  include_original_graph: bool,
                                  negative_sample: str,
                                  separate: bool = False,
                                  in_place: bool = True,
                                  directed_sampling: bool = False,
                                  auxloss_dict: ConfigDict = None):
    """
    rewire, addition is based on edge candidate set, deletion is from the original edges
    currently supports in-place rewire

    Args:
        collate_data:
        emb_model:
        samplek_dict:
        sampler_class:
        train_forward:
        val_forward:
        weight_edges:
        marginals_mask:
        include_original_graph:
        negative_sample:
        separate: to treat del / add as separate candidate graphs or on the same graph
        in_place: add edges on top the original edges, if no in-place, create a new graph of the added edges only
        directed_sampling:
        auxloss_dict:

    Returns:

    """
    assert in_place

    dat_batch, graphs = collate_data

    train = emb_model.training
    negative_sample = negative_sample if train else 'zero'
    # (sum_edges, ensemble)
    addition_logits, deletion_logits, edge_candidate_idx = emb_model(dat_batch)

    auxloss = 0.

    # ===============================edge addition======================================
    # (B x Nmax x E) (B x Nmax)
    output_logits, real_node_mask = to_dense_batch(addition_logits,
                                                   torch.arange(len(dat_batch.nnodes),
                                                                device=addition_logits.device).repeat_interleave(
                                                       dat_batch.num_edge_candidate),
                                                   max_num_nodes=dat_batch.num_edge_candidate.max())

    if train and auxloss_dict is not None and auxloss_dict.variance > 0:
        auxloss = auxloss + get_variance_regularization_3d(output_logits, auxloss_dict.variance, )

    padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = output_logits - padding_bias

    # (#sampled, B, Nmax, E), (B, Nmax, E)
    sampler_class.k = samplek_dict['add_k']
    node_mask, marginals = train_forward(logits) if train else val_forward(logits)
    VE, B, N, E = node_mask.shape

    if weight_edges == 'None' or weight_edges is None:
        sampled_edge_weights = node_mask
    else:
        if weight_edges == 'marginals':
            # Maybe we should also try this with softmax?
            sampled_edge_weights = marginals[None].repeat(VE, 1, 1, 1)
        elif weight_edges == 'logits':
            sampled_edge_weights = output_logits[None].repeat(VE, 1, 1, 1)
        elif weight_edges == 'sigmoid_logits':
            sampled_edge_weights = torch.sigmoid(output_logits)[None].repeat(VE, 1, 1, 1)
        else:
            raise ValueError(f"{weight_edges} not supported")

        if marginals_mask or not train:
            sampled_edge_weights = sampled_edge_weights * node_mask

    # num_edges x E x VE
    add_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
    add_edge_index = edge_candidate_idx.T

    if not directed_sampling:
        add_edge_index, add_edge_weight = to_undirected(add_edge_index,
                                                        add_edge_weight,
                                                        num_nodes=dat_batch.num_nodes)

    if E * VE > 1:
        edge_index_rel = (torch.arange(E * VE, device=add_edge_weight.device) * dat_batch.num_nodes).repeat_interleave(
            add_edge_index.shape[1])
        add_edge_index = add_edge_index.repeat(1, E * VE)
        add_edge_index += edge_index_rel

    add_edge_weight = add_edge_weight.T.flatten()

    # =============================edge deletion===================================
    if samplek_dict['del_k'] > 0:
        batch_idx = torch.arange(len(dat_batch.nedges), device=addition_logits.device).repeat_interleave(dat_batch.nedges)
        if directed_sampling:
            output_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                           batch_idx,
                                                           max_num_nodes=dat_batch.nedges.max())

        else:
            direct_mask = dat_batch.edge_index[0] <= dat_batch.edge_index[1]
            directed_edge_index = dat_batch.edge_index[:, direct_mask]
            num_direct_edges = scatter(direct_mask.long(), batch_idx, reduce='sum', dim_size=len(dat_batch.nedges))
            output_logits, real_node_mask = to_dense_batch(deletion_logits,
                                                           batch_idx[direct_mask],
                                                           max_num_nodes=num_direct_edges.max())


        if train and auxloss_dict is not None and auxloss_dict.variance > 0:
            auxloss = auxloss + get_variance_regularization_3d(output_logits, auxloss_dict.variance, )

        # we select the least scores
        output_logits = -output_logits
        padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
        logits = output_logits - padding_bias

        # (#sampled, B, Nmax, E), (B, Nmax, E)
        sampler_class.k = samplek_dict['del_k']
        node_mask, _ = train_forward(logits) if train else val_forward(logits)
        VE, B, N, E = node_mask.shape

        sampled_edge_weights = 1. - node_mask

        # num_edges x E x VE
        del_edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)
        if not directed_sampling:
            # reduce must be mean, otherwise the self loops have double weights
            _, del_edge_weight = to_undirected(directed_edge_index, del_edge_weight, num_nodes=dat_batch.num_nodes, reduce='mean')
        del_edge_weight = del_edge_weight.T.flatten()
    else:
        del_edge_weight = None

    new_graphs = graphs * (E * VE)

    if not separate:
        rewired_batch = Batch.from_data_list(new_graphs)
        merged_edge_index = torch.cat([rewired_batch.edge_index, add_edge_index], dim=1)
        if del_edge_weight is None:
            del_edge_weight = add_edge_weight.new_ones(dat_batch.edge_index.shape[1] * E * VE)
        merged_edge_weight = torch.cat([del_edge_weight, add_edge_weight], dim=0)
        if dat_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([rewired_batch.edge_attr,
                                          rewired_batch.edge_attr.new_zeros(add_edge_weight.shape[0],
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

        rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, negative_sample)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, None, auxloss
    else:
        del_rewired_batch = Batch.from_data_list(new_graphs)

        original_edge_index = del_rewired_batch.edge_index
        # for the graph adding with rewired edges in-place
        merged_edge_index = torch.cat([original_edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat([add_edge_weight.new_ones(original_edge_index.shape[1]), add_edge_weight], dim=0)
        if dat_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([del_rewired_batch.edge_attr,
                                          del_rewired_batch.edge_attr.new_zeros(add_edge_weight.shape[0],
                                                                                dat_batch.edge_attr.shape[1])], dim=0)
        else:
            merged_edge_attr = None

        # pyg coalesce force to merge duplicate edges, which is in conflict with our _slice_dict calculation
        merged_edge_index, merged_edge_attr, merged_edge_weight = non_merge_coalesce(
            edge_index=merged_edge_index,
            edge_attr=merged_edge_attr,
            edge_weight=merged_edge_weight,
            num_nodes=dat_batch.num_nodes * VE * E)
        add_rewired_batch = Batch.from_data_list(new_graphs)
        add_rewired_batch.edge_index = merged_edge_index
        add_rewired_batch.edge_attr = merged_edge_attr

        # inc dict
        original_num_edges = dat_batch.nedges.repeat(E * VE)
        new_num_edges = (dat_batch.num_edge_candidate * (2 if not directed_sampling else 1)).repeat(E * VE)
        add_rewired_batch._slice_dict['edge_index'] = torch.hstack([add_edge_index.new_zeros(1),
                                                                    (original_num_edges + new_num_edges).cumsum(dim=0)])

        add_rewired_batch = sparsify_edge_weight(add_rewired_batch, merged_edge_weight, negative_sample)
        candidates = [add_rewired_batch]

        if del_edge_weight is not None:
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, negative_sample)
            candidates.append(del_rewired_batch)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=candidates,
                                     y=add_rewired_batch.y,
                                     num_graphs=add_rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, None, auxloss


def construct_from_attention_mat(collate_data: Tuple[Data, List[Data]],
                                 emb_model: Callable,
                                 sample_policy: str,
                                 # samplek_dict: Dict,
                                 auxloss_dict: ConfigDict,
                                 sampler_class,
                                 train_forward: Callable,
                                 val_forward: Callable,
                                 weight_edges: str,
                                 marginals_mask: bool,
                                 device: torch.device,
                                 include_original_graph: bool,
                                 negative_sample: str,
                                 in_place: bool):
    dat_batch, graphs = collate_data

    train = emb_model.training
    negative_sample = negative_sample if train else 'zero'
    output_logits, real_node_node_mask = emb_model(dat_batch)

    padding_bias = (~real_node_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = output_logits - padding_bias

    auxloss = 0.
    if train and auxloss_dict is not None:
        if auxloss_dict.variance > 0:
            auxloss = auxloss + get_variance_regularization(output_logits,
                                                            auxloss_dict.variance,
                                                            real_node_node_mask)

    # (#sampled, B, N, N, E), (B, N, N, E)
    # # add edges from the N x N matrix
    # sampler_class.policy = sample_policy
    # sampler_class.k = samplek_dict['add_k']
    adj = batched_edge_index_to_batched_adj(dat_batch)
    sampler_class.adj = adj
    node_mask, marginals = train_forward(logits) if train else val_forward(logits)
    VE, B, N, _, E = node_mask.shape

    if weight_edges == 'None' or weight_edges is None:
        sampled_edge_weights = node_mask
    else:
        if weight_edges == 'marginals':
            # Maybe we should also try this with softmax?
            sampled_edge_weights = marginals[None].repeat(VE, 1, 1, 1, 1)
        elif weight_edges == 'logits':
            sampled_edge_weights = output_logits[None].repeat(VE, 1, 1, 1, 1)
        elif weight_edges == 'sigmoid_logits':
            sampled_edge_weights = torch.sigmoid(output_logits)[None].repeat(VE, 1, 1, 1, 1)
        else:
            raise ValueError(f"{weight_edges} not supported")

        if marginals_mask or not train:
            sampled_edge_weights = sampled_edge_weights * node_mask

    # B x E x VE
    edge_weight = sampled_edge_weights.permute((1, 2, 3, 4, 0))[real_node_node_mask]
    edge_weight = edge_weight.permute(2, 1, 0).flatten()

    new_graphs = [g.clone() for g in graphs]
    for g in new_graphs:
        g.edge_index = torch.from_numpy(np.vstack(np.triu_indices(g.num_nodes, k=-g.num_nodes))).to(device)
        del g.edge_attr

    new_graphs = new_graphs * (E * VE)
    rewired_batch = Batch.from_data_list(new_graphs)

    original_edge_index = dat_batch.edge_index
    original_num_edges = dat_batch.nedges.to(original_edge_index.device).repeat(E * VE)
    if E * VE > 1:
        edge_index_rel = (torch.arange(E * VE, device=edge_weight.device) * dat_batch.num_nodes).repeat_interleave(
            original_edge_index.shape[1])
        original_edge_index = original_edge_index.repeat(1, E * VE)
        original_edge_index += edge_index_rel

    if in_place:
        merged_edge_index = torch.cat([original_edge_index, rewired_batch.edge_index], dim=1)
        merged_edge_weight = torch.cat([edge_weight.new_ones(original_edge_index.shape[1]), edge_weight], dim=0)
        if dat_batch.edge_attr is not None:
            merged_edge_attr = torch.cat([dat_batch.edge_attr.repeat(E * VE, 1),
                                          dat_batch.edge_attr.new_zeros(edge_weight.shape[0],
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
        new_num_edges = (dat_batch.nnodes ** 2).repeat(E * VE)
        rewired_batch._slice_dict['edge_index'] = torch.hstack(
            [merged_edge_index.new_zeros(1), (original_num_edges + new_num_edges).cumsum(dim=0)])

        rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, negative_sample)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
    else:
        raise NotImplementedError
        # if dat_batch.edge_attr is not None:
        #     new_edge_index = rewired_batch.edge_index
        #     new_edge_index_id = (new_edge_index[0] * rewired_batch.num_nodes + new_edge_index[1]).cpu().numpy()
        #     original_edge_index_id = (
        #                 original_edge_index[0] * rewired_batch.num_nodes + original_edge_index[1]).cpu().numpy()
        #     new_edge_attr = dat_batch.edge_attr.new_zeros(new_edge_index.shape[1], dat_batch.edge_attr.shape[1])
        #     fill_idx = np.in1d(new_edge_index_id, original_edge_index_id)
        #     new_edge_attr[fill_idx] = dat_batch.edge_attr.repeat(E * VE, 1)
        #     rewired_batch.edge_attr = new_edge_attr
        #
        # rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, negative_sample)
        #
        # new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
        #                              candidates=[rewired_batch],
        #                              y=rewired_batch.y,
        #                              num_graphs=rewired_batch.num_graphs,
        #                              num_unique_graphs=len(graphs))
        # return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
