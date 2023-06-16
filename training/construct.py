from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from data.data_utils import (DuoDataStructure,
                             batched_edge_index_to_batched_adj,
                             non_merge_coalesce,
                             batch_repeat_edge_index)
from training.aux_loss import get_variance_regularization, get_variance_regularization_3d

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
        if edge_weight.dim() == 1:
            nonzero_idx = edge_weight.nonzero().reshape(-1)
            data.edge_index = data.edge_index[:, nonzero_idx]
            data.edge_weight = edge_weight[nonzero_idx]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[nonzero_idx]
            new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                          torch.arange(len(nedges), device=device).repeat_interleave(nedges), reduce='sum', dim_size=len(nedges))
            data._slice_dict['edge_index'] = torch.cumsum(torch.hstack([new_edges_per_graph.new_zeros(1),
                                                                        new_edges_per_graph]), dim=0)
        else:
            L = edge_weight.shape[0]
            idx = torch.where(edge_weight)
            num_effective_edges = torch.unique(idx[0], sorted=True, return_counts=True, dim=0)[1].cpu().tolist()
            data.edge_index = torch.split(data.edge_index[None].repeat(L, 1, 1)[idx[0], :, idx[1]].t(),
                                          num_effective_edges,
                                          dim=1)
            if data.edge_attr is not None:
                edge_attrs = data.edge_attr[None].repeat(L, 1, 1)[idx[0], idx[1], :]
                data.edge_attr = torch.split(edge_attrs, num_effective_edges, dim=0)
            else:
                data.edge_attr = [None] * L
            data.edge_weight = torch.split(edge_weight[idx[0], idx[1]], num_effective_edges, dim=0)

            new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                          torch.arange(len(nedges), device=device).repeat_interleave(nedges),
                                          reduce='sum',
                                          dim_size=len(nedges), dim=1)
            _slice_dicts = torch.cumsum(torch.hstack([new_edges_per_graph.new_zeros(L, 1), new_edges_per_graph]), dim=1)
            data._slice_dict['edge_index'] = [_slice_dicts[i] for i in range(L)]
    elif negative_sample == 'full':
        if edge_weight.dim() == 1:
            # same across the layers
            data.edge_weight = edge_weight
        else:
            # sample per layer
            L = edge_weight.shape[0]
            data.edge_index = [data.edge_index] * L
            data.edge_attr = [data.edge_attr] * L
            data.edge_weight = [t.squeeze() for t in torch.split(edge_weight, 1, dim=0)]
    else:
        raise ValueError

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
                    ensemble: int,
                    directed_sampling: bool,
                    auxloss: float,
                    auxloss_dict: ConfigDict,
                    device: torch.device):
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

    if auxloss_dict is not None and auxloss_dict.variance > 0:
        auxloss = auxloss + get_variance_regularization_3d(deletion_logits,
                                                           auxloss_dict.variance)

    # we select the least scores
    deletion_logits = -deletion_logits
    padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = deletion_logits - padding_bias

    # (#sampled, B, Nmax, E), (B, Nmax, E)
    sampler_class.policy = 'edge_candid'  # because we are sampling from edge_index candidate
    sampler_class.k = samplek_dict['del_k']
    node_mask, _ = forward_func(logits)
    VE, B, N, E = node_mask.shape
    E, L = ensemble, E // ensemble
    node_mask = node_mask.reshape(VE, B, N, E, L)

    sampled_edge_weights = 1. - node_mask

    # num_edges x E x VE
    del_edge_weight = sampled_edge_weights.permute((1, 2, 4, 3, 0))[real_node_mask].reshape(-1, L, E * VE)
    if not directed_sampling:
        # reduce must be mean, otherwise the self loops have double weights
        _, del_edge_weight = to_undirected(directed_edge_index, del_edge_weight,
                                           num_nodes=dat_batch.num_nodes,
                                           reduce='mean')
    del_edge_weight = del_edge_weight.permute((1, 2, 0)).reshape(L, -1).squeeze(0)

    return del_edge_weight, auxloss


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


def construct_from_edge_candidate(dat_batch: Data,
                                  graphs: List[Data],
                                  train: bool,
                                  addition_logits: torch.Tensor,
                                  deletion_logits: torch.Tensor,
                                  edge_candidate_idx: torch.Tensor,

                                  ensemble: int,
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
    assert in_place
    negative_sample = negative_sample if train else 'zero'

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
    E, L = ensemble, E // ensemble

    sampled_edge_weights = get_weighted_mask(weight_edges,
                                             node_mask,
                                             marginals,
                                             output_logits,
                                             marginals_mask or not train,
                                             VE)

    sampled_edge_weights = sampled_edge_weights.reshape(VE, B, N, E, L)

    # num_edges x L x E x VE
    add_edge_weight = sampled_edge_weights.permute((1, 2, 4, 3, 0))[real_node_mask].reshape(-1, L, E * VE)
    add_edge_index = edge_candidate_idx.T

    if not directed_sampling:
        add_edge_index, add_edge_weight = to_undirected(add_edge_index,
                                                        add_edge_weight,
                                                        num_nodes=dat_batch.num_nodes)

    add_edge_index = batch_repeat_edge_index(add_edge_index, dat_batch.num_nodes, E * VE)

    add_edge_weight = add_edge_weight.permute((1, 2, 0)).reshape(L, -1).squeeze(0)

    # =============================edge deletion===================================
    if samplek_dict['del_k'] > 0:
        del_edge_weight, auxloss = sample4deletion(dat_batch,
                                                   deletion_logits,
                                                   train_forward if train else val_forward,
                                                   sampler_class,
                                                   samplek_dict,
                                                   ensemble,
                                                   directed_sampling,
                                                   auxloss,
                                                   auxloss_dict if train else None,
                                                   add_edge_weight.device)
    else:
        del_edge_weight = None

    new_graphs = graphs * (E * VE)

    if not separate:
        rewired_batch = Batch.from_data_list(new_graphs)
        merged_edge_index = torch.cat([rewired_batch.edge_index, add_edge_index], dim=1)
        if del_edge_weight is None:
            del_edge_weight = add_edge_weight.new_ones(L, dat_batch.edge_index.shape[1] * E * VE).squeeze(0)
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
            edge_weight=merged_edge_weight.t(),  # (#edges x E * VE) x L
            num_nodes=dat_batch.num_nodes * VE * E)
        merged_edge_weight = merged_edge_weight.t()   # L x (#edges x E * VE)

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
        original_edge_index = batch_repeat_edge_index(dat_batch.edge_index, dat_batch.num_nodes, E * VE)
        # for the graph adding with rewired edges in-place
        merged_edge_index = torch.cat([original_edge_index, add_edge_index], dim=1)
        merged_edge_weight = torch.cat([add_edge_weight.new_ones(L, original_edge_index.shape[1]).squeeze(0), add_edge_weight], dim=-1)
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
            edge_weight=merged_edge_weight.t(),  # (#edges x E * VE) x L
            num_nodes=dat_batch.num_nodes * VE * E)
        merged_edge_weight = merged_edge_weight.t()  # L x (#edges x E * VE)
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
            del_rewired_batch = Batch.from_data_list(new_graphs)
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, negative_sample)
            candidates.append(del_rewired_batch)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=candidates,
                                     y=add_rewired_batch.y,
                                     num_graphs=add_rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, None, auxloss


def construct_from_attention_mat(dat_batch: Data,
                                 graphs: List[Data],
                                 train: bool,
                                 output_logits: torch.Tensor,
                                 real_node_node_mask: torch.Tensor,

                                 ensemble: int,
                                 sample_policy: str,
                                 samplek_dict: Dict,
                                 directed_sampling: bool,
                                 auxloss_dict: ConfigDict,
                                 sampler_class,
                                 train_forward: Callable,
                                 val_forward: Callable,
                                 weight_edges: str,
                                 marginals_mask: bool,
                                 device: torch.device,
                                 include_original_graph: bool,
                                 negative_sample: str,
                                 in_place: bool,
                                 separate: bool):
    negative_sample = negative_sample if train else 'zero'

    padding_bias = (~real_node_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = output_logits - padding_bias

    auxloss = 0.
    if train and auxloss_dict is not None:
        if auxloss_dict.variance > 0:
            auxloss = auxloss + get_variance_regularization(output_logits,
                                                            auxloss_dict.variance,
                                                            real_node_node_mask)

    # (#sampled, B, N, N, E), (B, N, N, E)
    # ==================add edges from the N x N matrix============================
    sampler_class.policy = sample_policy
    sampler_class.k = samplek_dict['add_k']
    original_adj, self_loop_adj = batched_edge_index_to_batched_adj(dat_batch)
    sampler_class.adj = self_loop_adj
    node_mask, marginals = train_forward(logits) if train else val_forward(logits)
    VE, B, N, _, E = node_mask.shape
    E, L = ensemble, E // ensemble

    sampled_edge_weights = get_weighted_mask(weight_edges,
                                             node_mask,
                                             marginals,
                                             output_logits,
                                             marginals_mask or not train,
                                             VE)

    sampled_edge_weights = sampled_edge_weights.reshape(VE, B, N, N, E, L)
    # #edges x L x E x VE
    add_edge_weight = sampled_edge_weights.permute((1, 2, 3, 5, 4, 0))[real_node_node_mask]
    add_edge_weight = add_edge_weight.permute((1, 2, 3, 0)).reshape(L, -1).squeeze(0)

    # =============================edge deletion===================================
    if samplek_dict['del_k'] > 0:
        deletion_logits = output_logits[original_adj]
        if not directed_sampling:
            direct_mask = dat_batch.edge_index[0] <= dat_batch.edge_index[1]
            deletion_logits = deletion_logits[direct_mask]

        del_edge_weight, auxloss = sample4deletion(dat_batch,
                                                   deletion_logits,
                                                   train_forward if train else val_forward,
                                                   sampler_class,
                                                   samplek_dict,
                                                   ensemble,
                                                   directed_sampling,
                                                   auxloss,
                                                   auxloss_dict if train else None,
                                                   add_edge_weight.device)
    else:
        del_edge_weight = None

    new_graphs = [g.clone() for g in graphs]
    for g in new_graphs:
        g.edge_index = torch.from_numpy(np.vstack(np.triu_indices(g.num_nodes, k=-g.num_nodes))).to(device)
        del g.edge_attr

    new_graphs = new_graphs * (E * VE)
    rewired_batch = Batch.from_data_list(new_graphs)

    original_edge_index = dat_batch.edge_index
    original_edge_index = batch_repeat_edge_index(original_edge_index, dat_batch.num_nodes, E * VE)
    original_num_edges = dat_batch.nedges.to(original_edge_index.device).repeat(E * VE)

    if in_place:
        if separate:
            # for the graph adding with rewired edges in-place
            merged_edge_index = torch.cat(
                [original_edge_index,
                 rewired_batch.edge_index],
                dim=1)
            merged_edge_weight = torch.cat(
                [add_edge_weight.new_ones(L, original_edge_index.shape[1]).squeeze(0),
                 add_edge_weight],
                dim=-1)
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
                edge_weight=merged_edge_weight.t(),
                num_nodes=dat_batch.num_nodes * VE * E)
            merged_edge_weight = merged_edge_weight.t()

            rewired_batch.edge_index = merged_edge_index
            rewired_batch.edge_attr = merged_edge_attr

            # inc dict
            original_num_edges = dat_batch.nedges.repeat(E * VE)
            new_num_edges = (dat_batch.nnodes ** 2).repeat(E * VE)
            rewired_batch._slice_dict['edge_index'] = torch.hstack([original_edge_index.new_zeros(1), (original_num_edges + new_num_edges).cumsum(dim=0)])

            add_rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, negative_sample)
            candidates = [add_rewired_batch]

            if del_edge_weight is not None:
                del_rewired_batch = Batch.from_data_list(graphs * (E * VE))
                del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, negative_sample)
                candidates.append(del_rewired_batch)

            new_batch = DuoDataStructure(
                org=dat_batch if include_original_graph else None,
                candidates=candidates,
                y=add_rewired_batch.y,
                num_graphs=add_rewired_batch.num_graphs,
                num_unique_graphs=len(graphs))
            return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
        else:
            merged_edge_index = torch.cat([original_edge_index, rewired_batch.edge_index], dim=1)
            if del_edge_weight is None:
                del_edge_weight = add_edge_weight.new_ones(L, original_edge_index.shape[1]).squeeze(0)
            merged_edge_weight = torch.cat([del_edge_weight, add_edge_weight], dim=-1)
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
                edge_weight=merged_edge_weight.t(),
                num_nodes=dat_batch.num_nodes * VE * E)
            merged_edge_weight = merged_edge_weight.t()
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
        # only for the tree dataset, supposed no edge deletion
        assert separate
        if dat_batch.edge_attr is not None:
            new_edge_index = rewired_batch.edge_index
            new_edge_index_id = (new_edge_index[0] * rewired_batch.num_nodes + new_edge_index[1]).cpu().numpy()
            original_edge_index_id = (
                        original_edge_index[0] * rewired_batch.num_nodes + original_edge_index[1]).cpu().numpy()
            new_edge_attr = dat_batch.edge_attr.new_zeros(new_edge_index.shape[1], dat_batch.edge_attr.shape[1])
            fill_idx = np.in1d(new_edge_index_id, original_edge_index_id)
            new_edge_attr[fill_idx] = dat_batch.edge_attr.repeat(E * VE, 1)
            rewired_batch.edge_attr = new_edge_attr

        rewired_batch = sparsify_edge_weight(rewired_batch, add_edge_weight, negative_sample)

        candidates = [rewired_batch]

        if del_edge_weight is not None:
            del_rewired_batch = Batch.from_data_list(graphs * (E * VE))
            del_rewired_batch = sparsify_edge_weight(del_rewired_batch, del_edge_weight, negative_sample)
            candidates.append(del_rewired_batch)

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=candidates,
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs,
                                     num_unique_graphs=len(graphs))
        return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
