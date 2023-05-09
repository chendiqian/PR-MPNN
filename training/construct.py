from typing import Tuple, List, Callable

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected, coalesce
from torch_scatter import scatter

from data.data_utils import DuoDataStructure, batched_edge_index_to_batched_adj
from training.aux_loss import get_variance_regularization, get_original_bias

LARGE_NUMBER = 1.e10


def sparsify_edge_weight(data, edge_weight, negative_sample):
    device = edge_weight.device
    if negative_sample == 'zero':
        edge_ptr = data._slice_dict['edge_index'].to(device)
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        data.edge_index = data.edge_index[:, nonzero_idx]
        data.edge_weight = edge_weight[nonzero_idx]
        new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                      torch.arange(len(edge_ptr) - 1, device=device).repeat_interleave(
                                          edge_ptr[1:] - edge_ptr[:-1]), reduce='sum', dim_size=len(edge_ptr) - 1)
        data._slice_dict['edge_index'] = torch.cumsum(
            torch.hstack([new_edges_per_graph.new_zeros(1), new_edges_per_graph]), dim=0)
    elif negative_sample == 'full':
        data.edge_weight = edge_weight
    elif negative_sample == 'same':
        # zero_idx = torch.where(edge_weight == 0.)[0]
        # zero_idx = zero_idx[torch.randint(low=0, high=zero_idx.shape[0],
        #                                   size=(negative_sample_size,),
        #                                   device=device)]
        # nonzero_idx = edge_weight.nonzero().reshape(-1)
        # idx = torch.cat((zero_idx, nonzero_idx), dim=0).unique()
        # data.edge_index = data.edge_index[:, idx]
        # data.edge_weight = edge_weight[idx]
        raise NotImplementedError
    else:
        raise ValueError
    return data


def construct_from_edge_candidates(collate_data: Tuple[Data, List[Data]],
                                   emb_model: Callable,
                                   train_forward: Callable,
                                   val_forward: Callable,
                                   weight_edges: str,
                                   marginals_mask: bool,
                                   include_original_graph: bool,
                                   negative_sample: str,
                                   in_place: bool = True,
                                   auxloss_dict: ConfigDict = None):
    dat_batch, graphs = collate_data

    train = emb_model.training
    # (sum_edges, ensemble)
    output_logits, edge_candidate_idx = emb_model(dat_batch)

    # (B x Nmax x E) (B x Nmax)
    output_logits, real_node_mask = to_dense_batch(output_logits,
                   torch.arange(len(dat_batch.nnodes), device=output_logits.device).repeat_interleave(
                       dat_batch.num_edge_candidate),
                   max_num_nodes=dat_batch.num_edge_candidate.max())

    padding_bias = (~real_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = output_logits - padding_bias

    auxloss = 0.
    if train and auxloss_dict is not None:
        raise NotImplementedError

    # (#sampled, B, Nmax, E), (B, Nmax, E)
    node_mask, marginals = train_forward(logits) if train else val_forward(logits)
    VE, B, N, E = node_mask.shape

    if weight_edges == 'marginals':
        # Maybe we should also try this with softmax?
        sampled_edge_weights = marginals[None].repeat(VE, 1, 1, 1)
    elif weight_edges == 'None' or weight_edges is None:
        sampled_edge_weights = node_mask
    else:
        raise ValueError(f"{weight_edges} not supported")

    if marginals_mask or not train:
        sampled_edge_weights = sampled_edge_weights * node_mask

    # num_edges x E x VE
    edge_weight = sampled_edge_weights.permute((1, 2, 3, 0))[real_node_mask].reshape(-1, E * VE)

    edge_index, edge_weight = to_undirected(edge_candidate_idx.T, edge_weight, num_nodes=dat_batch.num_nodes)
    if E * VE > 1:
        edge_index_rel = (torch.arange(E * VE, device=edge_weight.device) * dat_batch.num_nodes).repeat_interleave(edge_index.shape[1])
        edge_index = edge_index.repeat(1, E * VE)
        edge_index += edge_index_rel

    new_graphs = graphs * (E * VE)
    edge_weight = edge_weight.T.flatten()

    if in_place:
        original_edge_index = dat_batch.edge_index
        original_num_edges = (dat_batch._slice_dict['edge_index'][1:] - dat_batch._slice_dict['edge_index'][:-1]).to(edge_index.device)
        if E * VE > 1:
            edge_index_rel = (torch.arange(E * VE, device=edge_weight.device) *
                              dat_batch.num_nodes).repeat_interleave(original_edge_index.shape[1])
            original_edge_index = original_edge_index.repeat(1, E * VE)
            original_edge_index += edge_index_rel
        merged_edge_index = torch.cat([original_edge_index, edge_index], dim=1)
        merged_edge_weight = torch.cat([edge_weight.new_ones(original_edge_index.shape[1]), edge_weight], dim=0)

        merged_edge_index, merged_edge_weight = coalesce(edge_index=merged_edge_index,
                                                         edge_attr=merged_edge_weight,
                                                         num_nodes=dat_batch.num_nodes * VE * E)
        rewired_batch = Batch.from_data_list(new_graphs)
        rewired_batch.edge_index = merged_edge_index

        # inc dict
        original_num_edges = original_num_edges.repeat(E * VE)
        new_num_edges = (dat_batch.num_edge_candidate * 2).repeat(E * VE)
        rewired_batch._slice_dict['edge_index'] = torch.hstack([edge_index.new_zeros(1),
                                                                (original_num_edges + new_num_edges).cumsum(dim=0)])

        if train:
            rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, negative_sample)
        else:
            rewired_batch = sparsify_edge_weight(rewired_batch, merged_edge_weight, 'zero')

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs)
        return new_batch, None, auxloss
    else:
        rewired_batch = Batch.from_data_list(new_graphs)
        rewired_batch.edge_index = edge_index
        rewired_batch._slice_dict['edge_index'] = torch.hstack([edge_index.new_zeros(1),
                                                                (dat_batch.num_edge_candidate * 2).repeat(E * VE).cumsum(dim=0)])

        if train:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, negative_sample)
        else:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, 'zero')

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs)
        return new_batch, None, auxloss


def construct_from_attention_mat(collate_data: Tuple[Data, List[Data]],
                                 emb_model: Callable,
                                 sample_policy: str,
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
    output_logits, real_node_node_mask = emb_model(dat_batch)

    if sample_policy == 'global_topk_semi' or (train and auxloss_dict is not None and auxloss_dict.origin_bias > 0.):
        # need to compute the dense adj matrix
        adj = batched_edge_index_to_batched_adj(dat_batch, torch.float)
        sampler_class.adj = adj

    padding_bias = (~real_node_node_mask)[..., None].to(torch.float) * LARGE_NUMBER
    logits = output_logits - padding_bias

    auxloss = 0.
    if train and auxloss_dict is not None:
        if auxloss_dict.degree > 0:
            raise NotImplementedError
            # auxloss = auxloss + get_degree_regularization(node_mask, self.auxloss.degree, real_node_node_mask)
        if auxloss_dict.variance > 0:
            auxloss = auxloss + get_variance_regularization(logits,
                                                            auxloss_dict.variance,
                                                            real_node_node_mask)
        if auxloss_dict.origin_bias > 0.:
            auxloss = auxloss + get_original_bias(adj, logits,
                                                  auxloss_dict.origin_bias,
                                                  real_node_node_mask)

    # (#sampled, B, N, N, E), (B, N, N, E)
    node_mask, marginals = train_forward(logits) if train else val_forward(logits)
    VE, B, N, _, E = node_mask.shape

    if weight_edges == 'logits':
        # (#sampled, B, N, N, E)
        # sampled_edge_weights = torch.vmap(
        #     torch.vmap(
        #         torch.vmap(
        #             self_defined_softmax,
        #             in_dims=(None, 0),
        #             out_dims=0),
        #         in_dims=0, out_dims=0),
        #     in_dims=-1,
        #     out_dims=-1)(logits, node_mask)
        sampled_edge_weights = logits
    elif weight_edges == 'marginals':
        assert marginals is not None
        # Maybe we should also try this with softmax?
        sampled_edge_weights = marginals[None].repeat(node_mask.shape[0], 1, 1, 1, 1)
    elif weight_edges == 'None' or weight_edges is None:
        sampled_edge_weights = node_mask
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
    new_graphs = new_graphs * (E * VE)

    if in_place:
        # if include_original_graph:
        #     new_graphs += graphs * (E * VE)
        #     edge_weight = torch.cat(
        #         [edge_weight, edge_weight.new_ones(VE * E * dat_batch.num_edges)],
        #         dim=0)
        #
        # new_batch = Batch.from_data_list(new_graphs)
        # new_batch.y = new_batch.y[:B * E * VE]
        # new_batch.inter_graph_idx = torch.arange(B * E * VE).to(device).repeat(1 + int(include_original_graph))
        #
        # if train:
        #     new_batch = sparsify_edge_weight(new_batch, edge_weight, negative_sample)
        # else:
        #     new_batch = sparsify_edge_weight(new_batch, edge_weight, 'zero')
        # return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
        raise NotImplementedError
    else:
        rewired_batch = Batch.from_data_list(new_graphs)

        if train:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, negative_sample)
        else:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, 'zero')

        new_batch = DuoDataStructure(org=dat_batch if include_original_graph else None,
                                     candidates=[rewired_batch],
                                     y=rewired_batch.y,
                                     num_graphs=rewired_batch.num_graphs)
        return new_batch, output_logits.detach() * real_node_node_mask[..., None], auxloss
