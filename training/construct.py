from typing import Tuple, List, Callable

import torch
from ml_collections import ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_scatter import scatter

from data.data_utils import DuoDataStructure

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
                                   emb_model,
                                   train_forward: Callable,
                                   val_forward: Callable,
                                   weight_edges: str,
                                   marginals_mask: bool,
                                   include_original_graph: bool,
                                   negative_sample: str,
                                   merge_original_graph: bool = True,
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

    if merge_original_graph:
        raise NotImplementedError
    else:
        assert include_original_graph
        rewired_batch = Batch.from_data_list(new_graphs)
        rewired_batch.edge_index = edge_index
        rewired_batch._slice_dict['edge_index'] = torch.hstack([edge_index.new_zeros(1),
                                                                (dat_batch.num_edge_candidate * 2).repeat(E * VE).cumsum(dim=0)])
        original_batch = Batch.from_data_list(graphs * (E * VE))

        if train:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, negative_sample)
        else:
            rewired_batch = sparsify_edge_weight(rewired_batch, edge_weight, 'zero')

        new_batch = DuoDataStructure(data1=rewired_batch, data2=original_batch, y=rewired_batch.y, num_graphs=rewired_batch.num_graphs)
        return new_batch, None, auxloss
