import torch
from torch_scatter import scatter


def sparsify_edge_weight(data, edge_weight, negative_sample):
    device = edge_weight.device
    if negative_sample == 'zero':
        edge_ptr = data._slice_dict['edge_index'].to(device)
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        data.edge_index = data.edge_index[:, nonzero_idx]
        data.edge_weight = edge_weight[nonzero_idx]
        new_edges_per_graph = scatter((edge_weight > 0.).long(),
                                      torch.arange(len(edge_ptr) - 1, device=device).repeat_interleave(
                                          edge_ptr[1:] - edge_ptr[:-1]), reduce='sum')
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
