import torch


def sparsify_edge_weight(data, edge_weight, negative_sample, negative_sample_size = None, device = None):
    if negative_sample == 'zero':
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        data.edge_index = data.edge_index[:, nonzero_idx]
        data.edge_weight = edge_weight[nonzero_idx]
    elif negative_sample == 'full':
        data.edge_weight = edge_weight
    elif negative_sample == 'same':
        zero_idx = torch.where(edge_weight == 0.)[0]
        zero_idx = zero_idx[torch.randint(low=0, high=zero_idx.shape[0],
                                          size=(negative_sample_size,),
                                          device=device)]
        nonzero_idx = edge_weight.nonzero().reshape(-1)
        idx = torch.cat((zero_idx, nonzero_idx), dim=0).unique()
        data.edge_index = data.edge_index[:, idx]
        data.edge_weight = edge_weight[idx]
    else:
        raise ValueError
    return data
