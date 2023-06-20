import numpy as np
import torch
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    is_undirected
)


def get_second_smallest_eigval(data: Data):
    assert isinstance(data.edge_index, torch.Tensor)

    graph_is_undirected = is_undirected(data.edge_index, data.edge_weight, data.num_nodes)

    eig_fn = eigs if not graph_is_undirected else eigsh

    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_vals = eig_fn(
        L,
        k=2,
        which='SR' if not graph_is_undirected else 'SA',
        return_eigenvectors=False,
    )

    return np.real(eig_vals)[0]


def get_connectedness_metric(data: Batch, metric: str = 'eigval'):
    assert isinstance(data.edge_index, torch.Tensor)

    graphs = Batch.to_data_list(data)
    if metric == 'eigval':
        metrics = [get_second_smallest_eigval(g) for g in graphs]
    else:
        raise NotImplementedError(f"{metric} not supported")

    return metrics
