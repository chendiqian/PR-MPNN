from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from numba import jit, prange
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    is_undirected,
    to_networkx,
)


@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0:
                C[i, j] = 0
                continue

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                    (2 / d_max)
                    + (2 / d_min)
                    - 2
                    + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C


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


def get_forman_curvature(data: nx.Graph):
    frc = FormanRicci(data)
    frc.compute_ricci_curvature()
    edges = list(data.edges)
    curvatures = [frc.G[e[0]][e[1]]['formanCurvature'] for e in edges]
    return curvatures


def get_ollivier_curvature(data: nx.Graph):
    orc = OllivierRicci(data, alpha=0.5, verbose="ERROR")
    edges = list(data.edges)
    curv_dict = orc.compute_ricci_curvature_edges(edges)
    return list(curv_dict.values())


def get_connectedness_metric(data: Batch, metric: str = 'eigval'):
    assert isinstance(data.edge_index, torch.Tensor)

    if metric == 'all':
        metrics = ['forman_curvature',
                   'ollivier_curvature',
                   'balanced_forman',
                   'eigval']
    else:
        metrics = [metric.lower()]

    metrics_dict = defaultdict(list)

    graphs = Batch.to_data_list(data)
    nx_graphs = [to_networkx(g, to_undirected=True, remove_self_loops=True) for g in graphs]

    if 'eigval' in metrics:
        metrics_dict['eigval'] = [get_second_smallest_eigval(g) for g in graphs]

    if 'forman_curvature' in metrics:
        for g in nx_graphs:
            cur_metric = get_forman_curvature(g)
            metrics_dict['forman_curvature'].extend(cur_metric)
            metrics_dict['smallest_forman_curvature'].append(min(cur_metric))

    if 'ollivier_curvature' in metrics:
        for g in nx_graphs:
            cur_metric = get_ollivier_curvature(g)
            metrics_dict['ollivier_curvature'].extend(cur_metric)
            metrics_dict['smallest_ollivier_curvature'].append(min(cur_metric))

    if 'balanced_forman' in metrics:
        for g in graphs:
            A = np.zeros(shape=(g.num_nodes, g.num_nodes), dtype=np.float64)
            edge_index = g.edge_index.cpu().numpy()
            A[edge_index[0], edge_index[1]] = 1.

            cur_metric = balanced_forman_curvature(A).flatten().tolist()
            metrics_dict['balanced_forman'].extend(cur_metric)
            metrics_dict['smallest_balanced_forman'].append(min(cur_metric))

    return metrics_dict
