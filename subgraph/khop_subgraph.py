from typing import Tuple, Optional, List
import numpy as np
import numba
from data.data_utils import edgeindex2neighbordict


@numba.njit(cache=True, locals={'edge_mask': numba.bool_[::1], 'np_node_idx': numba.int64[::1]})
def numba_k_hop_subgraph(edge_index: np.ndarray,
                         seed_node: int,
                         khop: int,
                         num_nodes: int,
                         relabel: bool,
                         neighbor_dict: Optional[List[np.ndarray]] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k_hop_subgraph of PyG is too slow

    :param edge_index:
    :param seed_node:
    :param khop:
    :param num_nodes:
    :param relabel:
    :param neighbor_dict:
    :return:
    """
    node_idx = {seed_node}
    visited_nodes = {seed_node}

    if neighbor_dict is None:
        neighbor_dict = edgeindex2neighbordict(edge_index, num_nodes)

    cur_neighbors = {seed_node}
    for hop in range(khop):
        last_neighbors, cur_neighbors = cur_neighbors, set()

        for node in last_neighbors:
            for neighbor in neighbor_dict[node]:
                if neighbor not in visited_nodes:
                    cur_neighbors.add(neighbor)
                    visited_nodes.add(neighbor)
        node_idx.update(cur_neighbors)

    edge_mask = np.zeros(edge_index.shape[1], dtype=np.bool_)
    for i in range(edge_index.shape[1]):
        if edge_index[0][i] in node_idx and edge_index[1][i] in node_idx:
            edge_mask[i] = True

    edge_index = edge_index[:, edge_mask]
    node_idx = sorted(list(node_idx))

    if relabel:
        new_idx_dict = dict()
        for i, idx in enumerate(node_idx):
            new_idx_dict[idx] = i

        for i in range(2):
            for j in range(edge_index.shape[1]):
                edge_index[i, j] = new_idx_dict[edge_index[i, j]]

    np_node_idx = np.array(node_idx, dtype=np.int64)
    return np_node_idx, edge_index, edge_mask


@numba.njit(cache=True, parallel=False)
def parallel_khop_neighbor(edge_index: np.ndarray,
                           num_nodes: int,
                           khop: int):
    """

    :param edge_index:
    :param num_nodes:
    :param khop:
    :return:
    """
    masks = np.zeros((num_nodes, num_nodes), dtype=np.bool_)

    neighbor_dict = edgeindex2neighbordict(edge_index, num_nodes)

    for i in range(num_nodes):
        node_idx, _, _ = numba_k_hop_subgraph(edge_index, numba.int64(i), khop, num_nodes, False, neighbor_dict)
        for n in node_idx:
            masks[i, n] = True
    return masks
