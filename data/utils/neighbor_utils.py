from heapq import heappop, heappush
from typing import List
from numba.typed import List as TypedList
import numba
import numpy as np


@numba.njit(cache=True)
def edgeindex2neighbordict(edge_index, num_nodes: int) -> TypedList[TypedList[int]]:
    """

    :param edge_index: shape (2, E)
    :param num_nodes:
    :return:
    """
    neighbors = TypedList([TypedList([-1]) for _ in range(num_nodes)])
    for i, node in enumerate(edge_index[0]):
        neighbors[node].append(edge_index[1, i])

    for i, n in enumerate(neighbors):
        n.pop(0)
    return neighbors


@numba.njit(cache=True)
def get_2wl_local_neighbors(edge_index: np.ndarray, num_nodes: int, neighbordict: TypedList[TypedList[int]] = None):
    if neighbordict is None:
        neighbordict = edgeindex2neighbordict(edge_index, num_nodes)
    edge_index1 = [(0, 0)]
    edge_index2 = [(0, 0)]
    edge_index1.pop()
    edge_index2.pop()

    for i in range(num_nodes):
        for j in range(num_nodes):
            cur_node = i * num_nodes + j
            for nv in neighbordict[j]:
                edge_index1.append((cur_node, i * num_nodes + nv))
            for nv in neighbordict[i]:
                edge_index2.append((cur_node, nv * num_nodes + j))

    return edge_index1, edge_index2


@numba.njit(cache=True)
def get_khop_neighbors(root: int, neighbor_dict: List, k: int):
    neighbors = [[root]]
    close_list = {-1}

    neighbors[0].pop()
    h = [(0, root)]

    while len(h):
        l, node = heappop(h)
        if l > k:
            break

        if node in close_list:
            continue

        if l < len(neighbors):
            neighbors[l].append(node)
        else:
            neighbors.append([node])

        for i, n in enumerate(neighbor_dict[node]):
            if n not in close_list:
                heappush(h, (l + 1, n))

        close_list.add(node)

    merged_neighbors = neighbors[0]
    for i in range(1, len(neighbors)):
        merged_neighbors.extend(neighbors[i])

    return neighbors, merged_neighbors


@numba.njit(cache=True)
def get_subgraphs_nodeidx(edge_candidate, neighbor_dict, k):
    n1, n2 = edge_candidate
    _, neighbors1 = get_khop_neighbors(n1, neighbor_dict, k)
    _, neighbors2 = get_khop_neighbors(n2, neighbor_dict, k)
    neighbors = set(neighbors1)
    neighbors.update(neighbors2)
    return neighbors


@numba.njit(cache=True, parallel=True)
def batch_get_subgraphs_nodeidx(edge_candidates, edge_index, num_nodes, k):
    n_dict = edgeindex2neighbordict(edge_index, num_nodes)
    subgraph_indices = [{0, 1}] * edge_candidates.shape[0]
    for i in numba.prange(edge_candidates.shape[0]):
        neighbors = get_subgraphs_nodeidx(edge_candidates[i], n_dict, k)
        subgraph_indices[i] = neighbors
    return subgraph_indices
