from typing import List
from heapq import heappush, heappop
import numba
import numpy as np


@numba.njit(cache=True)
def edgeindex2neighbordict(edge_index: np.ndarray, num_nodes: int) -> List[List[int]]:
    """

    :param edge_index: shape (2, E)
    :param num_nodes:
    :return:
    """
    neighbors = [[-1] for _ in range(num_nodes)]
    for i, node in enumerate(edge_index[0]):
        neighbors[node].append(edge_index[1][i])

    for i, n in enumerate(neighbors):
        n.pop(0)
    return neighbors


@numba.njit(cache=True)
def greedy_expand(seed_nodes, value_lst, neighborlst, nodes_per_node, num_nodes):
    ret_nodes = np.zeros(num_nodes, dtype=np.bool_)
    close_set = set()

    vals = [(np.float32(1.e8), -1)]
    heappush(vals, (-value_lst[seed_nodes], seed_nodes))

    while len(vals) and len(close_set) < nodes_per_node:
        _, cur_node = heappop(vals)
        close_set.add(cur_node)
        ret_nodes[cur_node] = True
        for neighbors in neighborlst[cur_node]:
            if neighbors not in close_set:
                heappush(vals, (-value_lst[neighbors], neighbors))

    return ret_nodes


@numba.njit(cache=True)
def batch_greedy_expand(value_lst, edge_index, nodes_per_node, num_nodes):
    neighborlst = edgeindex2neighbordict(edge_index, num_nodes)
    return_mask = np.zeros((num_nodes, num_nodes,), dtype=np.bool_)
    for i in range(num_nodes):
        return_mask[i] = greedy_expand(i, value_lst[i], neighborlst, nodes_per_node, num_nodes)
    return return_mask
