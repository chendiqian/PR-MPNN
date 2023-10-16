from heapq import heappop, heappush
from typing import List
from numba.typed import List as TypedList
import numba


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
