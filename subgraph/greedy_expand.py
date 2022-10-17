from heapq import heappop, heappush
import numba
import numpy as np
import torch
from torch_geometric.data import Data
from data.data_utils import edgeindex2neighbordict


@numba.njit(cache=True)
def numba_greedy_grow_tree(num_neighbors: int, weights: np.ndarray, edge_index: np.ndarray, num_nodes: int, ):
    neighbor_dict = edgeindex2neighbordict(edge_index, num_nodes)
    mask = np.zeros((num_nodes, num_nodes), dtype=np.bool_)

    for seed_node in range(num_nodes):
        search_list = [(-weights[seed_node, seed_node], seed_node)]
        closelist = set()

        while mask[seed_node, :].sum() < num_neighbors:
            _, cur_node = heappop(search_list)
            mask[seed_node, cur_node] = True
            closelist.add(cur_node)
            for n in neighbor_dict[cur_node]:
                if n not in closelist:
                    heappush(search_list, (-weights[seed_node, n], n))

    return mask


def greedy_grow_tree(graph: Data, num_neighbors: int, weights: torch.Tensor) -> torch.Tensor:
    if num_neighbors >= graph.num_nodes:
        return torch.ones_like(weights, device=weights.device, dtype=torch.float)

    np_mask = numba_greedy_grow_tree(num_neighbors,
                                     weights.cpu().detach().numpy(),
                                     graph.edge_index.cpu().numpy(),
                                     graph.num_nodes, )
    return torch.from_numpy(np_mask).to(torch.float).to(weights.device)
