from typing import Optional
import torch
from torch_geometric.data import Data

from subgraph.greedy_expand import greedy_grow_tree


class GraphModification:
    """
    Base class, augmenting each graph with some features
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> Optional[Data]:
        return None


class AugmentWithRandomKNeighbors(GraphModification):
    """
    Sample best k neighbors randomly, return the induced subgraph
    Serves as transform because of its randomness
    """
    def __init__(self, sample_k: int):
        super(AugmentWithRandomKNeighbors, self).__init__()
        self.num_neighnors = sample_k

    def __call__(self, graph: Data):
        mask = greedy_grow_tree(graph,
                                self.num_neighnors,
                                torch.rand(graph.num_nodes, graph.num_nodes, device=graph.x.device),
                                target_dtype=torch.bool)
        graph.node_mask = mask.reshape(-1)
        return graph


def policy2transform(policy: str, sample_k: int) -> GraphModification:
    """
    transform for datasets

    :param policy:
    :param sample_k:
    :return:
    """
    if policy == 'greedy_neighbors':
        return AugmentWithRandomKNeighbors(sample_k)
