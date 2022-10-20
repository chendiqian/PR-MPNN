from typing import Optional
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, is_undirected
from subgraph.greedy_expand import greedy_grow_tree
from subgraph.khop_subgraph import parallel_khop_neighbor


class GraphModification:
    """
    Base class, augmenting each graph with some features
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> Optional[Data]:
        return None


class GraphToUndirected(GraphModification):
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if is_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes):
            return graph

        if graph.edge_attr is not None:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
        else:
            edge_index = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
            edge_attr = None
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


class GraphCoalesce(GraphModification):

    def __call__(self, graph: Data):
        if graph.edge_attr is None:
            edge_index = coalesce(graph.edge_index, None, num_nodes=graph.num_nodes)
            edge_attr = None
        else:
            edge_index, edge_attr = coalesce(graph.edge_index, graph.edge_attr, graph.num_nodes)
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


class AugmentwithNNodes(GraphModification):

    def __call__(self, graph: Data):
        graph.nnodes = torch.tensor([graph.num_nodes])
        return graph


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


class AugmentWithKhopMasks(GraphModification):
    """
    Should be used as pretransform, because it is deterministic
    """
    def __init__(self, k: int):
        super(AugmentWithKhopMasks, self).__init__()
        self.khop = k

    def __call__(self, graph: Data):
        np_mask = parallel_khop_neighbor(graph.edge_index.numpy(), graph.num_nodes, self.khop)
        graph.node_mask = torch.from_numpy(np_mask).reshape(-1).to(torch.bool)
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
    elif policy == 'khop':
        return AugmentWithKhopMasks(sample_k)
