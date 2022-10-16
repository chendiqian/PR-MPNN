import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


class GraphToUndirected:
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
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


class GraphCoalesce:

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


class AugmentwithNNodes:

    def __call__(self, graph: Data):
        graph.nnodes = torch.tensor([graph.num_nodes])
        return graph
