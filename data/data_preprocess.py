import random
from math import ceil
from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch
from torch_geometric.data.collate import collate
from torch_geometric.utils import is_undirected, to_undirected, add_remaining_self_loops, \
    coalesce, subgraph as pyg_subgraph
from torch_sparse import SparseTensor

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


class GraphAddRemainSelfLoop(GraphModification):
    def __call__(self, graph: Data):
        edge_index, edge_attr = add_remaining_self_loops(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
        graph.edge_index = edge_index
        if graph.edge_attr is not None:
            graph.edge_attr = edge_attr
        return graph


class GraphAttrToOneHot(GraphModification):
    def __init__(self, num_node_classes, num_edge_classes):
        super(GraphAttrToOneHot, self).__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

    def __call__(self, graph: Data):
        assert graph.x.dtype == torch.long
        assert graph.edge_attr.dtype == torch.long

        graph.x = torch.nn.functional.one_hot(graph.x.squeeze(), self.num_node_classes).to(torch.float)
        graph.edge_attr = torch.nn.functional.one_hot(graph.edge_attr.squeeze(), self.num_edge_classes).to(torch.float)

        return graph


class GraphExpandDim(GraphModification):
    def __call__(self, graph: Data):
        if graph.y.ndim == 1:
            graph.y = graph.y[None]
        if graph.edge_attr is not None and graph.edge_attr.ndim == 1:
            graph.edge_attr = graph.edge_attr[:, None]
        return graph


class GraphToUndirected(GraphModification):
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if not is_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes):
            if graph.edge_attr is not None:
                edge_index, edge_attr = to_undirected(graph.edge_index,
                                                      graph.edge_attr,
                                                      graph.num_nodes)
            else:
                edge_index = to_undirected(graph.edge_index,
                                           graph.edge_attr,
                                           graph.num_nodes)
                edge_attr = None
        else:
            if graph.edge_attr is not None:
                edge_index, edge_attr = coalesce(graph.edge_index,
                                                 graph.edge_attr,
                                                 graph.num_nodes)
            else:
                edge_index = coalesce(graph.edge_index,
                                      graph.edge_attr,
                                      graph.num_nodes)
                edge_attr = None
        new_data = Data(x=graph.x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=graph.y,
                        num_nodes=graph.num_nodes)
        for k, v in graph:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v
        return new_data


class AugmentwithNNodes(GraphModification):

    def __call__(self, graph: Data):
        graph.nnodes = torch.tensor([graph.num_nodes])
        return graph


class AugmentWithRandomKNeighbors(GraphModification):
    """
    Sample best k neighbors randomly, return the induced subgraph
    Serves as transform because of its randomness
    """

    def __init__(self, sample_k: int, ensemble: int):
        super(AugmentWithRandomKNeighbors, self).__init__()
        self.num_neighnors = sample_k
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        mask = greedy_grow_tree(graph,
                                self.num_neighnors,
                                torch.rand(graph.num_nodes, graph.num_nodes, self.ensemble, device=graph.x.device),
                                target_dtype=torch.bool)
        graph.node_mask = mask.reshape(graph.num_nodes ** 2, self.ensemble)
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


class AugmentWithShortedPathDistance(GraphModification):
    def __init__(self, max_num_nodes):
        super(AugmentWithShortedPathDistance, self).__init__()
        self.max_num_nodes = max_num_nodes

    def __call__(self, graph: Data):
        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        edge_index = graph.edge_index.numpy()
        mat = csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                         shape=(graph.num_nodes, graph.num_nodes))

        g_dist_mat = torch.zeros(graph.num_nodes, self.max_num_nodes, dtype=torch.float)
        g_dist_mat[:, :graph.num_nodes] = torch.from_numpy(shortest_path(mat, directed=False, return_predecessors=False, ))
        g_dist_mat[torch.isinf(g_dist_mat)] = 0.
        g_dist_mat /= g_dist_mat.max() + 1

        graph.g_dist_mat = g_dist_mat
        return graph


class AugmentWithPPR(GraphModification):
    def __init__(self, max_num_nodes, alpha = 0.2, iters = 20):
        super(AugmentWithPPR, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.alpha = alpha
        self.iters = iters

    def __call__(self, graph: Data):
        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        adj = SparseTensor.from_edge_index(graph.edge_index).to_dense()
        deg = adj.sum(0)

        # deg_inv_sqrt = deg.pow_(-0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        # adj = adj * deg_inv_sqrt.reshape(-1, 1) * deg_inv_sqrt.view(1, -1)

        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj = adj * deg_inv.view(1, -1)

        r = torch.eye(graph.num_nodes, dtype=torch.float)
        topics = torch.eye(graph.num_nodes, dtype=torch.float)

        for i in range(self.iters):
            r = (1 - self.alpha) * adj @ r + self.alpha * topics

        ppr_mat = torch.zeros(graph.num_nodes, self.max_num_nodes, dtype=torch.float)
        ppr_mat[:, :graph.num_nodes] = r
        graph.ppr_mat = ppr_mat
        return graph


class RandomSampleTopk(GraphModification):
    def __init__(self, k: int, ensemble: int):
        super(RandomSampleTopk, self).__init__()
        self.k = k
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        if isinstance(self.k, float):
            k = int(ceil(self.k * graph.num_nodes))
        elif isinstance(self.k, int):
            k = self.k
        else:
            raise TypeError

        if k >= graph.num_nodes:
            return collate(graph.__class__, [graph] * self.ensemble, increment=True, add_batch=False)[0]

        data_list = []
        for c in range(self.ensemble):
            mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
            mask[graph.target_mask] = True
            candidates = random.sample(range(graph.num_nodes), k)
            mask[candidates] = True
            edge_index, edge_attr = pyg_subgraph(subset=mask,
                                                 edge_index=graph.edge_index,
                                                 edge_attr=graph.edge_attr,
                                                 relabel_nodes=True,
                                                 num_nodes=graph.num_nodes)
            new_data = Data(x=graph.x[mask],
                            y=graph.y,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            # num_nodes=mask.sum(),
                            target_mask=graph.target_mask[mask])
            for k, v in graph:
                if k not in ['x', 'edge_index', 'edge_attr', 'num_nodes', 'target_mask', 'y']:
                    new_data[k] = v
            data_list.append(new_data)

        return collate(graph.__class__, data_list, increment=True, add_batch=False)[0]


class RandomSampleTopkperNode(GraphModification):
    def __init__(self, k: int, ensemble: int):
        super(RandomSampleTopkperNode, self).__init__()
        self.k = k
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        # if isinstance(self.k, float):
        #     k = int(ceil(self.k * graph.num_nodes))
        if isinstance(self.k, int):
            k = self.k
        else:
            raise TypeError

        if k >= graph.num_nodes:
            graph.node_mask = torch.ones(graph.num_nodes ** 2, self.ensemble, dtype=torch.bool)
        else:
            logit = torch.rand(graph.num_nodes, graph.num_nodes, self.ensemble)
            thresh = torch.topk(logit, k, dim=1, largest=True, sorted=True).values[:, -1, :]
            mask = (logit >= thresh[:, None, :])
            graph.node_mask = mask.reshape(graph.num_nodes ** 2, self.ensemble)
        return graph


class AugmentTopkSampledGraphsPerNode(GraphModification):
    def __init__(self, k: int, ensemble: int, subgraph2node_aggr: str = 'mean', rm_nodes: bool = True):
        super(AugmentTopkSampledGraphsPerNode, self).__init__()
        assert isinstance(k, int)
        self.k = k
        self.ensemble = ensemble
        self.rm_nodes = rm_nodes

        if subgraph2node_aggr == 'center':
            raise ValueError("center does not make sense, since center node not necessarily selected")
        elif subgraph2node_aggr not in ['mean', 'sum', 'add']:
            raise NotImplementedError

    def __call__(self, graph: Data):

        if self.k >= graph.num_nodes:
            node_mask = torch.ones(self.ensemble, graph.num_nodes, graph.num_nodes, dtype=torch.bool)
        else:
            logit = torch.rand(self.ensemble, graph.num_nodes, graph.num_nodes)
            thresh = torch.topk(logit, self.k, dim=2, largest=True, sorted=True).values[:, :, -1]
            node_mask = (logit >= thresh[:, :, None])

        graphs = [graph] * self.ensemble * graph.num_nodes
        full_graph_batch = Batch.from_data_list(graphs)
        flat_node_mask = node_mask.reshape(-1)    # ensemble1node1, ensemble1node2, ensemble1node3, ensemble2node1, ......

        if self.rm_nodes:
            new_edge_index, new_edge_attr = pyg_subgraph(flat_node_mask,
                                                         full_graph_batch.edge_index,
                                                         full_graph_batch.edge_attr,
                                                         relabel_nodes=True,
                                                         num_nodes=full_graph_batch.num_nodes)

            sampled_square_graph = Data(x=full_graph_batch.x[flat_node_mask],
                                        edge_index=new_edge_index,
                                        edge_attr=new_edge_attr,
                                        y=graph.y,)

            mask_node_2b_aggregated = torch.ones(node_mask.sum(), dtype=torch.bool)
            actual_sampled_nodes = node_mask.sum(2)   # ensemble, n_nodes
            subgraphs2nodes = torch.hstack([torch.repeat_interleave(torch.arange(len(ac)), ac) for ac in actual_sampled_nodes])
        else:
            *_, edge_mask = pyg_subgraph(flat_node_mask,
                                         full_graph_batch.edge_index,
                                         full_graph_batch.edge_attr,
                                         relabel_nodes=False,
                                         num_nodes=full_graph_batch.num_nodes,
                                         return_edge_mask=True)

            sampled_square_graph = Data(x=full_graph_batch.x,
                                        edge_index=full_graph_batch.edge_index[:, edge_mask],
                                        edge_attr=full_graph_batch.edge_attr[edge_mask],
                                        y=graph.y, )

            mask_node_2b_aggregated = torch.ones(node_mask.numel(), dtype=torch.bool)
            subgraphs2nodes = torch.repeat_interleave(torch.arange(graph.num_nodes),
                                                      graph.num_nodes).repeat(self.ensemble)

        sampled_square_graph.node_mask = mask_node_2b_aggregated
        sampled_square_graph.subgraphs2nodes = subgraphs2nodes

        return graph, sampled_square_graph


class AugmentFullGraphsPerNode(GraphModification):
    def __init__(self, ensemble: int):
        super(AugmentFullGraphsPerNode, self).__init__()
        self.ensemble = ensemble

    def __call__(self, graph: Data):

        graphs = [graph] * self.ensemble * graph.num_nodes
        full_graph_batch = Batch.from_data_list(graphs)

        square_graph = Data(x=full_graph_batch.x,
                            edge_index=full_graph_batch.edge_index,
                            edge_attr=full_graph_batch.edge_attr,
                            y=graph.y, )

        return graph, square_graph


def policy2transform(policy: str, sample_k: int, ensemble: int = 1, subgraph2node_aggr: str = 'mean') -> GraphModification:
    """
    transform for datasets

    :param policy:
    :param sample_k:
    :param ensemble:
    :param subgraph2node_aggr:
    :return:
    """
    if policy == 'greedy_neighbors':
        # return AugmentWithRandomKNeighbors(sample_k, ensemble)
        raise NotImplementedError("should implement this with aug data in dataloader")
    elif policy == 'khop':
        return AugmentWithKhopMasks(sample_k)
    elif policy == 'topk':
        return RandomSampleTopk(sample_k, ensemble)
    elif policy == 'graph_topk':
        return AugmentTopkSampledGraphsPerNode(sample_k, ensemble, subgraph2node_aggr)
    else:
        raise NotImplementedError
