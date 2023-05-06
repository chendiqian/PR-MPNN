from copy import deepcopy
from functools import reduce
from typing import Optional, List

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (is_undirected,
                                   to_undirected,
                                   add_remaining_self_loops,
                                   contains_self_loops,
                                   coalesce,
                                   to_networkx)
from torch_sparse import SparseTensor


class GraphModification:
    """
    Base class, augmenting each graph with some features
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> Optional[Data]:
        return None


class GraphAddSkipConnection(GraphModification):
    """
    only for tree dataset
    """
    def __init__(self, depth):
        super(GraphAddSkipConnection, self).__init__()
        self.depth = depth

    def __call__(self, graph: Data):
        edge_index = torch.hstack((
            graph.edge_index,
            torch.vstack(
                [torch.arange(2 ** self.depth, 2 ** (self.depth + 1) - 1),
                 torch.zeros(2 ** self.depth - 1, dtype=torch.long)])
        ))
        graph.edge_index = edge_index
        return graph


class GraphRedirect(GraphModification):
    """
    only for tree dataset
    """
    def __init__(self, depth):
        super(GraphRedirect, self).__init__()
        self.depth = depth

    def __call__(self, graph: Data):
        edge_index = torch.vstack(
            [torch.arange(2 ** self.depth, 2 ** (self.depth + 1) - 1),
             torch.zeros(2 ** self.depth - 1, dtype=torch.long)]
        )
        graph.edge_index = edge_index
        return graph


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


class GraphCanonicalYClass(GraphModification):
    def __call__(self, graph: Data):
        graph.y -= 1
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
                                           num_nodes=graph.num_nodes)
                edge_attr = None
        else:
            if graph.edge_attr is not None:
                edge_index, edge_attr = coalesce(graph.edge_index,
                                                 graph.edge_attr,
                                                 graph.num_nodes)
            else:
                edge_index = coalesce(graph.edge_index,
                                      num_nodes=graph.num_nodes)
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


class GraphCoalesce(GraphModification):
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if graph.edge_attr is not None:
            edge_index, edge_attr = coalesce(graph.edge_index,
                                             graph.edge_attr,
                                             graph.num_nodes)
        else:
            edge_index = coalesce(graph.edge_index,
                                  num_nodes=graph.num_nodes)
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


class AugmentWithLongestPathEdgeCandidate(GraphModification):
    def __init__(self, num_candidate):
        super(AugmentWithLongestPathEdgeCandidate, self).__init__()
        self.num_candidate = num_candidate

    def __call__(self, graph: Data):
        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        edge_index = graph.edge_index.numpy()
        mat = csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                         shape=(graph.num_nodes, graph.num_nodes))

        g_dist_mat = shortest_path(mat, directed=False, return_predecessors=False)
        g_dist_mat[np.isinf(g_dist_mat)] = -1.
        g_dist_mat[g_dist_mat == -1] = g_dist_mat.max() + 1

        triu_idx = np.vstack(np.triu_indices(graph.num_nodes, k=1))
        distances = g_dist_mat[triu_idx[0], triu_idx[1]]
        edge_candidate = triu_idx[:, np.argsort(distances)[-self.num_candidate:]]

        graph.edge_candidate = torch.from_numpy(edge_candidate).T
        graph.num_edge_candidate = edge_candidate.shape[1]
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


class AugmentWithPerNodeRewiredGraphs(GraphModification):
    def __init__(self, sample_k, include_original_graph, ensemble):
        super(AugmentWithPerNodeRewiredGraphs, self).__init__()
        raise DeprecationWarning

    def __call__(self, graph: Data):
        pass


class AugmentWithDirectedGlobalRewiredGraphs(GraphModification):
    def __init__(self, sample_k, include_original_graph, ensemble):
        super(AugmentWithDirectedGlobalRewiredGraphs, self).__init__()
        self.sample_k = sample_k
        self.include_original_graph = include_original_graph
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        if self.sample_k >= graph.num_nodes ** 2:
            new_graph = deepcopy(graph)
            full_edge_index = np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes,))
            new_graph.edge_index = torch.from_numpy(full_edge_index)
            graphs = [new_graph] * self.ensemble
            if self.include_original_graph:
                graphs.append(graph)
        else:
            full_edge_index = torch.from_numpy(np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes, )))
            graphs = []
            for i in range(self.ensemble):
                idx = np.sort(np.random.choice(graph.num_nodes ** 2, size=self.sample_k, replace=False))
                g = graph.clone()
                g.edge_index = full_edge_index[:, idx]
                graphs.append(g)
            if self.include_original_graph:
                graphs.append(graph)
        # graphs = collate(graph.__class__, graphs, increment=True, add_batch=False)[0]
        # graphs.y = graph.y
        return graphs


class AugmentWithUndirectedGlobalRewiredGraphs(GraphModification):
    def __init__(self, sample_k, include_original_graph, ensemble):
        super(AugmentWithUndirectedGlobalRewiredGraphs, self).__init__()
        self.sample_k = sample_k
        self.include_original_graph = include_original_graph
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        if self.sample_k >= (graph.num_nodes * (graph.num_nodes - 1)) // 2:
            new_graph = deepcopy(graph)
            full_edge_index = np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes,))
            self_loop_idx = full_edge_index[0] == full_edge_index[1]
            full_edge_index = full_edge_index[:, ~self_loop_idx]
            new_graph.edge_index = torch.from_numpy(full_edge_index)
            graphs = [new_graph] * self.ensemble
            if self.include_original_graph:
                graphs.append(graph)
        else:
            full_edge_index = torch.from_numpy(np.vstack(np.triu_indices(graph.num_nodes, k=1)))
            graphs = []
            for i in range(self.ensemble):
                idx = np.sort(np.random.choice(full_edge_index.shape[1], size=self.sample_k, replace=False))
                directed_edge_index = full_edge_index[:, idx]
                undirec_edge_index = to_undirected(directed_edge_index, num_nodes=graph.num_nodes)
                g = graph.clone()
                g.edge_index = undirec_edge_index
                graphs.append(g)
            if self.include_original_graph:
                graphs.append(graph)
        return graphs


class AugmentWithExtraUndirectedGlobalRewiredGraphs(GraphModification):
    def __init__(self, sample_k, include_original_graph, ensemble):
        super(AugmentWithExtraUndirectedGlobalRewiredGraphs, self).__init__()
        self.sample_k = sample_k
        self.include_original_graph = include_original_graph
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        original_edge_index = graph.edge_index
        original_unique_edges_idx = original_edge_index[0] < original_edge_index[1]
        if self.sample_k >= (graph.num_nodes * (graph.num_nodes - 1)) // 2 - original_unique_edges_idx.sum():
            new_graph = deepcopy(graph)
            full_edge_index = np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes,))
            if not contains_self_loops(original_edge_index):
                self_loop_idx = full_edge_index[0] == full_edge_index[1]
                full_edge_index = full_edge_index[:, ~self_loop_idx]
            new_graph.edge_index = torch.from_numpy(full_edge_index)
            graphs = [new_graph] * self.ensemble
            if self.include_original_graph:
                graphs.append(graph)
        else:
            original_unique_edges = original_edge_index[:, original_unique_edges_idx]
            original_id = original_unique_edges[0] * graph.num_nodes + original_unique_edges[1]
            full_edge_index = torch.from_numpy(np.vstack(np.triu_indices(graph.num_nodes, k=1)))
            full_id = full_edge_index[0] * graph.num_nodes + full_edge_index[1]
            sample_pool_idx = torch.logical_not(torch.isin(full_id, original_id))
            sample_pool = full_edge_index[:, sample_pool_idx]
            graphs = []
            for i in range(self.ensemble):
                idx = np.sort(np.random.choice(sample_pool.shape[1], size=self.sample_k, replace=False))
                directed_edge_index = torch.hstack([sample_pool[:, idx], original_edge_index])
                undirec_edge_index = to_undirected(directed_edge_index, num_nodes=graph.num_nodes)
                g = graph.clone()
                g.edge_index = undirec_edge_index
                graphs.append(g)
            if self.include_original_graph:
                graphs.append(graph)
        return graphs


class AugmentWithHybridRewiredGraphs(GraphModification):
    """
    remove edges from existing ones, add new non-existing edges
    """
    def __init__(self, sample_k, include_original_graph, ensemble):
        super(AugmentWithHybridRewiredGraphs, self).__init__()
        self.sample_k = sample_k
        self.include_original_graph = include_original_graph
        self.ensemble = ensemble

    def __call__(self, graph: Data):
        original_edge_index = graph.edge_index
        original_unique_edges_idx = original_edge_index[0] <= original_edge_index[1]
        original_directed_edges = original_edge_index[:, original_unique_edges_idx]
        original_directed_id = original_directed_edges[0] * graph.num_nodes + original_directed_edges[1]

        # remove edges
        if self.sample_k >= original_directed_edges.shape[1]:
            new_graph = deepcopy(graph)
            new_graph.edge_index = original_edge_index[:, :0]
            graphs = [new_graph] * self.ensemble
        else:
            graphs = []
            for k in range(self.ensemble):
                remove_idx = np.sort(np.random.choice(original_directed_edges.shape[1], size=self.sample_k, replace=False))
                removed_edges = original_directed_edges[:, remove_idx]
                remove_id = removed_edges[0] * graph.num_nodes + removed_edges[1]
                remaining_edges = original_directed_edges[:, torch.logical_not(torch.isin(original_directed_id, remove_id))]
                g = graph.clone()
                g.edge_index = remaining_edges  # directed!
                graphs.append(g)

        # add new edges
        full_edge_index = torch.from_numpy(np.vstack(np.triu_indices(graph.num_nodes, k=1)))
        full_edge_id = full_edge_index[0] * graph.num_nodes + full_edge_index[1]
        new_edges_id = torch.logical_not(torch.isin(full_edge_id, original_directed_id))
        new_edges_pool = full_edge_index[:, new_edges_id]
        if self.sample_k >= (graph.num_nodes * (graph.num_nodes - 1)) // 2 - (original_edge_index[0] < original_edge_index[1]).sum():
            for g in graphs:
                g.edge_index = torch.hstack([g.edge_index, new_edges_pool])
                g.edge_index = to_undirected(g.edge_index)
        else:
            for g in graphs:
                idx = np.sort(np.random.choice(new_edges_pool.shape[1], size=self.sample_k, replace=False))
                directed_edge_index = torch.hstack([new_edges_pool[:, idx], g.edge_index])
                g.edge_index = to_undirected(directed_edge_index, num_nodes=graph.num_nodes)

        if self.include_original_graph:
            graphs.append(graph)
        return graphs


class AugmentWithSpatialInfo(GraphModification):
    """
    adapted from https://github.com/rampasek/GraphGPS/blob/95a17d57767b34387907f42a43f91a0354feac05/graphgps/encoder/graphormer_encoder.py#L15
    """
    def __init__(self, distance: int, num_in_degrees: int, num_out_degrees: int):
        super(AugmentWithSpatialInfo, self).__init__()
        self.distance = distance
        self.num_in_degrees = num_in_degrees
        self.num_out_degrees = num_out_degrees

    def __call__(self, data: Data):
        graph: nx.DiGraph = to_networkx(data)

        data.in_degrees = torch.tensor([d for _, d in graph.in_degree()])
        data.out_degrees = torch.tensor([d for _, d in graph.out_degree()])

        max_in_degree = torch.max(data.in_degrees)
        max_out_degree = torch.max(data.out_degrees)
        if max_in_degree >= self.num_in_degrees:
            raise ValueError(
                f"Encountered in_degree: {max_in_degree}, set posenc_"
                f"GraphormerBias.num_in_degrees to at least {max_in_degree + 1}"
            )
        if max_out_degree >= self.num_out_degrees:
            raise ValueError(
                f"Encountered out_degree: {max_out_degree}, set posenc_"
                f"GraphormerBias.num_out_degrees to at least {max_out_degree + 1}"
            )

        N = len(graph.nodes)
        shortest_paths = nx.shortest_path(graph)

        spatial_types = torch.empty(N ** 2, dtype=torch.long).fill_(self.distance)
        graph_index = torch.empty(2, N ** 2, dtype=torch.long)

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if data.edge_attr.dim() > 1 and data.edge_attr.shape[1] > 1 and set(data.edge_attr.unique().tolist()) == {0, 1}:
                data_edge_attr = torch.where(data.edge_attr)[1]
            else:
                data_edge_attr = data.edge_attr
        else:
            data_edge_attr = None

        if data_edge_attr is not None:
            shortest_path_types = torch.zeros(N ** 2, self.distance, dtype=torch.long)
            edge_attr = torch.zeros(N, N, dtype=torch.long)
            edge_attr[data.edge_index[0], data.edge_index[1]] = data_edge_attr

        for i in range(N):
            for j in range(N):
                graph_index[0, i * N + j] = i
                graph_index[1, i * N + j] = j

        for i, paths in shortest_paths.items():
            for j, path in paths.items():
                if len(path) > self.distance:
                    path = path[:self.distance]

                assert len(path) >= 1
                spatial_types[i * N + j] = len(path) - 1

                if len(path) > 1 and data_edge_attr is not None:
                    path_attr = [
                        edge_attr[path[k], path[k + 1]] for k in
                        range(len(path) - 1)  # len(path) * (num_edge_types)
                    ]

                    # We map each edge-encoding-distance pair to a distinct value
                    # and so obtain dist * num_edge_features many encodings
                    shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(path_attr, dtype=torch.long)

        data.spatial_types = spatial_types
        data.graph_index = graph_index

        if data_edge_attr is not None:
            data.shortest_path_types = shortest_path_types
        return data


class AugmentWithPlotCoordinates(GraphModification):
    """
    for networkx plots, save the positions of the original graphs
    """
    def __init__(self, layout = nx.kamada_kawai_layout):
        super(AugmentWithPlotCoordinates, self).__init__()
        self.layout = layout

    def __call__(self, data: Data):
        nx_graph = to_networkx(data)
        pos = self.layout(nx_graph)  # return a dict
        pos = np.vstack([pos[n] for n in range(data.num_nodes)])
        data.nx_layout = torch.from_numpy(pos)
        return data


def my_collate_fn(graphs: List[List]):
    """

    Args:
        graphs: graphs are like [[g1_ensemble1, g1_ensemble2, ..., original g1], [], [], ...]

    Returns:

    """
    original_graphs = [l[-1] for l in graphs]
    ordered_graphs = reduce(lambda a, b: a+b, [g for g in zip(*graphs)])
    new_batch = Batch.from_data_list(ordered_graphs)
    repeats = len(graphs[0])
    batchsize = len(graphs)
    new_batch.inter_graph_idx = torch.arange(batchsize).repeat(repeats)
    new_batch.y = torch.cat([g[0].y for g in graphs], dim=0)
    return new_batch, original_graphs


def collate_fn_with_origin_list(graphs: List[Data]):
    return Batch.from_data_list(graphs), graphs
