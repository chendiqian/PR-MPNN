from typing import Optional, List

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (is_undirected,
                                   to_undirected,
                                   add_remaining_self_loops,
                                   coalesce,
                                   to_networkx,
                                   subgraph)
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_sparse import SparseTensor
from data.utils.datatype_utils import BatchOriginalDataStructure
from data.utils.neighbor_utils import batch_get_subgraphs_nodeidx


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


class AugmentwithNumbers(GraphModification):

    def __call__(self, graph: Data):
        graph.nnodes = graph.num_nodes
        graph.nedges = graph.num_edges
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


class AugmentWithEdgeCandidate(GraphModification):
    def __init__(self, heuristic, num_candidate, directed):
        super(AugmentWithEdgeCandidate, self).__init__()
        self.heu = heuristic
        self.num_candidate = num_candidate
        self.directed = directed

    def __call__(self, graph: Data):
        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        edge_index = graph.edge_index.numpy()

        if self.heu == 'longest_path':
            mat = csr_matrix((np.ones(edge_index.shape[1]),
                              (edge_index[0], edge_index[1])),
                             shape=(graph.num_nodes, graph.num_nodes))

            mat = shortest_path(mat, directed=False, return_predecessors=False)
            mat[np.isinf(mat)] = -1.
            mat[mat == -1] = mat.max() + 1
        elif self.heu == 'node_similarity':
            x = graph.x
            if x.dtype == torch.long:
                x = torch.nn.functional.one_hot(x).reshape(x.shape[0], -1).float()
            x = torch.nn.functional.normalize(x, p=2.0, dim=1).numpy()
            mat = x @ x.T
            mat[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0.
        else:
            raise NotImplementedError

        if not self.directed:
            candidate_idx = np.vstack(np.triu_indices(graph.num_nodes, k=1))
        else:
            candidate_idx = np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes))
            candidate_idx = candidate_idx[:, candidate_idx[0] != candidate_idx[1]]  # no self loop

        # exclude original edges
        candidate_idx_id = candidate_idx[0] * graph.num_nodes + candidate_idx[1]
        org_edge_index_id = edge_index[0] * graph.num_nodes + edge_index[1]
        multi_hop_idx = np.logical_not(np.in1d(candidate_idx_id, org_edge_index_id))
        candidate_idx = candidate_idx[:, multi_hop_idx]

        distances = mat[candidate_idx[0], candidate_idx[1]]
        edge_candidate = candidate_idx[:, np.argsort(distances)[-self.num_candidate:]]

        graph.edge_candidate = torch.from_numpy(edge_candidate).T
        graph.num_edge_candidate = edge_candidate.shape[1]
        return graph


class AugmentWithSEALSubgraphs(GraphModification):
    def __init__(self, khop):
        super(AugmentWithSEALSubgraphs, self).__init__()
        self.khop = khop

    def __call__(self, graph: Data):
        assert hasattr(graph, 'edge_candidate') and hasattr(graph, 'num_edge_candidate')

        subgraph_indices = batch_get_subgraphs_nodeidx(graph.edge_candidate.numpy(),
                                                       graph.edge_index.numpy(),
                                                       graph.num_nodes,
                                                       self.khop)

        gs = []

        if hasattr(graph, 'nx_layout'):
            x_range = graph.nx_layout[:, 0].max() - graph.nx_layout[:, 0].min()
            y_range = graph.nx_layout[:, 1].max() - graph.nx_layout[:, 1].min()

        for idx in subgraph_indices:
            idx = sorted(list(idx))
            edge_index, edge_attr = subgraph(idx, graph.edge_index, graph.edge_attr,
                                             relabel_nodes=True,
                                             num_nodes=graph.num_nodes)
            g = Data(x=graph.x[idx],
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     y=graph.y,
                     num_nodes=len(idx))
            g = AugmentwithNumbers()(g)

            # use normalized layout positions
            if hasattr(graph, 'nx_layout'):
                nx_pos = graph.nx_layout[idx]
                max_pos = nx_pos.max(0, keepdims=True).values
                min_pos = nx_pos.min(0, keepdims=True).values
                nx_pos = (nx_pos - min_pos) / (max_pos - min_pos) - 0.5
                nx_pos[:, 0] *= x_range
                nx_pos[:, 1] *= y_range
                g.nx_layout = nx_pos
            # Todo: maybe recalculate, but time costly, ~3h on ZINC
            if hasattr(graph, 'pestat_RWSE'):
                g.pestat_RWSE = graph.pestat_RWSE[idx]
                # g = AddRandomWalkPE(graph.pestat_RWSE.shape[1], 'pestat_RWSE')(g)
            if hasattr(graph, 'EigVecs'):
                g.EigVecs = graph.EigVecs[idx]
                # g = AddLaplacianEigenvectorPE(max(1, min(graph.EigVecs.shape[1], len(idx) - 2)), 'EigVecs', is_undirected=True)(g)
                # if g.EigVecs.shape[1] < graph.EigVecs.shape[1]:
                #     # pad with 0s
                #     g.EigVecs = torch.nn.functional.pad(g.EigVecs, (0, graph.EigVecs.shape[1] - g.EigVecs.shape[1]), mode='constant', value=0)
            gs.append(g)

        return gs


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


class AugmentWithPlotCoordinates(GraphModification):
    """
    for networkx plots, save the positions of the original graphs
    """
    def __init__(self, layout):
        super(AugmentWithPlotCoordinates, self).__init__()
        self.layout = layout

    def __call__(self, data: Data):
        nx_graph = to_networkx(data)
        pos = self.layout(nx_graph)  # return a dict
        pos = np.vstack([pos[n] for n in range(data.num_nodes)])
        data.nx_layout = torch.from_numpy(pos)
        return data


def collate_fn_with_origin_list(graphs: List[Data]):
    batch = Batch.from_data_list(graphs)
    return BatchOriginalDataStructure(batch=batch,
                                      list=graphs,
                                      y=batch.y,
                                      num_graphs=len(graphs))
