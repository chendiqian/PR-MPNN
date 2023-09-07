from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (to_undirected)

from data.utils.datatype_utils import DuoDataStructure
from data.utils.tensor_utils import non_merge_coalesce


def get_complement_edge_index(edge_index: torch.Tensor, num_nodes: int):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    possible_edges = np.vstack(np.triu_indices(num_nodes, -num_nodes))
    idx = possible_edges[0] * num_nodes + possible_edges[1]
    exist_idx = edge_index[0] * num_nodes + edge_index[1]
    exist_mask = np.in1d(idx, exist_idx)
    complement_mask = np.logical_not(exist_mask)
    complement_edge_index = possible_edges[:, complement_mask]
    complement_edge_index = complement_edge_index[:, complement_edge_index[0] != complement_edge_index[1]]  # remove self loop
    return torch.from_numpy(complement_edge_index).to(torch.long)


class AugmentWithRandomRewiredGraphs:
    """
    random baseline for adding and deleting on the same graph.
    supports in_place only for now
    """
    def __init__(self,
                 sample_k_add: int,
                 sample_k_del: int,
                 include_original_graph: bool,
                 in_place: bool,
                 ensemble: int,
                 separate: bool,
                 directed: bool):
        super(AugmentWithRandomRewiredGraphs, self).__init__()
        assert in_place
        self.sample_k_add = sample_k_add
        self.sample_k_del = sample_k_del
        self.include_original_graph = include_original_graph
        self.ensemble = ensemble
        self.separate = separate
        self.directed = directed

    def __call__(self, graph: Data):
        original_edge_index = graph.edge_index
        original_edge_attr = graph.edge_attr
        new_edges_pool = get_complement_edge_index(original_edge_index, graph.num_nodes)
        if not self.directed:
            original_direct_mask = original_edge_index[0] <= original_edge_index[1]  # self loops included
            original_edge_index = original_edge_index[:, original_direct_mask]
            if original_edge_attr is not None:
                original_edge_attr = original_edge_attr[original_direct_mask, :]
            new_edges_pool = new_edges_pool[:, new_edges_pool[0] < new_edges_pool[1]]

        new_graphs = {'add': [graph.clone() for _ in range(self.ensemble)],
                      'del': [graph.clone() for _ in range(self.ensemble)] if self.separate and self.sample_k_del > 0 else None}

        # remove edges
        if self.sample_k_del > 0:
            key = 'del' if self.separate else 'add'
            for k in range(self.ensemble):
                g = new_graphs[key][k]
                remain_idx = np.random.choice(original_edge_index.shape[1],
                                              size=max(0, original_edge_index.shape[1] - self.sample_k_del),
                                              replace=False)
                g.edge_index = original_edge_index[:, remain_idx]
                if original_edge_attr is not None:
                    g.edge_attr = original_edge_attr[remain_idx, :]
                if not self.directed:
                    edge_index, edge_attr = to_undirected(g.edge_index,
                                                          g.edge_attr,
                                                          g.num_nodes,
                                                          reduce='mean')
                    g.edge_index = edge_index
                    g.edge_attr = edge_attr
                # new_graphs[key][k] = g

        # add new edges
        for k in range(self.ensemble):
            g = new_graphs['add'][k]
            idx = np.random.choice(new_edges_pool.shape[1],
                                   size=min(new_edges_pool.shape[1], self.sample_k_add),
                                   replace=False)
            add_edge_index = new_edges_pool[:, idx]
            if not self.directed:
                add_edge_index = to_undirected(add_edge_index, num_nodes=graph.num_nodes)
            if original_edge_attr is not None:
                add_edge_attr = original_edge_attr.new_ones(add_edge_index.shape[1], original_edge_attr.shape[1])
            else:
                add_edge_attr = None

            edge_index = torch.hstack([g.edge_index, add_edge_index])
            if add_edge_attr is not None:
                edge_attr = torch.vstack([g.edge_attr, add_edge_attr])
            else:
                edge_attr = None

            merged_edge_index, merged_edge_attr, _ = non_merge_coalesce(
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_weight=None,
                num_nodes=graph.num_nodes)

            g.edge_index = merged_edge_index
            g.edge_attr = merged_edge_attr
            # new_graphs['add'][k] = g

        return_graphs = [new_graphs['add']]
        if new_graphs['del'] is not None:
            return_graphs.append(new_graphs['del'])
        if self.include_original_graph:
            return_graphs.append(graph)
        return return_graphs


def collate_random_rewired_batch(graphs: List[List], include_org: bool):
    if include_org:
        original_graphs = Batch.from_data_list([l.pop(-1) for l in graphs])
    else:
        original_graphs = None

    separate = len(graphs[0]) == 2   # deleted-edge graphs are separate from added-edge graphs
    add_graphs = []
    for g in graphs:
        batch = g.pop(0)
        add_graphs.extend(batch)
    if separate:
        del_graphs = []
        for g in graphs:
            batch = g.pop(-1)
            del_graphs.extend(batch)
    else:
        del_graphs = None

    assert not len(graphs[0])

    add_new_batch = Batch.from_data_list(add_graphs)
    candidates = [add_new_batch]
    if del_graphs is not None:
        candidates.append(Batch.from_data_list(del_graphs))
    return DuoDataStructure(org=original_graphs,
                            candidates=candidates,
                            y=add_new_batch.y,
                            num_graphs=add_new_batch.num_graphs,
                            num_unique_graphs=len(graphs))
