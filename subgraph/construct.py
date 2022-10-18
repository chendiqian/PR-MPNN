from typing import List

import torch
from torch_geometric.data import Data, Batch
from subgraph.grad_utils import Nodemask2Edgemask, CenterNodeIdentityMapping, nodemask2edgemask, centralize


def construct_imle_local_structure_subgraphs(graphs: List[Data],
                                             node_mask: torch.FloatTensor,
                                             nnodes_wo_duplicate: torch.LongTensor,
                                             batch_wo_duplicate: torch.LongTensor,
                                             y_wo_duplicate: torch.Tensor,
                                             subgraph2node_aggr: str,
                                             grad: bool):
    """

    :param graphs:
    :param node_mask:
    :param nnodes_wo_duplicate:
    :param subgraph2node_aggr:
    :param batch_wo_duplicate:
    :param y_wo_duplicate:
    :param grad:
    :return:
    """
    subgraphs = []
    for i, g in enumerate(graphs):
        subgraphs += [g] * g.nnodes

    new_batch = Batch.from_data_list(subgraphs)
    assert subgraph2node_aggr in ['add', 'center']

    n2e_func = Nodemask2Edgemask.apply if grad else nodemask2edgemask
    new_batch.edge_weights = n2e_func(node_mask,
                                      new_batch.edge_index,
                                      torch.tensor(new_batch.num_nodes,
                                                   device=node_mask.device))

    if subgraph2node_aggr == 'add':
        new_batch.node_mask = node_mask
    elif subgraph2node_aggr == 'center':
        center_func = CenterNodeIdentityMapping.apply if grad else centralize
        new_batch.node_mask = center_func(node_mask, nnodes_wo_duplicate, new_batch.nnodes)

    new_batch.subgraphs2nodes = new_batch.batch
    del new_batch.batch
    new_batch.batch = batch_wo_duplicate
    new_batch.y = y_wo_duplicate

    return new_batch
