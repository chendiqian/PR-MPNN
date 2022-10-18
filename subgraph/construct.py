from typing import List

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from subgraph.grad_utils import Nodemask2Edgemask, CenterNodeIdentityMapping, nodemask2edgemask, centralize


def construct_random_local_structure_subgraphs(graphs: List[Data],
                                               node_mask: torch.FloatTensor,
                                               nnodes_wo_duplicate: torch.LongTensor,
                                               batch_wo_duplicate: torch.LongTensor,
                                               y_wo_duplicate: torch.Tensor,
                                               subgraph2node_aggr: str, ):
    """
    NGNN also delete nodes for their subgraphs
    see https://github.com/muhanzhang/NestedGNN/blob/a5adccf62d397ad7f83bc73be34eba3765df73fa/utils.py#L34

    :param graphs:
    :param node_mask:
    :param nnodes_wo_duplicate:
    :param batch_wo_duplicate:
    :param y_wo_duplicate:
    :param subgraph2node_aggr:
    :return:
    """
    subgraphs = []
    for i, g in enumerate(graphs):
        subgraphs += [g] * g.nnodes

    new_batch = Batch.from_data_list(subgraphs)
    new_edge_index, new_edge_attr = subgraph(node_mask,
                                             new_batch.edge_index,
                                             new_batch.edge_attr,
                                             relabel_nodes=True,
                                             num_nodes=new_batch.num_nodes)

    new_data = Data(x=new_batch.x[node_mask],
                    edge_index=new_edge_index,
                    edge_attr=new_edge_attr,
                    edge_weights=None,
                    y=y_wo_duplicate,)
    new_data.subgraphs2nodes = new_batch.batch[node_mask]
    new_data.batch = batch_wo_duplicate
    new_data.num_graphs = len(subgraphs)

    assert subgraph2node_aggr in ['add', 'center']
    if subgraph2node_aggr == 'add':
        new_data.node_mask = torch.ones(node_mask.sum(), dtype=torch.bool, device=node_mask.device)
    elif subgraph2node_aggr == 'center':
        nnodes = torch.repeat_interleave(nnodes_wo_duplicate, nnodes_wo_duplicate, dim=0)
        new_data.node_mask = centralize(node_mask, nnodes_wo_duplicate, nnodes)
        new_data.node_mask = new_data.node_mask[node_mask]
        new_data.subgraphs2nodes = new_data.subgraphs2nodes[new_data.node_mask]

    return new_data


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
