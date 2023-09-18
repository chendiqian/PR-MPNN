# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
import torch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
import numpy as np


def _eval_mrr(y_pred_pos, y_pred_neg):
    y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    # hits1_list = (ranking_list <= 1).to(torch.float)
    # hits3_list = (ranking_list <= 3).to(torch.float)
    # hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)

    # return {'hits@1_list': hits1_list,
    #         'hits@3_list': hits3_list,
    #         'hits@10_list': hits10_list,
    #         'mrr_list': mrr_list}
    return mrr_list.mean().item()


def eval_mrr(y_true: torch.Tensor,
             y_pred: torch.Tensor,
             npreds: torch.Tensor,
             nnodes: torch.Tensor,
             edge_label_idx: torch.Tensor):
    device = y_true.device
    y_true = torch.split(y_true, npreds.cpu().tolist(), dim=0)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = edge_label_idx - offset[None]
    split_edge_label_idx_list = torch.split(edge_label_idx, npreds.cpu().tolist(), dim=1)

    mrr_list = []
    for pred, truth, edge_label_idx, nnode in zip(y_pred, y_true, split_edge_label_idx_list, nnodes):
        pred = pred[:nnode, :nnode]

        pos_edge_index = edge_label_idx[:, truth.squeeze() == 1]
        num_pos_edges = pos_edge_index.shape[1]

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]

        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, nnode], dtype=torch.bool, device=device)
            neg_mask[torch.arange(num_pos_edges, device=device), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            mrr = _eval_mrr(pred_pos, pred_neg)
        else:
            # Return empty stats.
            mrr = _eval_mrr(pred_pos, pred_pos)

        mrr_list.append(mrr)

    return np.mean(mrr_list)


def eval_mrr_batch(y_true: torch.Tensor,
                   y_pred: torch.Tensor,
                   npreds: torch.Tensor,
                   nnodes: torch.Tensor,
                   edge_label_idx: torch.Tensor):
    device = y_true.device
    num_graphs = len(nnodes)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = edge_label_idx - offset[None]

    # common tensors
    arange_num_graphs = torch.arange(num_graphs, device=device)

    edge_batch_index = torch.repeat_interleave(arange_num_graphs, npreds)

    # get positive edges
    pos_edge_index = edge_label_idx[:, y_true.squeeze() == 1]
    num_pos_edges = scatter(y_true.long().squeeze(), edge_batch_index, dim=0, reduce='sum')
    pos_edge_batch_index = edge_batch_index[y_true.squeeze() == 1]
    assert num_pos_edges.min() > 0
    pred_pos = y_pred[pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]]

    # get negative edges: npreds * (nnodes - 1)
    neg_mask = torch.ones(num_graphs, num_pos_edges.max(), nnodes.max(), dtype=torch.bool, device=device)
    neg_mask[arange_num_graphs.repeat_interleave(nnodes.max() - nnodes), :,
             torch.cat([torch.arange(n, nnodes.max(), device=device) for n in nnodes])] = False

    _, real_edge_mask = to_dense_batch(pos_edge_index[0],
                                       torch.repeat_interleave(arange_num_graphs, num_pos_edges))
    fake_edge_idx = (~real_edge_mask).nonzero().t()
    neg_mask[fake_edge_idx[0], fake_edge_idx[1], :] = False
    cat_arange_pos_edges = torch.cat([torch.arange(n, device=device) for n in num_pos_edges], dim=0)
    neg_mask[pos_edge_batch_index, cat_arange_pos_edges, pos_edge_index[1]] = False
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0]][neg_mask[pos_edge_batch_index, cat_arange_pos_edges]]

    # commen tensors
    list_arange_posedge = [torch.arange(num_pos_edges[i], device=device) for i in range(num_graphs)]

    # construct a concatenated matrix
    concat_edges = torch.ones(num_graphs, num_pos_edges.max(), nnodes.max(), device=device) * -1.e10
    # fill in negative edges
    idx0 = torch.repeat_interleave(arange_num_graphs, num_pos_edges * (nnodes - 1))
    idx1 = torch.hstack([list_arange_posedge[i].repeat_interleave(nnodes[i] - 1) for i in range(num_graphs)])
    idx2 = torch.hstack([torch.arange(1, nnodes[i], device=device).repeat(num_pos_edges[i]) for i in range(num_graphs)])
    concat_edges[idx0, idx1, idx2] = pred_neg
    # fill in positive edges
    idx0 = torch.repeat_interleave(arange_num_graphs, num_pos_edges)
    idx1 = torch.cat(list_arange_posedge)
    concat_edges[idx0, idx1, 0] = pred_pos

    # predict
    argsort = torch.argsort(concat_edges, dim=-1, descending=True)
    ranking_list = argsort == 0.
    batch_mmr = scatter(1 / (ranking_list[real_edge_mask].nonzero()[:, -1] + 1), idx0, reduce='mean', dim=0)

    return batch_mmr.mean()



def get_eval(task_type: str,
             y_true: torch.Tensor,
             y_pred: torch.Tensor,
             npreds: torch.Tensor,
             nnodes: torch.Tensor,
             edge_label_idx: torch.Tensor):

    if task_type == 'mrr':
        metric = eval_mrr_batch(y_true,
                                y_pred,
                                npreds=npreds,
                                nnodes=nnodes,
                                edge_label_idx=edge_label_idx)
    else:
        raise NotImplementedError

    return metric
