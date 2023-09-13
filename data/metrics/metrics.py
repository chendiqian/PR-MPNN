# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
import torch


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

    return sum(mrr_list) / len(mrr_list)


def get_eval(task_type: str,
             y_true: torch.Tensor,
             y_pred: torch.Tensor,
             npreds: torch.Tensor,
             nnodes: torch.Tensor,
             edge_label_idx: torch.Tensor):

    if task_type == 'mrr':
        metric = eval_mrr(y_true,
                          y_pred,
                          npreds=npreds,
                          nnodes=nnodes,
                          edge_label_idx=edge_label_idx)
    else:
        raise NotImplementedError

    return metric
