import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv

from models.nn_modules import MLP


# https://github.com/chrsmrrs/sparsewl/blob/master/neural_higher_order/ZINC/local_2_10K.py#L105
class EdgeSelector2WL(torch.nn.Module):
    def __init__(self, in_features, dim, num_classes, conv_layer, mlp_layer, directed_sampling):
        super(EdgeSelector2WL, self).__init__()

        self.directed_sampling = directed_sampling
        self.conv_layer = conv_layer

        self.conv1s = torch.nn.ModuleList([])
        self.conv2s = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.inter_mlps = torch.nn.ModuleList([])

        for i in range(conv_layer):
            in_dim = in_features if i ==0 else dim
            self.conv1s.append(GINConv(
                Sequential(Linear(in_dim, dim),
                           ReLU(),
                           Linear(dim, dim)),
                train_eps=True))
            self.conv2s.append(GINConv(
                Sequential(Linear(in_dim, dim),
                           ReLU(),
                           Linear(dim, dim)),
                train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(dim))
            self.inter_mlps.append(MLP([2 * dim, dim, dim]))

        self.mlp1 = MLP([dim * 4] + [dim] * (mlp_layer - 1) + [num_classes], batch_norm=True, dropout=0.)
        self.mlp2 = MLP([dim * 4] + [dim] * (mlp_layer - 1) + [num_classes], batch_norm=True, dropout=0.)

    def forward(self, data):
        x = data.x_2wl

        nnodes_2wl = data.nnodes ** 2
        edge_index_1 = data.edge_index1
        edge_index_2 = data.edge_index2

        intermediate_x = []
        for i in range(self.conv_layer):
            x_1 = F.relu(self.conv1s[i](x, edge_index_1))
            x_2 = F.relu(self.conv2s[i](x, edge_index_2))
            x = self.inter_mlps[i](torch.cat([x_1, x_2], dim=-1))
            x = self.bns[i](x)
            intermediate_x.append(x)

        x = torch.cat(intermediate_x, dim=-1)
        selection_scores = self.mlp1(x)
        deletion_scores = self.mlp2(x)

        # prepare for indexing
        cumsum = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device), torch.cumsum(nnodes_2wl, dim=0)[:-1]])
        edge_index = data.edge_index
        edge_index = edge_index - data._inc_dict['edge_index'].to(x.device).repeat_interleave(data.nedges)[None]

        edge_index_rel_local = data.nnodes.repeat_interleave(data.nedges)
        edge_index_rel_global = cumsum.repeat_interleave(data.nedges)

        edge_candid_rel_local = data.nnodes.repeat_interleave(data.num_edge_candidate)
        edge_candid_rel_global = cumsum.repeat_interleave(data.num_edge_candidate)

        if self.directed_sampling:
            select_edge_candidates = selection_scores[data.edge_candidate[..., 0] * edge_candid_rel_local +
                                                      data.edge_candidate[..., 1] + edge_candid_rel_global]
            delete_edge_candidates = deletion_scores[edge_index[0] * edge_index_rel_local +
                                                     edge_index[1] + edge_index_rel_global]
        else:
            select_edge_candidates = selection_scores[data.edge_candidate[..., 0] * edge_candid_rel_local +
                                                      data.edge_candidate[..., 1] + edge_candid_rel_global] + \
                                     selection_scores[data.edge_candidate[..., 1] * edge_candid_rel_local +
                                                      data.edge_candidate[..., 0] + edge_candid_rel_global]

            edge_index_dir_mask = edge_index[0] <= edge_index[1]  # self loops included
            edge_index = edge_index[:, edge_index_dir_mask]
            edge_index_rel_local = edge_index_rel_local[edge_index_dir_mask]
            edge_index_rel_global = edge_index_rel_global[edge_index_dir_mask]

            delete_edge_candidates = deletion_scores[edge_index[0] * edge_index_rel_local +
                                                     edge_index[1] + edge_index_rel_global]

        edge_rel = torch.hstack([torch.zeros(1, dtype=torch.long, device=x.device),
                                 torch.cumsum(data.nnodes, dim=0)[:-1]])
        edge_candidate_idx = data.edge_candidate + edge_rel.repeat_interleave(data.num_edge_candidate)[:, None]

        return select_edge_candidates, delete_edge_candidates, edge_candidate_idx
