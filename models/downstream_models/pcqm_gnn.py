import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from models.nn_modules import MLP

from models.my_convs import GINConv, GINEConv

import numpy as np


class PCQM_Net(torch.nn.Module):
    def __init__(
        self,
        encoder,
        gnn_type,
        edge_encoder,
        num_classes,
        emb_sizes,
        num_layers,
        drpt_prob=0.5,
        graph_pooling="max",
    ):
        super(PCQM_Net, self).__init__()
        self.drpt_prob = drpt_prob
        self.graph_pooling = graph_pooling

        self.initial_mlp = encoder
        self.initial_linear = Linear(emb_sizes, num_classes)

        gnn_layers = []
        linears = []
        # mlps = []
        for i in range(num_layers):
            if gnn_type in ['gin', 'gine']:
                mlp = Sequential(
                        Linear(emb_sizes, emb_sizes),
                        BatchNorm1d(emb_sizes),
                        ReLU(),
                        Linear(emb_sizes, emb_sizes),
                        BatchNorm1d(emb_sizes),
                        ReLU(),
                )
            # mlps.append(mlp)
                if gnn_type == 'gin':
                    gnn_layer = GINConv(emb_sizes, mlp)
                else:
                    gnn_layer = GINEConv(emb_sizes, mlp, edge_encoder)
            else:
                raise NotImplementedError
            gnn_layers.append(gnn_layer)

        self.gnn_modules = ModuleList(gnn_layers)
        self.linear_modules = ModuleList(linears)
        # self.mlp_moduls = ModuleList(mlps)   # unclear why they need this https://github.com/radoslav11/SP-MPNN/blob/main/src/models/gin.py

        self.layer_post_mp = Sequential(
            ReLU(),
            Linear(emb_sizes, emb_sizes),
            ReLU(),
            Linear(emb_sizes, 1),
        )
        self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)


        if graph_pooling == "sum":
            self.pooling = global_add_pool
        elif graph_pooling == 'max':
            self.pooling = global_max_pool
        elif graph_pooling == "mean":
            self.pooling = global_mean_pool


    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).

        Implementation from OGB:
        https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

        Args:
            y_pred_neg: array with shape (batch size, num_entities_neg).
            y_pred_pos: array with shape (batch size, )
        """

        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1. / ranking_list.to(torch.float)

            return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}

        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                    axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1. / ranking_list.astype(np.float32)

            return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}


    def _apply_index(self, batch):
        return batch.x[batch.edge_index_labeled], batch.edge_label

    def forward(self, data):
        x_feat = self.initial_mlp(data)  # Otherwise by an MLP
        out = F.dropout(
            self.pooling(self.initial_linear(x_feat), data.batch), p=self.drpt_prob
        )

        for gin_layer, linear_layer in zip(self.gnn_modules, self.linear_modules):
            x_feat = gin_layer(x_feat, data.edge_index, data.edge_attr, data.edge_weight)

            out += F.dropout(
                linear_layer(self.pooling(x_feat, data.batch)),
                p=self.drpt_prob,
                training=self.training,
            )

        out = self.layer_post_mp(out)
        pred, label = self._apply_index(out)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)

        return pred, label
