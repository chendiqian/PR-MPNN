import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from models.my_convs import GINConv, GINEConv


class QM9_Net(torch.nn.Module):
    def __init__(
        self,
        gnn_type,
        edge_encoder,
        num_features,
        num_classes,
        emb_sizes,
        num_layers,
        drpt_prob=0.5,
        graph_pooling="max",
    ):
        super(QM9_Net, self).__init__()
        self.drpt_prob = drpt_prob
        self.graph_pooling = graph_pooling

        self.initial_mlp = Sequential(Linear(num_features, emb_sizes),
                                      BatchNorm1d(emb_sizes),
                                      ReLU(),
                                      Linear(emb_sizes, emb_sizes),
                                      BatchNorm1d(emb_sizes),
                                      ReLU(), )
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
            linears.append(Linear(emb_sizes, num_classes))

        self.gnn_modules = ModuleList(gnn_layers)
        self.linear_modules = ModuleList(linears)
        # self.mlp_moduls = ModuleList(mlps)   # unclear why they need this https://github.com/radoslav11/SP-MPNN/blob/main/src/models/gin.py

        if graph_pooling == "sum":
            self.pooling = global_add_pool
        elif graph_pooling == 'max':
            self.pooling = global_max_pool
        elif graph_pooling == "mean":
            self.pooling = global_mean_pool

    def forward(self, data):
        x_feat = data.x

        x_feat = self.initial_mlp(x_feat)  # Otherwise by an MLP
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

        return out
