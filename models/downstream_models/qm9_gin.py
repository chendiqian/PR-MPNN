import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool


class QM9_NetGIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes,
        num_layers,
        drpt_prob=0.5,
        graph_pooling="max",
    ):
        super(QM9_NetGIN, self).__init__()
        self.drpt_prob = drpt_prob
        self.graph_pooling = graph_pooling

        self.initial_mlp_modules = ModuleList(
            [
                Linear(num_features, emb_sizes),
                BatchNorm1d(emb_sizes),
                ReLU(),
                Linear(emb_sizes, emb_sizes),
                BatchNorm1d(emb_sizes),
                ReLU(),
            ]
        )
        self.initial_mlp = Sequential(*self.initial_mlp_modules)
        self.initial_linear = Linear(emb_sizes, num_classes)

        gin_layers = []
        linears = []
        mlps = []
        for i in range(num_layers):
            mlp = ModuleList(
                [
                    Linear(emb_sizes, emb_sizes),
                    BatchNorm1d(emb_sizes),
                    ReLU(),
                    Linear(emb_sizes, emb_sizes),
                    BatchNorm1d(emb_sizes),
                    ReLU(),
                ]
            )
            mlps.append(mlp)
            gin_layer = GINConv(Sequential(*mlp), eps=0., train_eps=True)
            gin_layers.append(gin_layer)
            linears.append(Linear(emb_sizes, num_classes))

        self.gin_modules = ModuleList(gin_layers)
        self.linear_modules = ModuleList(linears)
        self.mlp_moduls = ModuleList(mlps)

        if graph_pooling == "sum":
            self.pooling = global_add_pool
        elif graph_pooling == 'max':
            self.pooling = global_max_pool
        elif graph_pooling == "mean":
            self.pooling = global_mean_pool

    def forward(self, data):
        x_feat = data.x
        edge_index = data.edge_index

        x_feat = self.initial_mlp(x_feat)  # Otherwise by an MLP
        out = F.dropout(
            self.pooling(self.initial_linear(x_feat), data.batch), p=self.drpt_prob
        )

        for gin_layer, linear_layer in zip(self.gin_modules, self.linear_modules):
            x_feat = gin_layer(x_feat, edge_index)

            out += F.dropout(
                linear_layer(self.pooling(x_feat, data.batch)),
                p=self.drpt_prob,
                training=self.training,
            )

        return out
