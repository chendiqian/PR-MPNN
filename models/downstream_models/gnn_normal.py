import torch

from models.my_convs import BaseGIN, BaseGINE, BasePNA, BaseGCN
from models.nn_modules import MLP


class GNN_Normal(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
                 deg_hist,
                 in_features,
                 num_layers,
                 hidden,
                 num_classes,
                 use_bn,
                 dropout,
                 residual,
                 mlp_layers_intragraph):
        super(GNN_Normal, self).__init__()

        self.encoder = encoder

        if base_gnn == 'gin':
            self.gnn = BaseGIN(in_features, num_layers, hidden, hidden, use_bn,
                                   dropout, residual, edge_encoder)
        elif base_gnn == 'gine':
            self.gnn = BaseGINE(in_features, num_layers, hidden, hidden, use_bn,
                               dropout, residual, edge_encoder)
        elif base_gnn == 'pna':
            assert deg_hist is not None
            self.gnn = BasePNA(in_features, num_layers, hidden, hidden, use_bn,
                                dropout, residual, deg_hist, edge_encoder)
        elif base_gnn == 'gcn':
            self.gnn = BaseGCN(in_features, num_layers, hidden, hidden, use_bn,
                                dropout, residual, edge_encoder)
        else:
            raise NotImplementedError

        if mlp_layers_intragraph > 0:
            self.mlp = MLP([hidden] * mlp_layers_intragraph + [num_classes], dropout=0.)
        else:
            self.mlp = lambda x: x

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        if self.encoder is not None:
            data.x = self.encoder(data)

        h_node = self.gnn(data.x, data.edge_index, data.edge_attr, data.edge_weight)
        h_node = self.mlp(h_node)

        return h_node
