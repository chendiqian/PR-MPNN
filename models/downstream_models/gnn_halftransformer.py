import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set

from models.nn_modules import MLP, TransformerLayer
from models.my_convs import BaseGIN, BaseGINE


class GNN_HalfTransformer(torch.nn.Module):
    def __init__(self,
                 encoder,
                 edge_encoder,
                 base_gnn,
                 head,
                 gnn_in_features,
                 num_layers,
                 tf_layers,
                 hidden,
                 tf_hidden,
                 tf_dropout,
                 attn_dropout,
                 num_classes,
                 mlp_layers_intragraph,
                 layer_norm,
                 batch_norm,
                 use_spectral_norm,
                 graph_pooling='mean'):
        super(GNN_HalfTransformer, self).__init__()

        self.encoder = encoder

        self.tf_layers = torch.nn.ModuleList([])
        for l in range(tf_layers):
            self.tf_layers.append(TransformerLayer(tf_hidden,
                                                   head,
                                                   'relu',
                                                   tf_dropout,
                                                   attn_dropout,
                                                   layer_norm,
                                                   batch_norm,
                                                   use_spectral_norm))

        if base_gnn == 'gin':
            model_class = BaseGIN
        elif base_gnn == 'gine':
            model_class = BaseGINE
        else:
            raise NotImplementedError

        self.lin = torch.nn.Linear(gnn_in_features, hidden)
        self.gnn = model_class(hidden, num_layers, hidden, hidden, True, 0., True, edge_encoder)

        # intra-graph pooling
        self.graph_pool_idx = 'batch'
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=6)
        elif graph_pooling == 'transductive':
            self.pool = lambda x, transductive_mask: x[transductive_mask]
            self.graph_pool_idx = 'transductive_mask'
        elif graph_pooling is None:  # node pred
            self.pool = lambda x, *args: x
        elif graph_pooling == 'root':
            self.pool = lambda x, root_mask: x[root_mask]
            self.graph_pool_idx = 'root_mask'
        else:
            raise NotImplementedError

        self.mlp = MLP([hidden if graph_pooling != 'set2set' else hidden * 2] * mlp_layers_intragraph + [num_classes], dropout=0.)

    def forward(self, data):
        h_node = self.encoder(data)

        for l in self.tf_layers:
            h_node = l(h_node, data)

        data.x = torch.relu(self.lin(h_node))
        h_node = self.gnn(data.x, data.edge_index, data.edge_attr, data.edge_weight)

        h_graph = self.pool(h_node, getattr(data, self.graph_pool_idx))
        h_graph = self.mlp(h_graph)

        return h_graph

    def reset_parameters(self):
        raise NotImplementedError
