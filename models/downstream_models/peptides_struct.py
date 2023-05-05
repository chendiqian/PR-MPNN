import torch
from torch_geometric.nn import Set2Set

from models.my_convs import BaseGIN
from models.nn_modules import MLP


class PepStruct_GIN_Duo(torch.nn.Module):
    def __init__(self,
                 encoder,
                 in_features,
                 num_layers,
                 hidden,
                 num_classes,
                 mlp_layers_intragraph,
                 mlp_layers_intergraph,
                 inter_graph_pooling=None):
        super(PepStruct_GIN_Duo, self).__init__()

        self.encoder = encoder

        self.gnn1 = BaseGIN(in_features, num_layers, hidden)
        self.gnn2 = BaseGIN(in_features, num_layers, hidden)

        # intra-graph pooling
        self.pool1 = Set2Set(hidden, processing_steps=6)
        self.pool2 = Set2Set(hidden, processing_steps=6)

        self.inter_graph_pooling = inter_graph_pooling
        # inter-graph pooling
        if inter_graph_pooling == 'mean':
            assert mlp_layers_intergraph > 0
            self.mlp1_1 = MLP([2 * hidden] + [hidden] * mlp_layers_intragraph, dropout=0.)
            self.mlp1_2 = MLP([2 * hidden] + [hidden] * mlp_layers_intragraph, dropout=0.)
            self.mlp2 = MLP([hidden] * mlp_layers_intergraph + [num_classes], dropout=0.)
        elif inter_graph_pooling == 'cat':
            assert mlp_layers_intergraph > 0
            self.mlp1_1 = MLP([2 * hidden] + [hidden] * mlp_layers_intragraph, dropout=0.)
            self.mlp1_2 = MLP([2 * hidden] + [hidden] * mlp_layers_intragraph, dropout=0.)
            self.mlp2 = MLP([hidden * 2] + [hidden] * (mlp_layers_intergraph - 1) + [num_classes], dropout=0.)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.mlp1_1.reset_parameters()
        self.mlp1_2.reset_parameters()
        self.mlp2.reset_parameters()

    def forward(self, data):
        data1, data2 = data.data1, data.data2
        if self.encoder is not None:
            data1.x = self.encoder(data1)
            data2.x = self.encoder(data2)

        h_node1 = self.gnn1(data1)
        h_node2 = self.gnn2(data2)

        h_graph1 = self.pool1(h_node1, data1.batch)
        h_graph2 = self.pool2(h_node2, data2.batch)
        h_graph1 = self.mlp1_1(h_graph1)
        h_graph2 = self.mlp1_2(h_graph2)

        # inter graph pooling
        if self.inter_graph_pooling == 'mean':
            h_graph = (h_graph1 + h_graph2) / 2.
        elif self.inter_graph_pooling == 'cat':
            h_graph = torch.cat([h_graph1, h_graph2], dim=1)
        else:
            raise ValueError

        h_graph = self.mlp2(h_graph)
        return h_graph
