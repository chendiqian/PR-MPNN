import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj

from models.nn_utils import reset_sequential_parameters

# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)
# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)
LARGE_NUMBER = 1.e10


class BiasEncoder(torch.nn.Module):
    def __init__(self, num_heads: int, num_spatial_types: int,
                 num_edge_types: int, use_graph_token: bool = False):
        """Implementation of the bias encoder of Graphormer.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.
        Args:
            num_heads: The number of heads of the Graphormer model
            num_spatial_types: The total number of different spatial types
            num_edge_types: The total number of different edge types
            use_graph_token: If True, pads the attn_bias to account for the
            additional graph token that can be added by the ``NodeEncoder``.
        """
        super().__init__()
        self.num_heads = num_heads

        # Takes into account disconnected nodes
        self.spatial_encoder = torch.nn.Embedding(
            num_spatial_types + 1, num_heads)
        self.edge_dis_encoder = torch.nn.Embedding(
            num_spatial_types * num_heads * num_heads, 1)
        self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads)

        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            # self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        self.edge_encoder.weight.data.normal_(std=0.02)
        self.edge_dis_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        """Computes the bias matrix that can be induced into multi-head attention
        via the attention mask.
        Adds the tensor ``attn_bias`` to the data object, optionally accounting
        for the graph token.
        """
        # To convert 2D matrices to dense-batch mode, one needs to decompose
        # them into index and value. One example is the adjacency matrix
        # but this generalizes actually to any 2D matrix
        spatial_types: torch.Tensor = self.spatial_encoder(data.spatial_types)
        spatial_encodings = to_dense_adj(data.graph_index,
                                         data.batch,
                                         spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        if hasattr(data, "shortest_path_types"):
            edge_types: torch.Tensor = self.edge_encoder(data.shortest_path_types)
            edge_encodings = to_dense_adj(data.graph_index,
                                          data.batch,
                                          edge_types)

            spatial_distances = to_dense_adj(data.graph_index,
                                             data.batch,
                                             data.spatial_types)
            spatial_distances = spatial_distances.float().clamp(min=1.0).unsqueeze(1)

            B, N, _, max_dist, H = edge_encodings.shape

            edge_encodings = edge_encodings.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_encodings = torch.bmm(edge_encodings, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads))
            edge_encodings = edge_encodings.reshape(max_dist, B, N, N, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_encodings = edge_encodings.sum(-2).permute(BATCH_HEAD_NODE_NODE) / spatial_distances
            bias += edge_encodings

        if self.use_graph_token:
            raise NotImplementedError
            # bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            # bias[:, :, 1:, 0] = self.graph_token
            # bias[:, :, 0, :] = self.graph_token

        return bias.permute((0, 2, 3, 1))


class NodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = False):
        """Implementation of the node encoder of Graphormer.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.
        Args:
            embed_dim: The number of hidden dimensions of the model
            num_in_degree: Maximum size of in-degree to encode
            num_out_degree: Maximum size of out-degree to encode
            input_dropout: Dropout applied to the input features
            use_graph_token: If True, adds the graph token to the incoming batch.
        """
        super().__init__()
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)

        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            # self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
            raise NotImplementedError
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.reset_parameters()

    def forward(self, x, data):
        in_degree_encoding = self.in_degree_encoder(data.in_degrees)
        out_degree_encoding = self.out_degree_encoder(data.out_degrees)

        x = x + in_degree_encoding + out_degree_encoding

        if self.use_graph_token:
            # data = add_graph_token(data, self.graph_token)
            raise NotImplementedError
        x = self.input_dropout(x)
        return x

    def reset_parameters(self):
        self.in_degree_encoder.weight.data.normal_(std=0.02)
        self.out_degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)


class GraphormerLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float,
                 attention_dropout: float, mlp_dropout: float):
        """
        https://github.com/rampasek/GraphGPS/blob/28015707cbab7f8ad72bed0ee872d068ea59c94b/graphgps/layer/graphormer_layer.py#L5
        """
        super().__init__()
        # self.attention = torch.nn.MultiheadAttention(embed_dim,
        #                                              num_heads,
        #                                              attention_dropout,
        #                                              batch_first=True)
        self.attention = AttentionLayer(embed_dim, embed_dim, num_heads, attention_dropout)
        self.input_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        # We follow the paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, data):
        x = self.input_norm(x)
        x, real_nodes = to_dense_batch(x, data.batch)

        x = self.attention(x, ~real_nodes, attn_bias)[0][real_nodes]
        x = self.dropout(x) + x
        x = self.mlp(x) + x
        return x

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.input_norm.reset_parameters()
        reset_sequential_parameters(self.mlp)


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden, head, attention_dropout):
        super(AttentionLayer, self).__init__()

        self.head_dim = hidden // head
        assert self.head_dim * head == hidden
        self.head = head
        self.attention_dropout = attention_dropout

        self.w_q = torch.nn.Linear(in_dim, hidden)
        self.w_k = torch.nn.Linear(in_dim, hidden)
        self.w_v = torch.nn.Linear(in_dim, hidden)
        self.w_o = torch.nn.Linear(hidden, hidden)

    def forward(self, x, key_pad: torch.BoolTensor = None, attn_mask: torch.FloatTensor = None):
        # x: batch, Nmax, F
        bsz, Nmax, feature = x.shape
        k = self.w_k(x).reshape(bsz, Nmax, self.head_dim, self.head)
        q = self.w_q(x).reshape(bsz, Nmax, self.head_dim, self.head)
        v = self.w_v(x).reshape(bsz, Nmax, self.head_dim, self.head)

        attention_score = torch.einsum('bnfh,bmfh->bnmh', k, q) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attention_score += attn_mask
        if key_pad is not None:
            attention_score -= key_pad[:, None, :, None].to(torch.float) * LARGE_NUMBER

        softmax_attn_score = torch.softmax(attention_score, dim=2)

        softmax_attn_score = torch.nn.functional.dropout(softmax_attn_score, p=self.attention_dropout, training=self.training)
        v = torch.einsum('bnmh,bmfh->bnfh', softmax_attn_score, v).reshape(bsz, Nmax, self.head * self.head_dim)
        out = self.w_o(v)

        return out, attention_score

    def reset_parameters(self):
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()


class Graphormer(torch.nn.Module):

    def __init__(self,
                 encoder,
                 bias_conder,
                 node_encoder,
                 hidden,
                 layers,
                 num_heads,
                 ensemble,
                 dropout,
                 attn_dropout,
                 mlp_dropout):
        super(Graphormer, self).__init__()

        self.encoder = encoder
        self.bias_conder = bias_conder
        self.node_encoder = node_encoder
        self.num_heads = num_heads
        self.tf_layers = torch.nn.ModuleList([])
        for l in range(layers):
            self.tf_layers.append(GraphormerLayer(hidden,
                                                  num_heads,
                                                  dropout,
                                                  attn_dropout,
                                                  mlp_dropout))

        self.self_attn = AttentionLayer(hidden, hidden, num_heads, 0.)
        self.head2ensemble = torch.nn.Linear(num_heads, ensemble, bias=False)

    def forward(self, batch):
        x = self.encoder(batch.x)

        atten_bias = self.bias_conder(batch)
        x = self.node_encoder(x, batch)

        for l in self.tf_layers:
            x = l(x, atten_bias, batch)
        x, mask = to_dense_batch(x, batch.batch)
        attention_score = self.self_attn(x, None, atten_bias)[1]
        attention_score = self.head2ensemble(attention_score)
        real_node_node_mask = torch.einsum('bn,bm->bnm', mask, mask)
        return attention_score, real_node_node_mask

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.bias_conder.reset_parameters()
        self.node_encoder.reset_parameters()
        for l in self.tf_layers:
            l.reset_parameters()
        self.self_attn.reset_parameters()
        self.head2ensemble.reset_parameters()
