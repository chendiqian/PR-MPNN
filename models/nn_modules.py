import torch
from torch import nn as nn

LARGE_NUMBER = 1.e10


class BiEmbedding(torch.nn.Module):
    def __init__(self,
                 dim_in,
                 hidden,):
        super(BiEmbedding, self).__init__()
        self.layer0_keys = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)
        self.layer0_values = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)

    def forward(self, x):
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed
        return x


class BiEmbedding_cat(torch.nn.Module):
    def __init__(self,
                 n_nodes,
                 n_features,
                 hidden,):
        super(BiEmbedding_cat, self).__init__()
        self.emb_node = nn.Embedding(num_embeddings=n_nodes, embedding_dim=hidden)
        self.emb_feature = nn.Embedding(num_embeddings=n_features, embedding_dim=hidden)

    def forward(self, x):
        x_node, x_feature = x[:, 0], x[:, 1]
        node_emb = self.emb_node(x_node)
        feature_emb = self.emb_feature(x_feature)
        x = torch.cat([node_emb, feature_emb], dim=-1)
        return x
