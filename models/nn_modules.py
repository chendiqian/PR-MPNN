from typing import List

import torch
from torch import nn as nn
from torch.nn.utils import spectral_norm
from torch_geometric import nn as pygnn
from torch_geometric.utils import to_dense_batch

from models.nn_utils import reset_sequential_parameters
LARGE_NUMBER = 1.e10


class MLP(torch.nn.Module):
    def __init__(self, hidden_dims: List,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 dropout: float = 0.5,
                 activate_last: bool = False):
        super(MLP, self).__init__()

        assert not (batch_norm and layer_norm)   # cannot be both true

        num_layers = len(hidden_dims) - 1
        modules = []
        for i in range(num_layers):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=i < num_layers - 1))
            if batch_norm and i < num_layers - 1:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))
            if layer_norm and i < num_layers - 1:
                modules.append(torch.nn.LayerNorm(hidden_dims[i + 1]))
            if i < num_layers - 1 or activate_last:
                modules.append(torch.nn.ReLU())
                modules.append(torch.nn.Dropout(p=dropout))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

    def reset_parameters(self):
        reset_sequential_parameters(self.mlp)


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

    def reset_parameters(self):
        self.layer0_keys.reset_parameters()
        self.layer0_values.reset_parameters()


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden, head, attention_dropout, use_spectral_norm = False):
        super(AttentionLayer, self).__init__()

        self.head_dim = hidden // head
        assert self.head_dim * head == hidden
        self.head = head
        self.attention_dropout = attention_dropout

        self.w_q = torch.nn.Linear(in_dim, hidden)
        self.w_k = torch.nn.Linear(in_dim, hidden)
        self.w_v = torch.nn.Linear(in_dim, hidden)
        self.w_o = torch.nn.Linear(hidden, hidden)

        if use_spectral_norm:
            self.w_q = spectral_norm(self.w_q)
            self.w_k = spectral_norm(self.w_k)
            self.w_v = spectral_norm(self.w_v)
            self.w_o = spectral_norm(self.w_o)

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
        self.w_v.reset_parameters()
        self.w_o.reset_parameters()


class TransformerLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self,
                 dim_h,
                 num_heads,
                 act='relu',
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 use_spectral_norm=False,):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # Local message-passing model.
        self.local_model = None

        # Global attention transformer-style model.
        self.self_attn = AttentionLayer(dim_h,
                                        dim_h,
                                        num_heads,
                                        attention_dropout=self.attn_dropout,
                                        use_spectral_norm=use_spectral_norm)

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = getattr(torch, act)
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, h, batch):
        h_in1 = h  # for first residual connection

        h_out_list = []

        # MPNN
        if self.local_model is not None:
            raise NotImplementedError

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            h_attn = self.self_attn(h_dense, key_pad=~mask)[0][mask]
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        return h

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def reset_parameters(self):
        self.self_attn._reset_parameters()
        if self.layer_norm or self.batch_norm:
            self.norm1_local.reset_parameters()
            self.norm1_attn.reset_parameters()
            self.norm2.reset_parameters()
        self.ff_linear1.reset_parameters()
        self.ff_linear2.reset_parameters()
