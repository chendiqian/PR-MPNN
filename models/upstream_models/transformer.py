from ml_collections import ConfigDict

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch

from .kernel_encoder import RWSENodeEncoder
from .lap_encoder import LapPENodeEncoder


class FeatureEncoder(torch.nn.Module):

    def __init__(self, dim_in, hidden, type_encoder: str, lap_encoder: ConfigDict = None, rw_encoder: ConfigDict = None):
        super(FeatureEncoder, self).__init__()

        if type_encoder == 'linear':
            lin_hidden = hidden
            if lap_encoder is not None:
                lin_hidden -= lap_encoder.dim_pe
            if rw_encoder is not None:
                lin_hidden -= rw_encoder.dim_pe
            self.linear_embed = nn.Linear(dim_in, lin_hidden)
        elif type_encoder == 'embedding':
            # https://github.com/rampasek/GraphGPS/blob/28015707cbab7f8ad72bed0ee872d068ea59c94b/graphgps/encoder/type_dict_encoder.py#L82
            raise NotImplementedError
        else:
            raise ValueError

        if lap_encoder is not None:
            self.lap_encoder = LapPENodeEncoder(hidden, hidden - rw_encoder.dim_pe, lap_encoder, expand_x=False)

        if rw_encoder is not None:
            self.rw_encoder = RWSENodeEncoder(hidden, hidden, rw_encoder, expand_x=False)

        # if cfg.dataset.node_encoder_bn:
        #     self.node_encoder_bn = BatchNorm1dNode(
        #         new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
        #                          has_bias=False, cfg=cfg))
        # # Update dim_in to reflect the new dimension of the node features
        # self.dim_in = cfg.gnn.dim_inner
        # if cfg.dataset.edge_encoder:
        #     # Hard-limit max edge dim for PNA.
        #     if 'PNA' in cfg.gt.layer_type:
        #         cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
        #     else:
        #         cfg.gnn.dim_edge = cfg.gnn.dim_inner
        #     # Encode integer edge features via nn.Embeddings
        #     EdgeEncoder = register.edge_encoder_dict[
        #         cfg.dataset.edge_encoder_name]
        #     self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
        #     if cfg.dataset.edge_encoder_bn:
        #         self.edge_encoder_bn = BatchNorm1dNode(
        #             new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
        #                              has_bias=False, cfg=cfg))

    def forward(self, batch):
        x = self.linear_embed(batch.x)
        x = self.lap_encoder(x, batch)
        x = self.rw_encoder(x, batch)
        return x

    def reset_parameters(self):
        self.linear_embed.reset_parameters()
        self.lap_encoder.reset_parameters()
        self.rw_encoder.reset_parameters()


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
                 batch_norm=True,):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # Local message-passing model.
        self.local_model = None

        # Global attention transformer-style model.
        self.self_attn = torch.nn.MultiheadAttention(dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)

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
            h_attn = self._sa_block(h_dense, None, ~mask)[mask]
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

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

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


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden, head):
        super(AttentionLayer, self).__init__()

        self.head_dim = hidden // head
        assert self.head_dim * head == hidden
        self.head = head

        self.w_q = torch.nn.Linear(in_dim, hidden, bias=False)
        self.w_k = torch.nn.Linear(in_dim, hidden, bias=False)

    def forward(self, x):
        # x: batch, Nmax, F
        bsz, Nmax, feature = x.shape
        k = self.w_k(x).reshape(bsz, Nmax, self.head_dim, self.head)
        q = self.w_q(x).reshape(bsz, Nmax, self.head_dim, self.head)

        attention_score = torch.einsum('bnfh,bmfh->bnmh', k, q) / (self.head_dim ** 0.5)

        return attention_score

    def reset_parameters(self):
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()


class Transformer(torch.nn.Module):

    def __init__(self,
                 encoder,
                 hidden,
                 layers,
                 num_heads,
                 ensemble,
                 act,
                 dropout,
                 attn_dropout,
                 layer_norm,
                 batch_norm):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.tf_layers = torch.nn.ModuleList([])
        for l in range(layers):
            self.tf_layers.append(TransformerLayer(hidden,
                                                   num_heads,
                                                   act,
                                                   dropout,
                                                   attn_dropout,
                                                   layer_norm,
                                                   batch_norm))

        self.self_attn = AttentionLayer(hidden, hidden, ensemble)

    def forward(self, batch):
        x = self.encoder(batch)
        for l in self.tf_layers:
            x = l(x, batch)
        x, mask = to_dense_batch(x, batch.batch)
        # batchsize, head, Nmax, Nmax
        attention_score = self.self_attn(x)
        real_node_node_mask = torch.einsum('bn,bm->bnm', mask, mask)
        return attention_score, real_node_node_mask

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for l in self.tf_layers:
            l.reset_parameters()
        self.self_attn.reset_parameters()
