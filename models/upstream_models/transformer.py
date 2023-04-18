from ml_collections import ConfigDict

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch

from .kernel_encoder import RWSENodeEncoder
from .lap_encoder import LapPENodeEncoder
from .my_attention_layer import AttentionLayer
from models.nn_utils import BiEmbedding


class FeatureEncoder(torch.nn.Module):

    def __init__(self,
                 dim_in,
                 hidden,
                 type_encoder: str,
                 lap_encoder: ConfigDict = None,
                 rw_encoder: ConfigDict = None):
        super(FeatureEncoder, self).__init__()

        lin_hidden = hidden
        if lap_encoder is not None:
            lin_hidden -= lap_encoder.dim_pe
        if rw_encoder is not None:
            lin_hidden -= rw_encoder.dim_pe

        if type_encoder == 'linear':
            self.linear_embed = nn.Linear(dim_in, lin_hidden)
        elif type_encoder == 'bi_embedding':
            self.linear_embed = BiEmbedding(dim_in, lin_hidden)
        elif type_encoder == 'embedding':
            # https://github.com/rampasek/GraphGPS/blob/28015707cbab7f8ad72bed0ee872d068ea59c94b/graphgps/encoder/type_dict_encoder.py#L82
            raise NotImplementedError
        else:
            raise ValueError

        if lap_encoder is not None:
            self.lap_encoder = LapPENodeEncoder(hidden,
                                                hidden - rw_encoder.dim_pe,
                                                lap_encoder,
                                                expand_x=False)
        else:
            self.lap_encoder = None

        if rw_encoder is not None:
            self.rw_encoder = RWSENodeEncoder(hidden, hidden, rw_encoder, expand_x=False)
        else:
            self.rw_encoder = None

    def forward(self, batch):
        x = self.linear_embed(batch.x)
        if self.lap_encoder is not None:
            x = self.lap_encoder(x, batch)
        if self.rw_encoder is not None:
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
                 batch_norm,
                 use_spectral_norm):
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
                                                   batch_norm,
                                                   use_spectral_norm))

        self.self_attn = AttentionLayer(hidden, hidden, ensemble, 0., use_spectral_norm)

    def forward(self, batch):
        x = self.encoder(batch)
        for l in self.tf_layers:
            x = l(x, batch)
        x, mask = to_dense_batch(x, batch.batch)
        # batchsize, head, Nmax, Nmax
        attention_score = self.self_attn(x)[1]
        real_node_node_mask = torch.einsum('bn,bm->bnm', mask, mask)
        return attention_score, real_node_node_mask

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for l in self.tf_layers:
            l.reset_parameters()
        self.self_attn.reset_parameters()
