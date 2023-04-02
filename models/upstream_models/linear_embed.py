from typing import Union

import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from models.my_convs import GINEConv
from models.nn_utils import MLP


class LinearEmbed(torch.nn.Module):
    def __init__(self,
                 encoder,
                 tuple_type,
                 heads,
                 edge_features,
                 hid_size,
                 gnn_layer,
                 mlp_layer,
                 dropout=0.5,
                 emb_edge=True,
                 emb_spd=False,
                 emb_ppr=False,
                 ensemble=1,
                 use_bn=False):
        super(LinearEmbed, self).__init__()

        self.emb_edge = emb_edge
        self.emb_spd = emb_spd
        self.emb_ppr = emb_ppr

        self.atom_encoder = encoder
        self.bond_encoder = Linear(edge_features, hid_size)

        # don't regularize these params
        # self.p_list.append({'params': self.atom_encoder.parameters(), 'weight_decay': 0.})
        # self.p_list.append({'params': self.bond_encoder.parameters(), 'weight_decay': 0.})

        self.gnn = torch.nn.ModuleList()

        for i in range(gnn_layer):
            if i == gnn_layer - 1:
                seq = Sequential(Linear(hid_size, hid_size), ReLU(), Linear(hid_size, hid_size),)
            else:
                seq = Sequential(
                        Linear(hid_size, hid_size),
                        ReLU(),
                        Linear(hid_size, hid_size),
                        BN(hid_size),
                        ReLU(),
                    )
            self.gnn.append(GINEConv(
                    hid_size,
                    seq,
                    bond_encoder=MLP([hid_size, hid_size, hid_size], dropout=dropout),))

        if tuple_type == 'cat':
            mlp_in_size = hid_size * 2
        elif tuple_type == 'inner':
            mlp_in_size = heads
            self.heads = heads
        elif tuple_type == 'outer':
            mlp_in_size = hid_size
        else:
            raise ValueError(f"Unsupported type {tuple_type}")
        self.tuple_type = tuple_type

        mlp_in_size += int(emb_edge) * hid_size + int(emb_spd) + int(emb_ppr)

        self.mlp = MLP([mlp_in_size] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       layer_norm=use_bn, dropout=dropout)

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        x = self.atom_encoder(data)
        edge_attr = self.bond_encoder(data.edge_attr)

        for gnn in self.gnn:
            x = gnn(x, edge_index, edge_attr, None)

        logits, real = to_dense_batch(x, data.batch)
        real_node_node_mask = torch.einsum('bn,bm->bnm', real, real)
        bsz, Nmax, feature_dims = logits.shape

        emb = []
        if self.tuple_type == 'cat':
            emb.append(logits[:, None, :, :].repeat(1, Nmax, 1, 1))
            emb.append(logits[:, :, None, :].repeat(1, 1, Nmax, 1))
        else:
            if self.tuple_type == 'inner':
                feature_dims = feature_dims // self.heads
                logits = logits.reshape(bsz, Nmax, feature_dims, self.heads)
                attention_mask = torch.einsum('bnfh,bmfh->bnmh', logits, logits) / (feature_dims ** 0.5)
                emb.append(attention_mask)
            elif self.tuple_type == 'outer':
                outer_prod = torch.einsum('bnk,bmk->bnmk', logits, logits)
                emb.append(outer_prod)

        if self.emb_edge:
            num_edges = (data._slice_dict['edge_attr'][1:] - data._slice_dict['edge_attr'][:-1]).to(edge_attr.device)
            graph_idx_mask = torch.repeat_interleave(torch.arange(bsz, device=edge_attr.device), num_edges)  # [0, 0, 0, 1, 1, 1, 2, 2, 2, ... batchsize - 1, ..]
            edge_index_rel = torch.repeat_interleave(data._inc_dict['edge_index'].to(edge_attr.device), num_edges)
            local_edge_index = edge_index - edge_index_rel
            edge_embeddings = edge_attr.new_zeros(bsz, Nmax, Nmax, edge_attr.shape[-1])
            edge_embeddings[graph_idx_mask, local_edge_index[0], local_edge_index[1]] = edge_attr
            emb.append(edge_embeddings)

        if self.emb_spd:
            spd_mat, _ = to_dense_batch(data.g_dist_mat, data.batch)  # batchsize, Nmax, max_node_dataset
            spd_mat = spd_mat[..., :Nmax, None]  # batchsize, Nmax, Nmax, 1
            emb.append(spd_mat)

        if self.emb_ppr:
            ppr, _ = to_dense_batch(data.ppr_mat, data.batch)  # batchsize, Nmax, max_node_dataset
            ppr = ppr[..., :Nmax, None]  # batchsize, Nmax, Nmax, 1
            emb.append(ppr)

        emb = torch.cat(emb, dim=-1)
        emb = self.mlp(emb)

        return emb, real_node_node_mask

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.bond_encoder.reset_parameters()
        self.mlp.reset_parameters()
        for gnn in self.gnn:
            gnn.reset_parameters()
