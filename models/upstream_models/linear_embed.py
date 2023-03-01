from typing import Union

import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from models.my_encoder import AtomEncoder, BondEncoder
from models.my_convs import GINEConv
from models.nn_utils import MLP


class LinearEmbed(torch.nn.Module):
    def __init__(self,
                 tuple_type,
                 heads,
                 in_features,
                 edge_features,
                 hid_size,
                 gnn_layer,
                 mlp_layer,
                 dropout=0.5,
                 emb_edge=True,
                 emb_spd=False,
                 emb_ppr=False,
                 ensemble=1,
                 use_bn=False,
                 use_ogb_encoder=True):
        super(LinearEmbed, self).__init__()

        self.emb_edge = emb_edge
        self.emb_spd = emb_spd
        self.emb_ppr = emb_ppr

        self.p_list = []

        if use_ogb_encoder:
            self.atom_encoder = AtomEncoder(emb_dim=hid_size)
            self.bond_encoder = BondEncoder(emb_dim=hid_size)
        else:
            self.atom_encoder = Linear(in_features, hid_size)
            self.bond_encoder = Linear(edge_features, hid_size)

        # don't regularize these params
        self.p_list.append({'params': self.atom_encoder.parameters(), 'weight_decay': 0.})
        self.p_list.append({'params': self.bond_encoder.parameters(), 'weight_decay': 0.})

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
            # don't regularize these params
            self.p_list.append({'params': self.gnn[-1].parameters(), 'weight_decay': 0.})

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

        mlp_in_size += int(emb_edge) + int(emb_spd) + int(emb_ppr)

        self.mlp = MLP([mlp_in_size] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       layer_norm=use_bn, dropout=dropout)
        # regularize these params
        self.p_list.append({'params': self.mlp.parameters(),})

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        # ptr = data.ptr.cpu().numpy() if hasattr(data, 'ptr') else np.zeros(1, dtype=np.int32)
        # nnodes = data.nnodes.cpu().numpy()
        # idx = np.concatenate([np.triu_indices(n, -n) + ptr[i] for i, n in enumerate(nnodes)], axis=-1)
        # idx_no_acum = np.concatenate([np.triu_indices(n, -n) for i, n in enumerate(nnodes)], axis=-1)
        x = self.atom_encoder(data.x)
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
            # emb_e = SparseTensor.from_edge_index(edge_index,
            #                                      sparse_sizes=(data.num_nodes, data.num_nodes),
            #                                      is_sorted=True).to_dense()
            # emb.append(emb_e[idx[0], idx[1]][:, None])
            raise NotImplementedError("see fwl2 how to embed edge")

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
