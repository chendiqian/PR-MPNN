from typing import Union

import torch
from torch.nn import Linear, LayerNorm, ModuleList
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from models.my_encoder import AtomEncoder, BondEncoder
from models.nn_utils import MLP


class Fwl2Embed(torch.nn.Module):
    def __init__(self, in_features,
                 edge_features,
                 hid_size,
                 fwl_layer,
                 mlp_layer,
                 dropout=0.5,
                 emb_edge=True,
                 emb_spd=False,
                 ensemble=1,
                 use_norm=False,
                 use_ogb_encoder=True):
        super(Fwl2Embed, self).__init__()

        self.emb_edge = emb_edge
        self.emb_spd = emb_spd
        self.dropout = dropout

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

        # 2FWL layers
        fwl_in_size = hid_size * 2
        if emb_edge:
            fwl_in_size += hid_size
        if emb_spd:
            fwl_in_size += hid_size
            self.spd_encoder = Linear(1, hid_size)

        self.fwl = ModuleList([Linear(2 * fwl_in_size, hid_size)])
        for i in range(fwl_layer - 1):
            self.fwl.append(Linear(hid_size * 2, hid_size))
        self.p_list.append({'params': list(self.fwl.parameters()), 'weight_decay': 0.})
        if use_norm:
            self.norms = ModuleList([LayerNorm(hid_size) for _ in range(fwl_layer)])
            self.p_list.append({'params': list(self.norms.parameters()), 'weight_decay': 0.})
        else:
            self.norms = None

        # MLP layers
        self.mlp = MLP([hid_size] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       layer_norm=use_norm, dropout=dropout)
        # don't regularize these params
        self.p_list.append({'params': list(self.mlp.parameters())[:-1], 'weight_decay': 0.})
        # regularize these params
        self.p_list.append({'params': list(self.mlp.parameters())[-1],})

    def forward(self, data: Union[Data, Batch]):
        x = self.atom_encoder(data.x)
        x, real_nodes = to_dense_batch(x, data.batch)  # batchsize, Nmax, F
        nnodes_max = x.shape[1]
        x = torch.cat((x[:, :, None, :].repeat(1, 1, nnodes_max, 1),
                       x[:, None, :, :].repeat(1, nnodes_max, 1, 1)), dim=-1)  # batchsize, Nmax, Nmax, 2F

        if self.emb_edge:
            # edge_attr = self.bond_encoder(data.edge_attr)
            # emb_e = SparseTensor.from_edge_index(data.edge_index,
            #                                      data.edge_attr,
            #                                      sparse_sizes=(data.num_nodes, data.num_nodes),
            #                                      is_sorted=True).to_dense()
            # emb.append(emb_e)
            raise NotImplementedError

        if self.emb_spd:
            # spd = data.g_dist_mat[idx[0], idx_no_acum[1]]
            # emb.append(self.spd_encoder(spd[:, None]))
            raise NotImplementedError

        for i, fwl in enumerate(self.fwl):
            x = torch.einsum('bijl,bjkl->bikl', x, x)
            x = torch.cat([x, x], dim=-1)
            x = fwl(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        emb = self.mlp(x)
        real_nodes = (real_nodes[:, :, None] * real_nodes[:, None, :])
        return emb[real_nodes]

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.bond_encoder.reset_parameters()
        self.mlp.reset_parameters()
        for fwl in self.fwl:
            fwl.reset_parameters()
        if self.norms is not None:
            for n in self.norms:
                n.reset_parameters()
        if self.emb_spd:
            self.spd_encoder.reset_parameters()
