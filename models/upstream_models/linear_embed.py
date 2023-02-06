import pdb
from typing import Union

import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor

from models.my_encoder import AtomEncoder, BondEncoder
from models.my_convs import GINEConv
from models.nn_utils import MLP


class LinearEmbed(torch.nn.Module):
    def __init__(self, in_features,
                 edge_features,
                 hid_size,
                 gnn_layer,
                 mlp_layer,
                 dropout=0.5,
                 ensemble=1,
                 use_bn=False,
                 use_ogb_encoder=True):
        super(LinearEmbed, self).__init__()
        if use_ogb_encoder:
            self.atom_encoder = AtomEncoder(emb_dim=hid_size)
            self.bond_encoder = BondEncoder(emb_dim=hid_size)
        else:
            self.atom_encoder = Linear(in_features, hid_size)
            self.bond_encoder = Linear(edge_features, hid_size)
        self.dropout = dropout

        self.gnn = torch.nn.ModuleList([GINEConv(
                    hid_size,
                    Sequential(
                        Linear(hid_size, hid_size),
                        ReLU(),
                        Linear(hid_size, hid_size),
                        BN(hid_size),
                        ReLU(),
                    ),
                    bond_encoder=MLP([hid_size, hid_size, hid_size], norm=False, dropout=0.),)
                for _ in range(gnn_layer)])

        # self.node_emb = Linear(hid_size, hid_size)
        # self.edge_emb = Linear(hid_size, hid_size)

        self.mlp = MLP([hid_size * 3] + [hid_size] * (mlp_layer - 1) + [ensemble], norm=use_bn, dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        ptr = data.ptr.cpu().numpy() if hasattr(data, 'ptr') else np.zeros(1, dtype=np.int32)
        nnodes = data.nnodes.cpu().numpy()
        idx = np.concatenate([np.triu_indices(n, -n) + ptr[i] for i, n in enumerate(nnodes)], axis=-1)
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        for gnn in self.gnn:
            x = gnn(x, edge_index, edge_attr, None)

        # emb_n = self.node_emb(x)
        # emb_e = self.edge_emb(edge_attr)
        emb_e = SparseTensor.from_edge_index(edge_index,
                                             edge_attr,
                                             sparse_sizes=(data.num_nodes, data.num_nodes),
                                             is_sorted=True).to_dense()

        emb_e = emb_e[idx[0], idx[1]]

        emb = torch.cat((x[idx[0]], x[idx[1]], emb_e), dim=-1)
        emb = self.mlp(emb)

        return emb

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.bond_encoder.reset_parameters()
        # self.node_emb.reset_parameters()
        # self.edge_emb.reset_parameters()
        self.mlp.reset_parameters()
        for gnn in self.gnn:
            gnn.reset_parameters()
