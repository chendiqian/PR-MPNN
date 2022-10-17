from typing import Union
from .encoder import AtomEncoder, BondEncoder
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
import torch
import numpy as np


class UpStream(torch.nn.Module):
    def __init__(self, hid_size, num_layer):
        super(UpStream, self).__init__()
        self.num_layer = num_layer
        self.atom_encoder = AtomEncoder(emb_dim=hid_size)
        self.bond_encoder = BondEncoder(emb_dim=hid_size)

        self.node_emb1 = torch.nn.Linear(hid_size, hid_size)
        self.node_emb2 = torch.nn.Linear(hid_size, hid_size)
        self.edge_emb = torch.nn.Linear(hid_size, hid_size)

        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layer - 1):
            if i == 0:
                self.lins.append(torch.nn.Linear(hid_size * 3, hid_size))
            else:
                self.lins.append(torch.nn.Linear(hid_size, hid_size))

            self.bns.append(torch.nn.BatchNorm1d(hid_size))
        self.lins.append(torch.nn.Linear(hid_size, 1))

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        ptr = data.ptr.cpu().numpy()
        nnodes = data.nnodes.cpu().numpy()
        idx = np.concatenate([np.triu_indices(n, -n) + ptr[i] for i, n in enumerate(nnodes)], axis=-1)
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        emb1 = self.node_emb1(x)[idx[0]]
        emb2 = self.node_emb2(x)[idx[1]]
        emb_e = self.edge_emb(edge_attr)
        emb_e = SparseTensor.from_edge_index(edge_index,
                                             emb_e,
                                             sparse_sizes=(data.num_nodes, data.num_nodes),
                                             is_sorted=True).to_dense()

        emb_e = emb_e[idx[0], idx[1]]

        emb = torch.cat((emb1, emb2, emb_e), dim=-1)

        for i in range(self.num_layer):
            emb = self.lins[i](emb)
            if i != self.num_layer - 1:
                emb = self.bns[i](emb)
                emb = torch.relu(emb)

        return emb.squeeze()

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.bond_encoder.reset_parameters()
        self.node_emb1.reset_parameters()
        self.node_emb2.reset_parameters()
        self.edge_emb.reset_parameters()

        for l in self.lins:
            l.reset_parameters()

        for l in self.bns:
            l.reset_parameters()
