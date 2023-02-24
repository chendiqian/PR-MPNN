from typing import Union

import torch
from torch.nn import Linear, LayerNorm, ModuleList
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from models.my_encoder import AtomEncoder, BondEncoder
from models.nn_utils import MLP


class RegularBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, num_mlp_layers=2):
        super().__init__()

        self.mlp1 = MLP([in_features] + [out_features] * num_mlp_layers, dropout=0., activate_last=True)
        self.mlp2 = MLP([in_features] + [out_features] * num_mlp_layers, dropout=0., activate_last=True)
        self.skip = Linear(in_features + out_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        mult = torch.einsum('bijl,bjkl->bikl', mlp1, mlp2)

        out = self.skip(torch.cat([inputs, mult], dim=-1))
        return out

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.skip.reset_parameters()


class Fwl2Embed(torch.nn.Module):
    def __init__(self, in_features,
                 edge_features,
                 hid_size,
                 fwl_layer,
                 mlp_layer,
                 dropout=0.5,
                 emb_edge=False,
                 emb_spd=False,
                 emb_ppr=False,
                 ensemble=1,
                 use_norm=False,
                 use_ogb_encoder=True):
        super(Fwl2Embed, self).__init__()

        self.emb_edge = emb_edge
        self.emb_spd = emb_spd
        self.emb_ppr = emb_ppr
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

        # input head
        fwl_in_size = hid_size
        if emb_edge:
            fwl_in_size += hid_size
        if emb_spd:
            fwl_in_size += hid_size
            self.spd_encoder = Linear(1, hid_size)
        if emb_ppr:
            fwl_in_size += hid_size
            self.ppr_encoder = Linear(1, hid_size)
        self.input_head = MLP([fwl_in_size, hid_size], dropout=0., layer_norm=use_norm, activate_last=True)
        self.p_list.append({'params': self.input_head.parameters(), 'weight_decay': 0.})

        # 2-FWL layers
        self.fwl = ModuleList([RegularBlock(hid_size, hid_size) for _ in range(fwl_layer)])
        self.p_list.append({'params': self.fwl.parameters(), 'weight_decay': 0.})

        # norm layers
        if use_norm:
            self.norms = ModuleList([LayerNorm(hid_size) for _ in range(fwl_layer)])
            self.p_list.append({'params': self.norms.parameters(), 'weight_decay': 0.})
        else:
            self.norms = None

        # MLP layers
        self.mlp = MLP([hid_size] + [hid_size] * (mlp_layer - 1) + [ensemble],
                       layer_norm=use_norm, dropout=dropout)
        # regularize these params
        self.p_list.append({'params': self.mlp.parameters(),})

    def forward(self, data: Union[Data, Batch]):
        x = self.atom_encoder(data.x)
        x, real_nodes = to_dense_batch(x, data.batch)  # batchsize, Nmax, F
        num_graphs, nnodes_max, features = x.shape
        # x = torch.cat((x[:, :, None, :].repeat(1, 1, nnodes_max, 1),
        #                x[:, None, :, :].repeat(1, nnodes_max, 1, 1)), dim=-1)  # batchsize, Nmax, Nmax, 2F

        diag = torch.arange(nnodes_max, device=x.device)
        xs = x.new_zeros(num_graphs, nnodes_max, nnodes_max, features)
        xs[:, diag, diag, :] = x

        xs = [xs]

        if self.emb_edge:
            edge_attr = self.bond_encoder(data.edge_attr)
            num_edges = (data._slice_dict['edge_attr'][1:] - data._slice_dict['edge_attr'][:-1]).to(edge_attr.device)
            graph_idx_mask = torch.repeat_interleave(torch.arange(num_graphs, device=edge_attr.device), num_edges)  # [0, 0, 0, 1, 1, 1, 2, 2, 2, ... batchsize - 1, ..]
            edge_index_rel = torch.repeat_interleave(data._inc_dict['edge_index'].to(edge_attr.device), num_edges)
            local_edge_index = data.edge_index - edge_index_rel
            edge_embeddings = edge_attr.new_zeros(num_graphs, nnodes_max, nnodes_max, edge_attr.shape[-1])
            edge_embeddings[graph_idx_mask, local_edge_index[0], local_edge_index[1]] = edge_attr
            xs.append(edge_embeddings)

        if self.emb_spd:
            spd_mat, _ = to_dense_batch(data.g_dist_mat, data.batch)   # batchsize, Nmax, max_node_dataset
            spd_mat = spd_mat[..., :nnodes_max][..., None]   # batchsize, Nmax, Nmax, 1
            spd_mat = self.spd_encoder(spd_mat)
            xs.append(spd_mat)

        if self.emb_ppr:
            ppr_mat, _ = to_dense_batch(data.ppr_mat, data.batch)  # batchsize, Nmax, max_node_dataset
            ppr_mat = ppr_mat[..., :nnodes_max][..., None]  # batchsize, Nmax, Nmax, 1
            ppr_mat = self.ppr_encoder(ppr_mat)
            xs.append(ppr_mat)

        x = torch.cat(xs, dim=-1)
        x = self.input_head(x)
        for i, fwl in enumerate(self.fwl):
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
        self.input_head.reset_parameters()
        self.mlp.reset_parameters()
        for fwl in self.fwl:
            fwl.reset_parameters()
        if self.norms is not None:
            for n in self.norms:
                n.reset_parameters()
        if self.emb_spd:
            self.spd_encoder.reset_parameters()
        if self.emb_ppr:
            self.ppr_encoder.reset_parameters()
