from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor

from models.my_encoder import AtomEncoder, BondEncoder
from models.nn_utils import reset_modulelist_parameters, residual, MLP


class TransformerLayer(torch.nn.Module):
    def __init__(self, in_dim, kq_dim, v_dim, edge_dims, edge_mlp_hid,):
        super(TransformerLayer, self).__init__()

        # Todo: multi head attention
        self.w_q = torch.nn.Linear(in_dim, kq_dim, bias=False)
        self.w_k = torch.nn.Linear(in_dim, kq_dim, bias=False)
        self.w_v = torch.nn.Linear(in_dim, v_dim)

        self.edge_embed = MLP([edge_dims, edge_mlp_hid, 1], norm=False, dropout=0.)
        self.lins = MLP([v_dim, v_dim, v_dim], norm=False)

    def forward(self, x, edge_attr):
        k = self.w_k(x)
        q = self.w_q(x)
        attention_scores = k @ q.T / k.shape[1] ** 0.5 + self.edge_embed(edge_attr).squeeze()
        attention_scores = torch.softmax(attention_scores, dim=1)
        v = attention_scores @ self.w_v(x)
        v = residual(x, v)
        v_new = self.lins(v)
        return residual(v, v_new)

    def reset_parameters(self):
        self.w_v.reset_parameters()
        self.w_q.reset_parameters()
        self.w_v.reset_parameters()
        self.edge_embed.reset_parameters()
        self.lins.reset_parameters()


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, kq_dim, v_dim, edge_mlp_hid, ensemble):
        super(Transformer, self).__init__()

        self.atom_encoder = AtomEncoder(v_dim)
        self.edge_encoder = BondEncoder(v_dim)

        self.transformers = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.transformers.append(TransformerLayer(v_dim, kq_dim, v_dim, v_dim, edge_mlp_hid))

        self.util_head = MLP([v_dim * 2, v_dim, ensemble], norm=False, dropout=0.)

    def forward(self, data: Union[Data, Batch]):
        ptr = data.ptr.cpu().numpy() if hasattr(data, 'ptr') else np.zeros(1, dtype=np.int32)
        nnodes = data.nnodes.cpu().numpy()
        idx = np.concatenate([np.triu_indices(n, -n) + ptr[i] for i, n in enumerate(nnodes)], axis=-1)

        emb_e = self.edge_encoder(data.edge_attr)
        emb_e = SparseTensor.from_edge_index(data.edge_index,
                                             emb_e,
                                             sparse_sizes=(data.num_nodes, data.num_nodes),
                                             is_sorted=True).to_dense()

        x = self.atom_encoder(data.x)
        for t in self.transformers:
            x = t(x, emb_e)
        emb = torch.cat((x[idx[0]], x[idx[1]]), dim=-1)
        return self.util_head(emb)

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        reset_modulelist_parameters(self.transformers)
        self.util_head.reset_parameters()
