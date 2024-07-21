# credits to OGB team
# https://github.com/snap-stanford/ogb/blob/f83d9724112c06ce230cbffa57202155b2c7a06c/ogb/graphproppred/mol_encoder.py

import torch
from ml_collections import ConfigDict
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred.mol_encoder import AtomEncoder

from torch import nn as nn

from torch_geometric.nn import MLP

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


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
        elif type_encoder == 'bi_embedding_cat':
            assert lin_hidden % 2 == 0, 'lin_hidden must be even'
            # n_features hardcoded right now
            self.linear_embed = BiEmbedding_cat(n_nodes=dim_in, n_features=2, hidden=lin_hidden//2)
        elif type_encoder == 'atomencoder':
            self.linear_embed = AtomEncoder(lin_hidden)
        elif type_encoder == 'embedding':
            # https://github.com/rampasek/GraphGPS/blob/28015707cbab7f8ad72bed0ee872d068ea59c94b/graphgps/encoder/type_dict_encoder.py#L82
            raise NotImplementedError
        else:
            raise ValueError

        if lap_encoder is not None:
            self.lap_encoder = LapPENodeEncoder(hidden,
                                                hidden - rw_encoder.dim_pe if rw_encoder is not None else hidden,
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


class LapPENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/laplace_pos_encoder.py
    """Laplace Positional Embedding node encoder.
    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()

        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        max_freqs = pecfg.max_freqs  # Num. eigenvectors (frequencies)

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if pecfg.raw_norm_type is None or pecfg.raw_norm_type == 'None':
            self.raw_norm = None
        elif pecfg.raw_norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            raise ValueError

        self.pe_encoder = MLP([max_freqs] + (n_layers - 1) * [2 * dim_pe] + [dim_pe], dropout=0.)


    def forward(self, x, batch):
        if not hasattr(batch, 'EigVecs'):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        pos_enc = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(pos_enc.size(1), device=pos_enc.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            pos_enc = pos_enc * sign_flip.unsqueeze(0)

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        return x


class KernelPENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/kernel_pos_encoder.py
    """Configurable kernel-based Positional Encoding node encoder.
    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.
    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = pecfg.kernel
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        n_layers = pecfg.layers

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        self.pe_encoder = MLP([num_rw_steps] + (n_layers - 1) * [2 * dim_pe] + [dim_pe], dropout=0.)

    def forward(self, x, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        return x


class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'


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


class BiEmbedding_cat(torch.nn.Module):
    def __init__(self,
                 n_nodes,
                 n_features,
                 hidden,):
        super(BiEmbedding_cat, self).__init__()
        self.emb_node = nn.Embedding(num_embeddings=n_nodes, embedding_dim=hidden)
        self.emb_feature = nn.Embedding(num_embeddings=n_features, embedding_dim=hidden)

    def forward(self, x):
        x_node, x_feature = x[:, 0], x[:, 1]
        node_emb = self.emb_node(x_node)
        feature_emb = self.emb_feature(x_feature)
        x = torch.cat([node_emb, feature_emb], dim=-1)
        return x
