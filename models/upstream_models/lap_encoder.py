# https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/laplace_pos_encoder.py

import torch
import torch.nn as nn
from models.nn_utils import MLP


class LapPENodeEncoder(torch.nn.Module):
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

    def reset_parameters(self):
        if self.expand_x:
            self.linear_x.reset_parameters()
        if self.raw_norm:
            self.raw_norm.reset_parameters()
        self.pe_encoder.reset_parameters()
