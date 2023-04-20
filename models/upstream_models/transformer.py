import torch
from torch_geometric.utils import to_dense_batch

from ..nn_modules import AttentionLayer, TransformerLayer


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
