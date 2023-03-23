import torch
from torch.nn.utils import spectral_norm
LARGE_NUMBER = 1.e10


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden, head, attention_dropout, use_spectral_norm = False):
        super(AttentionLayer, self).__init__()

        self.head_dim = hidden // head
        assert self.head_dim * head == hidden
        self.head = head
        self.attention_dropout = attention_dropout

        self.w_q = torch.nn.Linear(in_dim, hidden)
        self.w_k = torch.nn.Linear(in_dim, hidden)
        self.w_v = torch.nn.Linear(in_dim, hidden)
        self.w_o = torch.nn.Linear(hidden, hidden)

        if use_spectral_norm:
            self.w_q = spectral_norm(self.w_q)
            self.w_k = spectral_norm(self.w_k)
            self.w_v = spectral_norm(self.w_v)
            self.w_o = spectral_norm(self.w_o)

    def forward(self, x, key_pad: torch.BoolTensor = None, attn_mask: torch.FloatTensor = None):
        # x: batch, Nmax, F
        bsz, Nmax, feature = x.shape
        k = self.w_k(x).reshape(bsz, Nmax, self.head_dim, self.head)
        q = self.w_q(x).reshape(bsz, Nmax, self.head_dim, self.head)
        v = self.w_v(x).reshape(bsz, Nmax, self.head_dim, self.head)

        attention_score = torch.einsum('bnfh,bmfh->bnmh', k, q) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attention_score += attn_mask
        if key_pad is not None:
            attention_score -= key_pad[:, None, :, None].to(torch.float) * LARGE_NUMBER

        softmax_attn_score = torch.softmax(attention_score, dim=2)

        softmax_attn_score = torch.nn.functional.dropout(softmax_attn_score, p=self.attention_dropout, training=self.training)
        v = torch.einsum('bnmh,bmfh->bnfh', softmax_attn_score, v).reshape(bsz, Nmax, self.head * self.head_dim)
        out = self.w_o(v)

        return out, attention_score

    def reset_parameters(self):
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()
        self.w_v.reset_parameters()
        self.w_o.reset_parameters()
