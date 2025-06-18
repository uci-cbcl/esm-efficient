from einops import repeat, rearrange
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from flash_attn import flash_attn_varlen_func
from esme.alphabet import padding_mask
from esme.data import LabeledDataModule
from esme.pooling import partition_mean_pool

# class Attention1d(nn.Module):

#     def __init__(self, in_dim):
#         super().__init__()
#         self.layer = nn.Linear(in_dim, 1)

#     def forward(self, x, input_mask=None):
#         n, _, _ = x.shape
#         attn = self.layer(x).squeeze(-1)

#         if input_mask is not None:
#             attn = attn.masked_fill_(
#                 ~input_mask.view(n, -1).bool(), float('-inf')
#             )

#         attn = F.softmax(attn, dim=-1).view(n, -1, 1)
#         out = (attn * x).sum(dim=1)

#         return out

# class ESMAttention1d(nn.Module):
#     """Outputs of the ESM model with the attention1d"""

#     def __init__(self, d_embedding):
#         super(ESMAttention1d, self).__init__()
#         self.attention1d = Attention1d(in_dim=d_embedding)  # ???
#         self.linear = nn.Linear(d_embedding, d_embedding)
#         self.relu = nn.ReLU()
#         self.final = nn.Linear(d_embedding, 1)

#     def forward(self, x, pad_args):
#         cu_lens, max_len = pad_args
#         input_mask = padding_mask(cu_lens, max_len)
#         x = self.attention1d(x, input_mask=input_mask)
#         x = self.relu(self.linear(x))
#         x = self.final(x)
#         return x.squeeze(1)

# class Attention1d(nn.Module):

#     def __init__(self, in_dim, dropout_p=0.0):
#         super().__init__()
#         self.layer = nn.Linear(in_dim, in_dim)
#         self.cls = nn.Parameter(torch.ones(1, in_dim, dtype=torch.bfloat16))

#     # def forward(self, embed, pad_args):
#     #     cu_lens, max_len = pad_args

#     #     attn = self.layer(embed)
#     #     _attn = partition_mean_pool(attn, cu_lens)

#     #     # if input_mask is not None:
#     #     #     attn = attn.masked_fill_(
#     #     #         ~input_mask.view(n, -1).bool(), float('-inf')
#     #     #     )

#     #     # attn = F.softmax(attn, dim=-1).view(n, -1, 1)
#     #     out = (attn * x) # .sum(dim=1)
#     #     out = partition_mean_pool(out, cu_lens)

#     #     return out

#     def forward(self, embed, pad_args):
#         cu_lens, max_len = pad_args

#         n_seq, n_cls = len(cu_lens) - 1, self.cls.shape[0]

#         k = rearrange(self.layer(embed), 't (h d) -> t h d', h=20)
#         v = rearrange(embed, 't (h d) -> t h d', h=20)

#         q = repeat(self.cls, 'c (h d) -> (m c) h d', m=n_seq, h=20)
#         cu_lens_q = torch.arange(0, n_seq + 1,
#                                  dtype=torch.int32, device=q.device)

#         attn = flash_attn_varlen_func(
#             q, k, v,
#             cu_seqlens_q=cu_lens_q,
#             cu_seqlens_k=cu_lens,
#             max_seqlen_q=1,
#             max_seqlen_k=max_len,
#             dropout_p=0,  # self.dropout_p,
#             causal=False
#         )
#         return rearrange(attn, 's h d -> s (h d)')


# class ESMAttention1d(nn.Module):
#     """Outputs of the ESM model with the attention1d"""

#     def __init__(self, d_embedding):
#         super(ESMAttention1d, self).__init__()
#         self.attention1d = Attention1d(in_dim=d_embedding)  # ???
#         self.linear = nn.Linear(d_embedding, d_embedding)
#         self.relu = nn.ReLU()
#         self.final = nn.Linear(d_embedding, 1)

#     def forward(self, x, pad_args):
#         x = self.attention1d(x, pad_args)
#         x = self.relu(self.linear(x))
#         x = self.final(x)
#         return x.squeeze(1)


# class AttentionPool(nn.Module):
#     def __init__(self, ndim: int):
#         super().__init__()
#         self.norm = nn.LayerNorm(ndim)
#         self.k = nn.Linear(ndim, ndim, bias=True)
#         self.v = nn.Linear(ndim, ndim, bias=True)
#         self.proj = nn.Linear(ndim, ndim)

#     def forward(self, token: torch.Tensor, cls: torch.Tensor, cu_lens: torch.Tensor, max_len: int):
#         """"""
#         x = self.norm(token)
#         k = self.k(x)
#         v = self.v(x)
#         q = cls.unsqueeze(1)

#         attn_output = flash_attn_varlen_func(
#             q, k, v,
#             cu_seqlens_q=cu_lens,
#             cu_seqlens_k=cu_lens,
#             max_seqlen_q=max_len,
#             max_seqlen_k=max_len,
#             dropout_p=0.0,
#             causal=False
#         )

#         return self.proj(attn_output)


# class LearnedAggregation(nn.Module):
#     """
#     Learned Aggregation from https://arxiv.org/abs/2112.13692
#     https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling
#     """

#     def __init__(self, ndim: int, ffn_expand=4):
#         super().__init__()
#         self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ndim))
#         self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ndim))
#         self.attn = AttentionPool(ndim)
#         self.norm = nn.LayerNorm(ndim)

#         self.ffn = nn.Sequential(
#             nn.Linear(ndim, int(ndim*ffn_expand)),
#             nn.GELU(),
#             nn.Linear(int(ndim*ffn_expand), ndim)
#         )

#         self.apply(self._init_weights)

#     def forward(self, token: torch.Tensor, cls: torch.Tensor, mask=None):
#         x = cls + self.gamma_1 * self.attn(token, cls, mask=mask)
#         return x + self.gamma_2 * self.ffn(self.norm(x))

#     @torch.no_grad()
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=1)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)


# class BinaryLearnedAggregation(LearnedAggregation):

#     def __init__(self, ndim: int):
#         super().__init__(ndim)
#         self.cls = nn.Parameter(torch.ones(1, ndim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.trunc_normal_(self.cls, std=1)

#     def forward(self, token: torch.Tensor, mask=None):
#         return super().forward(token, self.cls, mask=mask).squeeze(1)
