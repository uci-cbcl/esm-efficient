from typing import Tuple
import torch
import torch.nn as nn
from einops import repeat, rearrange
from flash_attn import flash_attn_varlen_func


class PartitionMeanPool(nn.Module):
    '''
    PartitionMeanPool is a pooling layer that averages the embeddings of each
    partition in a sequence. The partition is defined by the cumulative lens
    cu_lens.

    Example:
    >>> embed = torch.tensor([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9],
    ...     [10, 11, 12],
    ...     [13, 14, 15],
    ...     [16, 17, 18],
    ...     [19, 20, 21],
    ... ])
    >>> cu_lens = torch.tensor([0, 3, 5, 7])
    >>> pool = PartitionMeanPool()
    >>> pool(embed, cu_lens)    
    '''

    def __init__(self):
        super().__init__()

    def forward(self, embed, cu_lens):
        return partition_mean_pool(embed, cu_lens)

    @staticmethod
    def _indices(cu_lens):
        index = torch.zeros(
            cu_lens[-1], dtype=torch.long, device=cu_lens.device)
        for i in range(1, len(cu_lens)):
            index[cu_lens[i - 1]:cu_lens[i]] = i - 1
        return index


def partition_mean_pool(embed, cu_lens):
    '''
    PartitionMeanPool is a pooling layer that averages the embeddings of each
    partition in a sequence. The partition is defined by the cumulative lens
    cu_lens.

    Example:
    >>> embed = torch.tensor([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9],
    ...     [10, 11, 12],
    ...     [13, 14, 15],
    ...     [16, 17, 18],
    ...     [19, 20, 21],
    ... ])
    >>> cu_lens = torch.tensor([0, 3, 5, 7])
    >>> partition_mean_pool(embed, cu_lens)    
    '''
    pooled = torch.zeros(
        (len(cu_lens) - 1, embed.shape[1]), dtype=embed.dtype, device=embed.device
    )
    lens = (cu_lens[1:] - cu_lens[:-1]).unsqueeze(1).to(embed.dtype)
    indices = PartitionMeanPool._indices(cu_lens)

    return pooled.index_add_(0, indices, embed) / lens


class AttentionPool(nn.Module):

    def __init__(self, attention_heads: int, embed_dim: int, dropout_p=0.0, 
                 dtype=torch.bfloat16):
        super().__init__()
        self.attention_heads = attention_heads
        self.dropout_p = dropout_p
        self.k = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        # self.cls = nn.Parameter(torch.ones(1, in_dim, dtype=torch.bfloat16))

    def forward(self, cls: torch.Tensor, embed: torch.Tensor, pad_args: Tuple[torch.Tensor, int]):
        """
        Applies attention pooling to the input embeddings using the class tokens
        as queries. The function computes the attention scores between the class tokens
        and the embeddings, and returns the pooled embeddings.

        Args:
            embed: (t, d) tensor of embeddings
            cls: (c, d) tensor of class tokens
            cu_lens: (n,) tensor of cumulative lengths
            max_len: maximum length of the sequence
        Returns:
            (n, c, d) tensor of pooled embeddings

        Example:
        >>> embed = torch.randn(10, 64)
        >>> cls = torch.randn(1, 64)
        >>> cu_lens = torch.tensor([0, 3, 5, 7])
        >>> max_len = 10
        >>> attn_pool = AttentionPool(4, 64)
        >>> output = attn_pool(embed, cls, cu_lens, max_len)
        >>> print(output.shape)
        torch.Size([2, 1, 64])
        """
        cu_lens, max_len = pad_args

        n_seq, n_cls = len(cu_lens) - 1, cls.shape[0]

        k = self.k(embed)
        k = repeat(k, 't (h d) -> (m t) h d', m=n_cls, h=self.attention_heads)

        q = repeat(cls, 'c (h d) -> (c m) h d', m=n_seq, h=self.attention_heads)
        cu_lens_q = torch.arange(0, n_seq * n_cls + 1,
                                 dtype=torch.int32, device=q.device)

        v = repeat(embed, 't (h d) -> (m t) h d', m=n_cls, h=self.attention_heads)

        cu_lens = torch.cat([
            torch.tensor([0], device=cu_lens.device, dtype=torch.int32),
            (cu_lens[1:] - cu_lens[:-1]
             ).repeat(n_cls).cumsum(0, dtype=torch.int32)
        ])

        attn = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_lens_q,
            cu_seqlens_k=cu_lens,
            max_seqlen_q=1,
            max_seqlen_k=max_len,
            dropout_p=self.dropout_p,
            causal=False
        )
        return rearrange(attn, '(c s) h d -> s c (h d)',  c=n_cls, s=n_seq, 
                         h=self.attention_heads)


class LearnedAttentionPool(AttentionPool):
    """
    LearnedAttentionPool is a pooling layer that uses learned class tokens
    as queries for attention pooling. The class tokens are learned during
    training and are used to compute the attention scores between the class
    tokens and the embeddings. The function returns the pooled embeddings.

    Args:
        num_cls: number of class tokens
        attention_heads: number of attention heads
        embed_dim: dimension of the embeddings
        dropout_p: dropout probability
        dtype: data type of the parameters

    Example:
    >>> embed = torch.randn(10, 64)
    >>> cu_lens = torch.tensor([0, 3, 5, 10])
    >>> max_len = 5
    >>> attn_pool = LearnedAttentionPool(4, 4, 64)
    >>> output = attn_pool(embed, (cu_lens, max_len))
    """

    def __init__(self, num_cls, attention_heads, embed_dim: int, dropout_p=0.0, 
                 dtype=torch.bfloat16):
        super().__init__(attention_heads, embed_dim, dropout_p, dtype)
        self.cls = nn.Parameter(torch.ones(num_cls, embed_dim, dtype=dtype))
        # self.reset_parameters()

    def forward(self, embed: torch.Tensor, pad_args: Tuple[torch.Tensor, int]):
        """
        Applies attention pooling to the input embeddings using the learned class tokens
        as queries. The function computes the attention scores between the class tokens
        and the embeddings, and returns the pooled embeddings.

        Args:
            embed: (t, d) tensor of embeddings
            cu_lens: (n,) tensor of cumulative lengths
            max_len: maximum length of the sequence
        Returns:
            (n, c, d) tensor of pooled embeddings
        """
        return super().forward(self.cls, embed, pad_args)

    # def reset_parameters(self):
    #     nn.init.trunc_normal_(self.cls, std=1)


class LearnedAggregation(nn.Module):
    """
    Learned Aggregation  is a pooling layer that uses learned class tokens
    as queries for attention pooling. The class tokens are learned during
    training and are used to compute the attention scores between the class
    tokens and the embeddings. The function returns the pooled embeddings.
    
    Args:
        num_cls: number of class tokens
        attention_heads: number of attention heads
        embed_dim: dimension of the embeddings
        dropout_p: dropout probability
        dtype: data type of the parameters

    Example:
    >>> embed = torch.randn(10, 64)
    >>> cu_lens = torch.tensor([0, 3, 5, 10])
    >>> max_len = 5
    >>> attn_pool = LearnedAggregation(4, 4, 64)
    >>> output = attn_pool(embed, (cu_lens, max_len))
    >>> print(output.shape)
    torch.Size([2, 1])
    """

    def __init__(self, num_cls, attention_heads: int, embed_dim: int, 
                 dropout_p=.0, dtype=torch.bfloat16):
        super().__init__()
        # self.gamma = nn.Parameter(1e-4 * torch.ones(num_cls, embed_dim, dtype=dtype))
        self.attn = LearnedAttentionPool(num_cls, attention_heads, embed_dim, 
                                         dropout_p=dropout_p, dtype=dtype)
        self.linear = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.relu = nn.ReLU()
        self.final = nn.Linear(embed_dim, 1, dtype=dtype)

    def forward(self, embed: torch.Tensor, pad_args: Tuple[torch.Tensor, int]):
        # x = cls + self.gamma_1 * self.attn(cls, embed, pad_args)
        # return self.final(x + self.gamma_2 * self.ffn(x)).squeeze(-1)

        # x = self.attn.cls + self.gamma * self.attn(embed, pad_args)

        x = self.attn(embed, pad_args)
        x = self.final(self.relu(self.linear(x)))
        return x.squeeze(1)


class BinaryLearnedAggregation(LearnedAggregation):

    def __init__(self, attention_heads: int, embed_dim: int, dropout_p=0.0,
                 dtype=torch.bfloat16):
        super().__init__(1, attention_heads, embed_dim, dropout_p, dtype)

    def forward(self, embed: torch.Tensor, pad_args: Tuple[torch.Tensor, int]):
        return super().forward(embed, pad_args).squeeze(-1)