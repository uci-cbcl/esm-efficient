import torch
import torch.nn as nn
import torch.nn.functional as F
from esme.alphabet import Alphabet


class LearnedPositionalEmbedding(nn.Embedding):
    '''
    Learned positional embedding with padding. The padding index is 1. The position indices are
    computed by cumulatively summing the non-padding elements in the input tensor.

    Args:
        num_embeddings: int - the number of embeddings.
        embedding_dim: int - the dimension of the embeddings.
        dtype: torch.dtype - the data type of the embeddings.

    Example:
        >>> num_embeddings = 33
        >>> embedding_dim = 4096
        >>> learned_positional_embedding = LearnedPositionalEmbedding(
        ...     num_embeddings, embedding_dim, torch.bfloat16)
        >>> x = torch.tensor([[20, 29, 28], [8, 13, 9]])
        >>> pos_embed = learned_positional_embedding.positions(x)
        >>> print(pos_embed)
        ... tensor([[2, 3, 4],
        ...         [2, 3, 4]])
    '''

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 dtype=torch.bfloat16):
        num_embeddings_ = num_embeddings + 2
        super().__init__(num_embeddings_, embedding_dim,
                         Alphabet.padding_idx, dtype=dtype)
        self.max_positions = num_embeddings

    def positions(self, input: torch.Tensor):
        '''
        Position indices for input sequences with padding.

        Args:
            input: torch.Tensor - the input tensor with shape (batch_size, seq_len).

        Returns:
            torch.Tensor - the position indices for the input tensor with shape (batch_size, seq_len). 
        '''
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        pad = input.ne(self.padding_idx).int()
        return (torch.cumsum(pad, dim=1).type_as(pad) * pad).long() + self.padding_idx

    def position_unpad(self, input: torch.Tensor, pad_args):
        '''
        Position indices for un-padded input sequences.

        Args:
            input: torch.Tensor - the input tensor with shape (seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences.

        Returns:
            torch.Tensor - the position indices for the input tensor with shape (seq_len).
        '''
        assert input.ndim == 1

        cu_lens, max_len = pad_args

        if max_len > self.max_positions:
            raise ValueError(
                f"Sequence length {max_len} above maximum "
                f" sequence length of {self.max_positions}"
            )

        lens = cu_lens[1:] - cu_lens[:-1]

        return torch.cat([
            torch.arange(1, l + 1, device=input.device)
            for l in lens
        ], dim=0).long() + self.padding_idx

    def forward(self, input: torch.Tensor, pad_args=None):
        '''
        Forward pass through the embedding layer.

        Args:
            input: torch.Tensor - the input tensor with shape (batch_size, seq_len).
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences.

        Returns:
            torch.Tensor - the embeddings for the input tensor with 
                shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim).
        '''
        pos_embed = self.positions(input) if pad_args is None \
            else self.position_unpad(input, pad_args)

        return F.embedding(
            pos_embed,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
