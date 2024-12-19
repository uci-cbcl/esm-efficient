import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from esme.pooling import PartitionMeanPool


class RobertaLMHead(nn.Module):
    """
    Roberta Head to predict amino acid probabilities

    Args:
        embed_dim: int, the dimension of the input embeddings
        output_dim: int, the dimension of the output embeddings
        weight: torch.Tensor, the embedding weights
        dtype: torch.dtype, the datatype to use for the weights
    """

    def __init__(self, embed_dim, vocab_size, dtype=torch.bfloat16):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.layer_norm = nn.LayerNorm(embed_dim, dtype=dtype)
        self.final = nn.Linear(embed_dim, vocab_size, dtype=dtype)

    def forward(self, features):
        x = self.layer_norm(F.gelu(self.dense(features)))
        return self.final(x)


class ClsHead(nn.Module):
    '''
    Classification head that averages the embeddings of each partition in a
    sequence and passes them through a two-layer ffn.

    Args:
        embed_dim: int, the dimension of the input embeddings
        hidden_dim: int, the dimension of the hidden layer

    Returns:
        torch.Tensor, the output of the head as logits for classification with 
            shape (len(cu_lens) - 1,)

    Example:
    >>> embed = torch.tensor([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9],
    ...     [10, 11, 12],
    ...     [13, 14, 15],
    ...     ])
    >>> cu_lens = torch.tensor([0, 3, 5])
    >>> head = ClsHead(embed_dim=3, hidden_dim=4)
    >>> head(embed, cu_lens)
    ... tensor([.1, -1.2])
    '''

    def __init__(self, embed_dim, num_cls=1, hidden_dim=4096,
                 dtype=torch.bfloat16):
        super().__init__()
        self.pool = PartitionMeanPool()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cls, dtype=dtype),
        )

    def forward(self, x, cu_lens):
        return self.head(self.pool(x, cu_lens)).squeeze(-1)
