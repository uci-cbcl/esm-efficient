import torch
import torch.nn as nn


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
        index = torch.zeros(cu_lens[-1], dtype=torch.long, device=cu_lens.device)
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
