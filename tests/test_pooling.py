import torch
from esme.pooling import PartitionMeanPool


def test_PartitionMeanPool_indices():
    cu_lens = torch.tensor([0, 3, 5, 7])
    embed = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    ])
    indices = PartitionMeanPool._indices(cu_lens)
    assert torch.all(indices == torch.tensor([0, 0, 0, 1, 1, 2, 2]))

    
def test_PartitionMeanPool():
    cu_lens = torch.tensor([0, 3, 5, 7])
    embed = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    ], dtype=torch.float32)
    pool = PartitionMeanPool()
    assert torch.all(pool(embed, cu_lens) == torch.tensor([
        [4., 5., 6.],
        [11.5, 12.5, 13.5],
        [17.5, 18.5, 19.5]       
    ], dtype=torch.float32))