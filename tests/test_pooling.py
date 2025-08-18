import torch
from esme.pooling import PartitionMeanPool, AttentionPool, LearnedAttentionPool, \
    LearnedAggregation, BinaryLearnedAggregation
from conftest import device


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


def test_AttentionPool():
    ndim = 64 * 4
    cu_lens = torch.tensor([0, 125, 625], dtype=torch.int32, device=device)
    max_len = 500

    embed = torch.randn(625, ndim, device=device, dtype=torch.bfloat16)

    attn_pool = AttentionPool(4, ndim, dtype=torch.bfloat16).to(device)
    cls = torch.randn(1, ndim, device=device, dtype=torch.bfloat16)
    output = attn_pool(cls, embed, (cu_lens, max_len))
    assert output.shape == (2, 1, ndim)


def test_LearnedAttentionPool():
    ndim = 64 * 8
    cu_lens = torch.tensor([0, 125, 625], dtype=torch.int32, device=device)
    max_len = 500
    embed = torch.randn(625, ndim, device=device, dtype=torch.bfloat16)

    pool = LearnedAttentionPool(4, 4, 512).to(device)
    output = pool(embed, (cu_lens, max_len))
    assert output.shape == (2, 4, 512)


def test_LearnedAggregation():
    ndim = 64 * 8
    cu_lens = torch.tensor([0, 125, 625], dtype=torch.int32, device=device)
    max_len = 500
    embed = torch.randn(625, ndim, device=device, dtype=torch.bfloat16)

    pool = LearnedAggregation(4, 4, 512).to(device)
    output = pool(embed, (cu_lens, max_len))
    assert output.shape == (2, 4, 1)


def test_BinaryLearnedAggregation():
    ndim = 64 * 8
    cu_lens = torch.tensor([0, 125, 625], dtype=torch.int32, device=device)
    max_len = 500
    embed = torch.randn(625, ndim, device=device, dtype=torch.bfloat16)

    pool = BinaryLearnedAggregation(4, 512).to(device)
    output = pool(embed, (cu_lens, max_len))
    assert output.shape == (2,)
