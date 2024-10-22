import torch
from esme.rotary import RotaryEmbedding
from esm.rotary_embedding import RotaryEmbedding as EsmRotaryEmbedding


def test_RotaryEmbedding():

    rotary_embedding = RotaryEmbedding(dim=64, device=0)

    qkv = torch.rand(60 + 40 + 180, 3, 8, 64, device=0)
    _qkv = qkv.clone()
    cu_seqlens = torch.tensor([0, 60, 100, 280], device=0)
    max_seqlen = 280

    rotary_embedding(qkv, cu_seqlens, max_seqlen)  # inplace operation

    assert qkv.shape == _qkv.shape

    rotary_embedding_esm = EsmRotaryEmbedding(dim=64).to(0)
    q, k = qkv[:, 0], qkv[:, 1]
    q, k, rotary_embedding_esm(q, k)

    assert torch.allclose(q, qkv[:, 0], atol=1e-4)
    assert torch.allclose(k, qkv[:, 1], atol=1e-4)
