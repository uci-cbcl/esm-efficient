import pytest
import torch
from flash_attn.bert_padding import unpad_input
from fair_esm.rotary_embedding import RotaryEmbedding as EsmRotaryEmbedding, \
    apply_rotary_pos_emb as apply_rotary_pos_emb_esm
from esm.layers.rotary import RotaryEmbedding as RotaryEmbeddingESMC
from esme.rotary import RotaryEmbedding, apply_rotary
from conftest import device


@pytest.fixture
def rotary_embedding_esm():
    return EsmRotaryEmbedding(dim=64).to(device)


@pytest.fixture
def rotary_embedding_esmc():
    return RotaryEmbeddingESMC(dim=64, device=device)


@pytest.fixture
def rotary_embedding():
    return RotaryEmbedding(dim=64, device=device)


@pytest.fixture
def q_padded():
    q_padded = torch.rand(3, 180, 8, 64, device=device)
    q_padded[0, 60:] = 0
    q_padded[1, 40:] = 0
    q_padded[2, 180:] = 0
    return q_padded


@pytest.fixture
def q_padded_t(q_padded):
    return q_padded.transpose(0, 1).reshape(180, 3 * 8, 64).transpose(0, 1)


@pytest.fixture
def q_unpadded(q_padded):
    attention_mask = torch.ones(3, 180, device=0)
    attention_mask[0, 60:] = 0
    attention_mask[1, 40:] = 0
    attention_mask[2, 180:] = 0
    q_unpadded, _, q_cu_lens, _, _ = unpad_input(
        q_padded, attention_mask=attention_mask)
    return q_unpadded, q_cu_lens


@pytest.fixture
def k_padded():
    k_padded = torch.rand(3, 180, 8, 64, device=0)
    k_padded[0, 60:] = 0
    k_padded[1, 40:] = 0
    k_padded[2, 180:] = 0
    return k_padded


@pytest.fixture
def k_padded_t(k_padded):
    return k_padded.transpose(0, 1).reshape(180, 3 * 8, 64).transpose(0, 1)


@pytest.fixture
def k_unpadded(k_padded):
    attention_mask = torch.ones(3, 180, device=0)
    attention_mask[0, 60:] = 0
    attention_mask[1, 40:] = 0
    attention_mask[2, 180:] = 0
    k_unpadded, _, k_cu_lens, _, _ = unpad_input(
        k_padded, attention_mask=attention_mask)
    return k_unpadded, k_cu_lens


def test_apply_rotary(rotary_embedding_esm, q_padded_t, q_unpadded):

    cos_esm, sin_esm = rotary_embedding_esm._update_cos_sin_tables(q_padded_t)

    q_unpadded, q_cu_lens = q_unpadded
    q_rotated = apply_rotary(q_unpadded, cos_esm[0], sin_esm[0], q_cu_lens)

    q_padded_rotated = apply_rotary_pos_emb_esm(q_padded_t, cos_esm, sin_esm)
    q_padded_rotated = q_padded_rotated.reshape(3, 8, 180, 64).transpose(1, 2)

    assert torch.allclose(q_padded_rotated[0, :60], q_rotated[:60])
    assert torch.allclose(q_padded_rotated[1, :40], q_rotated[60:100])
    assert torch.allclose(q_padded_rotated[2, :180], q_rotated[100:])


def test_RotaryEmbedding_update_cos_sin_cache(rotary_embedding, rotary_embedding_esm, q_padded_t):

    cos_esm, sin_esm = rotary_embedding_esm._update_cos_sin_tables(q_padded_t)

    rotary_embedding._update_cos_sin_cache(180, device=0, dtype=torch.float32)
    cos_flash = rotary_embedding._cos_cached
    sin_flash = rotary_embedding._sin_cached

    assert torch.allclose(cos_esm, cos_flash, atol=1e-6)
    assert torch.allclose(sin_esm, sin_flash, atol=1e-6)


def test_RotaryEmbedding(rotary_embedding, rotary_embedding_esm, rotary_embedding_esmc,
                         q_padded, q_padded_t, q_unpadded, k_padded, k_padded_t, k_unpadded):
    q_unpadded, _ = q_unpadded
    k_unpadded, _ = k_unpadded

    cu_lens = torch.tensor([0, 60, 100, 280], device=0)
    max_len = 180

    q_r, k_r = rotary_embedding(q_unpadded, k_unpadded, cu_lens, max_len)
    q_esm, k_esm = rotary_embedding_esm(q_padded_t, k_padded_t)
    q_esmc, k_esmc = rotary_embedding_esmc(q_padded, k_padded)

    q_esm = q_esm.reshape(3, 8, 180, 64).transpose(1, 2)
    k_esm = k_esm.reshape(3, 8, 180, 64).transpose(1, 2)

    # ESM1,2
    assert torch.allclose(q_esm[0, :60], q_r[:60], atol=2e-6)
    assert torch.allclose(k_esm[0, :60], k_r[:60], atol=2e-6)

    assert torch.allclose(q_esm[1, :40], q_r[60:100], atol=2e-6)
    assert torch.allclose(k_esm[1, :40], k_r[60:100], atol=2e-6)

    assert torch.allclose(q_esm[2, :180], q_r[100:], atol=2e-6)
    assert torch.allclose(k_esm[2, :180], k_r[100:], atol=2e-6)

    # ESMC
    assert torch.allclose(q_esmc[0, :60], q_r[:60], atol=1e-6)
    assert torch.allclose(k_esmc[0, :60], k_r[:60], atol=1e-6)

    assert torch.allclose(q_esmc[1, :40], q_r[60:100], atol=1e-6)
    assert torch.allclose(k_esmc[1, :40], k_r[60:100], atol=1e-6)

    assert torch.allclose(q_esmc[2, :180], q_r[100:], atol=1e-6)
    assert torch.allclose(k_esmc[2, :180], k_r[100:], atol=1e-6)
