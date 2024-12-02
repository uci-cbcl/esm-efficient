import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
from einops import rearrange
import esm
from flash_attn.bert_padding import pad_input, unpad_input
from esm.modules import TransformerLayer
from esm.multihead_attention import MultiheadAttention
from esme.attention import FlashTransformerLayer, FlashMultiheadAttention, RotaryEmbedding as FlashRotaryEmbedding
from esm.rotary_embedding import RotaryEmbedding as EsmRotaryEmbedding
from conftest import device, embed_dim, n_heads, bz, seq_len


@pytest.fixture
def embedding_layer(esm2_model):
    return esm2_model.embed_tokens


@pytest.fixture
def embedding(token, embedding_layer):
    return embedding_layer(token).to(device)


@pytest.fixture
def embedding_token(token, embedding_layer):
    return embedding_layer(token).to(device), token.to(device)


@pytest.fixture
def transformer_layer(esm2_model):
    return esm2_model.layers[0]


@pytest.fixture
def multihead_attention(transformer_layer):
    return transformer_layer.self_attn


@pytest.fixture
def flash_multihead_attention(flash_transformer_layer):
    return flash_transformer_layer.self_attn


@pytest.fixture
def flash_transformer_layer(transformer_layer):
    flash_layer = FlashTransformerLayer(
        embed_dim,
        embed_dim * 4,
        n_heads,
        dtype=torch.bfloat16
    ).to(device)
    state_dict = {
        k.replace('_proj', ''): v
        for k, v in transformer_layer.state_dict().items()
    }
    del state_dict['self_attn.rot_emb.inv_freq']
    flash_layer.load_state_dict(state_dict)
    return flash_layer


def test_q_k_v(embedding, transformer_layer, flash_transformer_layer):
    # query
    output = transformer_layer.self_attn.q_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    output_flash = flash_transformer_layer.self_attn.q(
        embedding.to(torch.bfloat16))
    assert torch.allclose(output.to(torch.bfloat16), output_flash, atol=1e-1)
    # key
    output = transformer_layer.self_attn.k_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    output_flash = flash_transformer_layer.self_attn.k(
        embedding.to(torch.bfloat16))
    assert torch.allclose(output.to(torch.bfloat16), output_flash, atol=1e-1)
    # value
    output = transformer_layer.self_attn.v_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    output_flash = flash_transformer_layer.self_attn.v(
        embedding.to(torch.bfloat16))
    assert torch.allclose(output.to(torch.bfloat16), output_flash, atol=1e-1)
    # output
    output = transformer_layer.self_attn.out_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    output_flash = flash_transformer_layer.self_attn.out(
        embedding.to(torch.bfloat16))
    assert torch.allclose(output.to(torch.bfloat16), output_flash, atol=1e-1)


def test_qkv(embedding_token, multihead_attention, flash_multihead_attention):
    embedding, token = embedding_token

    x, indices, cu_lens, max_len, _ = unpad_input(
        hidden_states=embedding, attention_mask=~token.eq(1))

    qkv = flash_multihead_attention._qkv(x.to(torch.bfloat16))
    q_flash, k_flash = qkv[:, 0], qkv[:, 1]

    _rot_emb = EsmRotaryEmbedding(16).to(device).to(torch.bfloat16)
    multihead_attention.__dict__['rot_emb'] = Mock(side_effect=_rot_emb)

    x = embedding.transpose(0, 1)
    multihead_attention(
        x, x, x,
    )[0].transpose(0, 1)

    q, k = multihead_attention.rot_emb.call_args.args
    q = rearrange(q, '(b h) s d -> (b s) h d', b=len(cu_lens) - 1)[indices]
    k = rearrange(k, '(b h) s d -> (b s) h d', b=len(cu_lens) - 1)[indices]

    assert torch.allclose(q.to(torch.bfloat16), q_flash, atol=.1)
    assert torch.allclose(k.to(torch.bfloat16), k_flash, atol=.1)


def test_multihead_attention(embedding_token, multihead_attention,
                             flash_multihead_attention):
    embedding, token = embedding_token

    multihead_attention.__dict__['out_proj'] = Mock(
        side_effect=multihead_attention.out_proj)

    flash_multihead_attention.__dict__['out'] = Mock(
        side_effect=flash_multihead_attention.out)

    x = embedding.transpose(0, 1)
    output = multihead_attention(
        x, x, x, key_padding_mask=token.eq(1),
    )[0].transpose(0, 1).to(torch.bfloat16)

    x, indices, cu_lens, max_len, _ = unpad_input(
        hidden_states=embedding, attention_mask=~token.eq(1))

    output_flash = flash_multihead_attention(
        x.to(torch.bfloat16), cu_lens, max_len)

    out_proj = multihead_attention.out_proj \
        .call_args.args[0].transpose(0, 1).to(torch.bfloat16)
    out_proj_flash = flash_multihead_attention.out \
        .call_args.args[0]

    sim = torch.nn.functional.cosine_similarity(
        out_proj_flash,
        rearrange(out_proj, 'b s e -> (b s) e')[indices]
    )
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-2)

    output = rearrange(output, 'b s e -> (b s) e')[indices]
    assert torch.allclose(output_flash, output, atol=1e-1)

    sim = torch.nn.functional.cosine_similarity(output_flash, output)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-2)


def test_multihead_attention_varlen_len(embedding_token, multihead_attention,
                                        flash_multihead_attention):
    embedding, token = embedding_token

    x = embedding.transpose(0, 1)
    output = multihead_attention(
        x, x, x, key_padding_mask=token.eq(1),
    )[0].transpose(0, 1)

    from flash_attn.bert_padding import pad_input, unpad_input

    x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(
        hidden_states=embedding, attention_mask=~token.eq(1))

    output_flash = flash_multihead_attention(
        x_unpad.to(torch.bfloat16), cu_seqlens, max_seqlen)
    output_flash = pad_input(output_flash, indices,
                             batch=token.shape[0], seqlen=token.shape[1])

    for i, seq_len in enumerate((~token.eq(1)).sum(dim=1)):
        out = output[i, :seq_len, :].to(torch.bfloat16)
        out_flash = output_flash[i, :seq_len, :]
        sim = torch.nn.functional.cosine_similarity(out, out_flash)
        assert torch.allclose(out_flash, out, atol=1e-1)
        assert torch.allclose(sim, torch.ones_like(sim), atol=1e-2)
