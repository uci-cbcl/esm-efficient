import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from fair_esm.modules import TransformerLayer
from fair_esm.multihead_attention import MultiheadAttention
from fair_esm.rotary_embedding import RotaryEmbedding as EsmRotaryEmbedding
from esme.attention import FlashTransformerLayer, FlashMultiheadAttention, RotaryEmbedding as FlashRotaryEmbedding
from conftest import device, embed_dim, n_heads, bz, seq_len


@pytest.fixture
def embedding_layer(esm2_model):
    return esm2_model.embed_tokens


@pytest.fixture
def embedding_layer_esmc(esmc_model):
    return esmc_model.embed


@pytest.fixture
def embedding(token, embedding_layer):
    return embedding_layer(token).to(device)


@pytest.fixture
def embedding_esmc(token, embedding_layer_esmc):
    return embedding_layer_esmc(token).to(device)


@pytest.fixture
def embedding_token(token, embedding_layer):
    return embedding_layer(token).to(device), token.to(device)


@pytest.fixture
def embedding_token_esmc(token, embedding_layer_esmc):
    return embedding_layer_esmc(token).to(device), token.to(device)


@pytest.fixture
def transformer_layer(esm2_model):
    return esm2_model.layers[0]


@pytest.fixture
def transformer_layer_esmc(esmc_model):
    return esmc_model.transformer.blocks[0]


@pytest.fixture
def multihead_attention(transformer_layer):
    return transformer_layer.self_attn


@pytest.fixture
def multihead_attention_esmc(transformer_layer_esmc):
    return transformer_layer_esmc.attn


@pytest.fixture
def flash_multihead_attention(flash_transformer_layer):
    return flash_transformer_layer.self_attn


@pytest.fixture
def flash_multihead_attention_esmc(flash_transformer_layer_esmc):
    return flash_transformer_layer_esmc.self_attn


@pytest.fixture
def flash_transformer_layer(transformer_layer):
    flash_layer = FlashTransformerLayer(
        embed_dim, 4,
        n_heads,
        bias=True,
        pre_layernorm=False,
        final_activation='gelu',
        dtype=torch.bfloat16,
    ).to(device)

    state_dict = transformer_layer.state_dict()
    state_dict_new = {
        'self_attn.norm.weight': state_dict['self_attn_layer_norm.weight'],
        'self_attn.norm.bias': state_dict['self_attn_layer_norm.bias'],
        'self_attn.q.weight': state_dict['self_attn.q_proj.weight'],
        'self_attn.q.bias': state_dict['self_attn.q_proj.bias'],
        'self_attn.k.weight': state_dict['self_attn.k_proj.weight'],
        'self_attn.k.bias': state_dict['self_attn.k_proj.bias'],
        'self_attn.v.weight': state_dict['self_attn.v_proj.weight'],
        'self_attn.v.bias': state_dict['self_attn.v_proj.bias'],
        'self_attn.out.weight': state_dict['self_attn.out_proj.weight'],
        'self_attn.out.bias': state_dict['self_attn.out_proj.bias'],
        'final.0.weight': state_dict['final_layer_norm.weight'],
        'final.0.bias': state_dict['final_layer_norm.bias'],
        'final.1.weight': state_dict['fc1.weight'],
        'final.1.bias': state_dict['fc1.bias'],
        'final.3.weight': state_dict['fc2.weight'],
        'final.3.bias': state_dict['fc2.bias'],
    }
    flash_layer.load_state_dict(state_dict_new)
    return flash_layer


@pytest.fixture
def flash_transformer_layer_esmc(transformer_layer_esmc):
    flash_layer = FlashTransformerLayer(
        960,  8 / 3,
        attention_heads=15,
        bias=False,
        residue_scaling=(30 / 36)**0.5,
        pre_layernorm=True,
        final_activation='swiglu',
        dtype=torch.bfloat16,
    ).to(device)

    state_dict = transformer_layer_esmc.state_dict()
    qkv = state_dict['attn.layernorm_qkv.1.weight']

    state_dict_new = {
        'self_attn.norm.weight': state_dict['attn.layernorm_qkv.0.weight'],
        'self_attn.norm.bias': state_dict['attn.layernorm_qkv.0.bias'],
        'self_attn.q.weight': qkv[:960],
        'self_attn.k.weight': qkv[960:1920],
        'self_attn.v.weight': qkv[1920:],
        'self_attn.out.weight': state_dict['attn.out_proj.weight'],
        'self_attn.layernorm_q.weight': state_dict['attn.q_ln.weight'],
        'self_attn.layernorm_k.weight': state_dict['attn.k_ln.weight'],
        'final.0.weight': state_dict['ffn.0.weight'],
        'final.0.bias': state_dict['ffn.0.bias'],
        'final.1.activation.weight': state_dict['ffn.1.weight'][:2560, :],
        'final.1.fc.weight': state_dict['ffn.1.weight'][2560:, :],
        'final.2.weight': state_dict['ffn.3.weight'],
    }
    flash_layer.load_state_dict(state_dict_new)
    return flash_layer


def test_q_k_v(embedding, transformer_layer, flash_transformer_layer):

    output_flash_q = flash_transformer_layer.self_attn.q(embedding)
    output_flash_k = flash_transformer_layer.self_attn.k(embedding)
    output_flash_v = flash_transformer_layer.self_attn.v(embedding)

    output = transformer_layer.self_attn.q_proj(embedding)
    assert torch.allclose(output, output_flash_q)

    output = transformer_layer.self_attn.q_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    assert torch.allclose(output, output_flash_q, atol=5e-2)

    # key
    output = transformer_layer.self_attn.k_proj(embedding)
    assert torch.allclose(output, output_flash_k)

    output = transformer_layer.self_attn.k_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    assert torch.allclose(output, output_flash_k, atol=5e-2)

    # value
    output = transformer_layer.self_attn.v_proj(embedding)
    assert torch.allclose(output, output_flash_v)

    output = transformer_layer.self_attn.v_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    assert torch.allclose(output, output_flash_v, atol=5e-2)

    # output
    output = transformer_layer.self_attn.out_proj(embedding)
    output_flash = flash_transformer_layer.self_attn.out(embedding)
    assert torch.allclose(output, output_flash)

    output = transformer_layer.self_attn.out_proj(
        embedding.transpose(0, 1)).transpose(0, 1)
    output_flash = flash_transformer_layer.self_attn.out(embedding)
    assert torch.allclose(output, output_flash, atol=5e-2)


def test_qkv(embedding_token, multihead_attention, flash_multihead_attention):
    embedding, token = embedding_token

    x, indices, cu_lens, _, _ = unpad_input(hidden_states=embedding,
                                            attention_mask=~token.eq(1))

    flash_multihead_attention.norm = nn.Identity()  # norm move to MHA
    q_flash, k_flash, _ = flash_multihead_attention._qkv(x)

    _rot_emb = EsmRotaryEmbedding(16).to(device).to(torch.bfloat16)
    multihead_attention.__dict__['rot_emb'] = Mock(side_effect=_rot_emb)

    x = embedding.transpose(0, 1).contiguous()
    multihead_attention(x, x, x)

    q, k = multihead_attention.rot_emb.call_args.args
    q = rearrange(q, '(b h) s d -> (b s) h d', b=len(cu_lens) - 1)[indices]
    k = rearrange(k, '(b h) s d -> (b s) h d', b=len(cu_lens) - 1)[indices]

    scaling = flash_multihead_attention.head_dim**-0.5
    assert torch.allclose(q, q_flash * scaling)
    assert torch.allclose(k, k_flash)


def test_multihead_attention(embedding_token, multihead_attention,
                             flash_multihead_attention):
    embedding, token = embedding_token

    x = embedding.transpose(0, 1)
    output = multihead_attention(
        x, x, x, key_padding_mask=token.eq(1),
    )[0].transpose(0, 1)

    x, indices, cu_lens, max_len, _ = unpad_input(
        hidden_states=embedding, attention_mask=~token.eq(1))

    flash_multihead_attention.norm = nn.Identity()  # norm move to MHA
    output_flash = flash_multihead_attention(x, cu_lens, max_len)

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

    flash_multihead_attention.norm = nn.Identity()  # norm move inside MHA
    output_flash = flash_multihead_attention(x_unpad, cu_seqlens, max_seqlen)
    output_flash = pad_input(output_flash, indices, batch=token.shape[0],
                             seqlen=token.shape[1])

    for i, seq_len in enumerate((~token.eq(1)).sum(dim=1)):
        out = output[i, :seq_len, :]
        out_flash = output_flash[i, :seq_len, :]
        assert torch.allclose(out_flash, out, atol=.2)


def test_multihead_attention_esmc(embedding_token_esmc, multihead_attention_esmc,
                                  flash_multihead_attention_esmc):
    embedding, token = embedding_token_esmc
    output = multihead_attention_esmc(embedding, token == 1)

    x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(
        hidden_states=embedding, attention_mask=~token.eq(1))
    output_flash = flash_multihead_attention_esmc(
        x_unpad, cu_seqlens, max_seqlen)
    output_flash = pad_input(output_flash, indices, batch=token.shape[0],
                             seqlen=token.shape[1])

    for i, seq_len in enumerate((~token.eq(1)).sum(dim=1)):
        out = output[i, :seq_len, :]
        out_flash = output_flash[i, :seq_len, :]
        assert torch.allclose(out_flash, out, atol=.3)
