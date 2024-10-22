import pytest
import torch
from einops import rearrange
from torchmetrics.text import Perplexity
from safetensors.torch import safe_open
from esme.alphabet import tokenize_unpad
from esme.esm import ESM2
from conftest import device, esm2e_8M_path, p53_human


def test_esm2(token, flash_esm2, esm2_model):
    logit = esm2_model(token)['logits']
    flash_logit = flash_esm2(token, pad_output=True).float()

    perplexity = Perplexity(ignore_index=1).to(device)

    for i, seq in enumerate(token):
        _token = token[i:i+1, seq != 1]

        _flash_logit = flash_logit[i:i+1, seq != 1, :]
        _logit = logit[i:i+1, seq != 1, :]

        perp_flash = perplexity(_flash_logit, _token)
        perp = perplexity(_logit, _token)
        assert (perp - perp_flash).abs() < 1

        sim = torch.nn.functional.cosine_similarity(_logit[0], _flash_logit[0])
        assert all(sim > .99)


def test_esm2_p53(token_p53, flash_esm2, esm2_model):
    flash_logit = flash_esm2(token_p53, pad_output=True).float()
    logit = esm2_model(token_p53)['logits']

    perplexity = Perplexity(ignore_index=1).to(device)
    perp_flash = perplexity(flash_logit, token_p53)

    perplexity = Perplexity(ignore_index=1).to(device)
    perp = perplexity(logit, token_p53)
    assert perp > perp_flash

    prob_flash = torch.softmax(flash_logit, dim=-1)
    prob = torch.softmax(logit, dim=-1).float()

    indices_flash = torch.argmax(prob_flash, dim=-1)
    indices = torch.argmax(prob, dim=-1)

    assert torch.all((indices_flash == token_p53) | (indices != token_p53))

    sim = torch.nn.functional.cosine_similarity(
        rearrange(flash_logit, 'b s e -> (b s) e'),
        rearrange(logit, 'b s e -> (b s) e'),
    )
    assert all(sim > .99)

    tokens_unpad, indices, cu_lens, max_len = tokenize_unpad(
        [p53_human, p53_human * 2])
    tokens_unpad = tokens_unpad.to(device)
    cu_lens = cu_lens.to(device)

    flash_logit_unpad = flash_esm2(tokens_unpad, (cu_lens, max_len)).float()

    sim = torch.nn.functional.cosine_similarity(
        flash_logit[0],
        flash_logit_unpad[:cu_lens[1]]
    )
    assert all(sim > .99)


def test_ESM2_from_pretrained():
    model = ESM2.from_pretrained(esm2e_8M_path)
    assert (model.lm_head.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    with safe_open(esm2e_8M_path, framework="pt") as f:
        metadata = f.metadata()
        assert model.num_layers == int(metadata['num_layers'])
        assert model.embed_dim == int(metadata['embed_dim'])
        assert model.attention_heads == int(metadata['attention_heads'])

        for k in f.keys():
            assert (params[k] == f.get_tensor(k)).all()

    model = ESM2.from_pretrained(esm2e_8M_path, quantization='8bit',
                                 device=device)
    assert (model.lm_head.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    for k, v in params.items():
        for l in ['k', 'v', 'out', 'fc1', 'fc2']:
            if k.endswith(f'.{l}.weight'):
                assert v.dtype == torch.int8

    with safe_open(esm2e_8M_path, framework="pt", device=device) as f:
        for k in f.keys():
            if 'bias' in k:
                assert (params[k] == f.get_tensor(k)).all()

    model = ESM2.from_pretrained(esm2e_8M_path, quantization='4bit',
                                 device=device)
    assert (model.lm_head.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    for k, v in params.items():
        for l in ['k', 'v', 'out', 'fc1', 'fc2']:
            if k.endswith(f'.{l}.weight'):
                assert v.dtype == torch.uint8

    with safe_open(esm2e_8M_path, framework="pt") as f:
        for k in f.keys():
            if 'bias' in k:
                assert (params[k] == f.get_tensor(k).to(device)).all()
