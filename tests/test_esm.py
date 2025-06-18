import pytest
import torch
from einops import rearrange
from torchmetrics.text import Perplexity
from safetensors.torch import safe_open
from esme.alphabet import tokenize_unpad, Alphabet3
from esme.esm import ESM2, ESM, ESM1v, ESM1b
from conftest import device, esm2e_8M_path, p53_human


def test_esmc_p53(token_p53, flash_esmc, esmc_model):
    flash_logit = flash_esmc(token_p53, pad_output=True).float()
    logit = esmc_model(token_p53, token_p53 != 1).sequence_logits

    perplexity = Perplexity(ignore_index=1).to(device)
    perp_flash = perplexity(flash_logit, token_p53)

    perplexity = Perplexity(ignore_index=1).to(device)
    perp = perplexity(logit, token_p53)
    assert (perp - perp_flash).abs() < .1

    prob_flash = torch.softmax(flash_logit, dim=-1)
    prob = torch.softmax(logit, dim=-1).float()

    sim = torch.nn.functional.cosine_similarity(
        rearrange(flash_logit, 'b s e -> (b s) e'),
        rearrange(logit, 'b s e -> (b s) e'),
    )
    assert all(sim > .99)

    tokens_unpad, _, cu_lens, max_len = tokenize_unpad(
        [p53_human, p53_human * 2])
    tokens_unpad = tokens_unpad.to(device)
    cu_lens = cu_lens.to(device)

    flash_logit_unpad = flash_esmc(tokens_unpad, (cu_lens, max_len)).float()

    sim = torch.nn.functional.cosine_similarity(
        flash_logit[0],
        flash_logit_unpad[:cu_lens[1]]
    )
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


def test_esm1b_p53(token_p53):
    esm1b = ESM1b().to(device)
    logit = esm1b(token_p53.to(device))

    tokens_unpad, indices, cu_lens, max_len = tokenize_unpad([p53_human])
    tokens_unpad = tokens_unpad.to(device)
    cu_lens = cu_lens.to(device)

    logit_packed = esm1b(tokens_unpad.to(device), (cu_lens, max_len))
    assert torch.all(logit == logit_packed)


def test_esm1v_p53(token_p53):
    esm1v = ESM1v().to(device)
    logit = esm1v(token_p53.to(device))

    tokens_unpad, indices, cu_lens, max_len = tokenize_unpad([p53_human])
    tokens_unpad = tokens_unpad.to(device)
    cu_lens = cu_lens.to(device)

    logit_packed = esm1v(tokens_unpad.to(device), (cu_lens, max_len))
    assert torch.all(logit == logit_packed)

def test_ESM2_from_pretrained():
    model = ESM2.from_pretrained(esm2e_8M_path)
    assert (model.lm_head.final.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    with safe_open(esm2e_8M_path, framework="pt") as f:
        metadata = f.metadata()
        assert model.num_layers == int(metadata['num_layers'])
        assert model.embed_dim == int(metadata['embed_dim'])
        assert model.attention_heads == int(metadata['attention_heads'])

        for k in f.keys():
            assert (params[k] == f.get_tensor(k)).all()


def test_ESM2_from_pretrained_int8():
    model = ESM2.from_pretrained(esm2e_8M_path, quantization='8bit',
                                 device=device)
    assert (model.lm_head.final.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    for k, v in params.items():
        for l in ['q', 'k', 'v', 'out', 'fc1', 'fc2']:
            if k.endswith(f'.{l}.weight'):
                assert v.dtype == torch.int8

    with safe_open(esm2e_8M_path, framework="pt", device=device) as f:
        for k in f.keys():
            if 'bias' in k:
                assert (params[k] == f.get_tensor(k)).all()


def test_ESM2_from_pretrained_int4():
    model = ESM2.from_pretrained(esm2e_8M_path, quantization='4bit',
                                 device=device)
    assert (model.lm_head.final.weight == model.embed_tokens.weight).all()
    params = model.state_dict()

    for k, v in params.items():
        for l in ['k', 'v', 'out', 'fc1', 'fc2']:
            if k.endswith(f'.{l}.weight'):
                assert v.dtype == torch.uint8

    with safe_open(esm2e_8M_path, framework="pt") as f:
        for k in f.keys():
            if 'bias' in k:
                assert (params[k] == f.get_tensor(k).to(device)).all()


def test_ESM_from_pretrained():
    model = ESM.from_pretrained('esm2_8m')
    assert (model.lm_head.final.weight == model.embed_tokens.weight).all()
    model = ESM.from_pretrained('esm2_8m', quantization='8bit', device=device)
    model = ESM.from_pretrained('esm2_8m', checkpointing=True)
