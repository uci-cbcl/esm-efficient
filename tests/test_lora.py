import safetensors
import torch
from torch import nn
from esme.lora import LoRA, mark_only_lora_as_trainable, lora_state_dict
from esme.esm import ESM2
from esme.quantization import Linear8bit
import pytest
from conftest import esm2e_8M_path, device


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('rank', [16, 32])
@pytest.mark.parametrize('dropout_p', [0, 0.5])
@pytest.mark.parametrize('names', [None, ['test']])
@pytest.mark.parametrize('layer_type', ['linear'])
def test_LoRA(dtype, rank, dropout_p, names, layer_type):
    layer = torch.nn.Linear(256, 512, dtype=dtype, device=device)

    if layer_type == '8bit':
        layer = Linear8bit(layer.weight, layer.bias, device=device)

    lora = LoRA(layer, rank=rank, alpha=1, dropout_p=dropout_p, names=names)
    assert lora.layer == layer
    if names is None:
        assert 'default' in lora.lora_A
        assert 'default' in lora.lora_B
    else:
        assert 'test' in lora.lora_A
        assert 'test' in lora.lora_B

    x = torch.randn((4 * 32, 256), dtype=dtype, device=device)
    y = lora(x)

    assert y.shape == (4 * 32, 512)
    assert y.dtype == dtype
    torch.testing.assert_close(y, layer(x))


def test_mark_only_lora_as_trainable():

    model = nn.Sequential(*[
        LoRA(torch.nn.Linear(512, 512), rank=16,
             alpha=1, dropout_p=0.5, names=['test'])
        for _ in range(5)
    ])

    mark_only_lora_as_trainable(model)

    for i in range(5):
        assert model[i].lora_A['test'].requires_grad
        assert model[i].lora_B['test'].requires_grad
        assert not model[i].layer.weight.requires_grad
        assert not model[i].layer.bias.requires_grad

    model = nn.Sequential(*[
        LoRA(torch.nn.Linear(512, 512), rank=16,
             alpha=1, dropout_p=0.5, names=['x', 'y'])
        for _ in range(5)
    ])

    mark_only_lora_as_trainable(model, names=['x'])

    for i in range(5):
        assert model[i].lora_A['x'].requires_grad
        assert model[i].lora_B['x'].requires_grad
        assert not model[i].lora_A['y'].requires_grad
        assert not model[i].lora_B['y'].requires_grad
        assert not model[i].layer.weight.requires_grad
        assert not model[i].layer.bias.requires_grad


def test_lora_state_dict():

    model = nn.Sequential(*[
        LoRA(torch.nn.Linear(512, 512), rank=16,
             alpha=1, dropout_p=0.5, names=['test'])
        for _ in range(5)
    ])

    state = lora_state_dict(model)
    assert set(state.keys()) == set([
        f'{i}.lora_A.test' for i in range(5)
    ] + [
        f'{i}.lora_B.test' for i in range(5)
    ])

    model = nn.Sequential(*[
        LoRA(torch.nn.Linear(512, 512), rank=16,
             alpha=1, dropout_p=0.5, names=['x', 'y'])
        for _ in range(5)
    ])

    state = lora_state_dict(model)
    assert set(state.keys()) == set([
        f'{i}.lora_A.x' for i in range(5)
    ] + [
        f'{i}.lora_B.x' for i in range(5)
    ] + [
        f'{i}.lora_A.y' for i in range(5)
    ] + [
        f'{i}.lora_B.y' for i in range(5)
    ])

    model = nn.Sequential(*[
        LoRA(torch.nn.Linear(512, 512), rank=16,
             alpha=1, dropout_p=0.5, names=['x'])
        for _ in range(5)
    ])

    state = lora_state_dict(model)
    assert set(state.keys()) == set([
        f'{i}.lora_A.x' for i in range(5)
    ] + [
        f'{i}.lora_B.x' for i in range(5)
    ])


def test_lora_add_state_dict():

    model = ESM2.from_pretrained(esm2e_8M_path)

    model.add_lora(16, 0.5, adapter_names=['test'],
                   layers=('query', 'value', 'output'))

    state_dict = model.lora_state_dict(['test'])

    assert set(state_dict.keys()) == set([
        f'layers.{i}.self_attn.{j}.lora_{a}.test'
        for i, _ in enumerate(model.layers)
        for j in ['q', 'v', 'out']
        for a in ['A', 'B']
    ])


def test_lora_save_load(tmp_path):

    model = ESM2.from_pretrained(esm2e_8M_path)

    model.add_lora(16, 0.5, adapter_names=['test_a', 'test_b'],
                   layers=('query', 'value', 'output'))

    path = tmp_path / 'model.safetensors'
    model.save_lora(path)

    with safetensors.safe_open(path, 'pt') as sf:
        assert set(sf.keys()) == set([
            f'layers.{i}.self_attn.{j}.lora_{a}.test_{n}'
            for i, _ in enumerate(model.layers)
            for j in ['q', 'v', 'out']
            for a in ['A', 'B']
            for n in ['a', 'b']
        ])

        metadata = sf.metadata()
        metadata_expected = {
            'rank': '16',
            'dropout_p': '0.0',
            'alpha': '0.5',
            'names': 'test_a,test_b',
            'layers': 'output,query,value',
            'format': 'pt'
        }

        for k in metadata:
            if k in {'layers', 'test'}:
                assert set(metadata[k].split(',')) \
                    == set(metadata_expected[k].split(','))
            else:
                assert metadata[k] == metadata_expected[k]

    model = ESM2.from_pretrained(esm2e_8M_path) \
        .load_lora(path)

    with safetensors.safe_open(path, 'pt') as sf:
        for i, _ in enumerate(model.layers):
            layer = model.layers[i]
            for j in ['q', 'v', 'out']:
                assert torch.allclose(
                    getattr(layer.self_attn, j).lora_A.test_a,
                    sf.get_tensor(f'layers.{i}.self_attn.{j}.lora_A.test_a')
                )
                assert torch.allclose(
                    getattr(layer.self_attn, j).lora_A.test_b,
                    sf.get_tensor(f'layers.{i}.self_attn.{j}.lora_A.test_b')
                )
                assert torch.allclose(
                    getattr(layer.self_attn, j).lora_B.test_a,
                    sf.get_tensor(f'layers.{i}.self_attn.{j}.lora_B.test_a')
                )
                assert torch.allclose(
                    getattr(layer.self_attn, j).lora_B.test_b,
                    sf.get_tensor(f'layers.{i}.self_attn.{j}.lora_B.test_b')
                )
