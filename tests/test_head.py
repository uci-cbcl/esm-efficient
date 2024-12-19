import torch
import torch.nn.functional as F
from esme.head import ClsHead, RobertaLMHead
from conftest import device


def test_ClsHead():
    head = ClsHead(512, 1024)
    embed = torch.normal(0, 1, size=(8, 512), dtype=torch.bfloat16)
    cu_lens = torch.tensor([0, 3, 8])

    out = head(embed, cu_lens)
    assert list(out.shape) == [2, 1024]


def test_RobertaLMHead(flash_esm2, esm2_model):

    lm_head = flash_esm2.lm_head
    esm_head = esm2_model.lm_head

    embed = flash_esm2.forward_representation(
        torch.tensor([[
            0, 20, 32, 9, 14, 16, 8, 13, 14, 8, 7, 9, 14, 14, 4, 8, 16, 2
        ]], device=device)
    )

    out_flash = torch.softmax(lm_head(embed), axis=-1)
    out_esm = torch.softmax(esm_head(embed), axis=-1)

    assert torch.allclose(lm_head.final.weight, esm_head.weight)
    assert torch.allclose(lm_head.final.bias, esm_head.bias)

    assert torch.allclose(out_flash, out_esm, atol=1e-2)
