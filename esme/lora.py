from typing import Dict
import math
import torch
from torch import nn
import torch.nn.functional as F


class LoRA(nn.Module):

    def __init__(
        self,
        layer: nn.Module,
        rank: int = 16,
        alpha: int = 1,
        dropout_p: float = 0.,
        names: list = None,
        dtype=None
    ):
        """
        LoRA layer

        Args:
            layer (nn.Module): The layer to be enhanced with LoRA
            rank (int, optional): The rank of the LoRA approximation. Defaults to 0.
            alpha (int, optional): The number of LoRA layers to be used. Defaults to 1.
            dropout (float, optional): The dropout rate. Defaults to 0..
        """
        super().__init__()
        assert getattr(layer, 'in_features', None) is not None, \
            "The layer must have an attribute in_features"
        assert getattr(layer, 'out_features', None) is not None, \
            "The layer must have an attribute out_features"
        assert rank >= 0, "The rank must be a non-negative integer"

        self.layer = layer
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout_p
        if self.dropout_p > 0.:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = lambda x: x
        self.scaling = self.alpha / self.rank
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        names = names or ['default']
        self.names = set(names)

        device = next(layer.parameters()).device
        dtype = dtype or next(layer.parameters()).dtype
        if dtype == torch.uint8 or dtype == torch.int8:
            dtype = torch.bfloat16

        self.lora_A = nn.ParameterDict({
            name: nn.Parameter(
                torch.zeros((rank, self.in_features),
                            device=device, dtype=dtype))
            for name in names
        })
        self.lora_B = nn.ParameterDict({
            name: nn.Parameter(
                torch.zeros((self.out_features, rank),
                            device=device, dtype=dtype))
            for name in names
        })
        self.reset_parameters()

    def reset_parameters(self):
        for name in self.names:
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])

    def forward(self, x: torch.Tensor, names=None):
        y = self.layer(x)
        return y + self.lora_forward(x, y, names)

    def lora_forward(self, x: torch.Tensor, y: torch.Tensor, names=None):
        result = torch.zeros_like(y)
        names = names or self.names

        for name in names:
            y = F.linear(
                F.linear(x, self.lora_A[name]), self.lora_B[name]
            ) * self.scaling

            if self.dropout_p > 0:
                y = self.dropout(y)

            result += y

        return result

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}, dropout_p={self.dropout_p}'


def mark_only_lora_as_trainable(model: nn.Module, names=None) -> None:
    names = set(names or [])

    for n, p in model.named_parameters():
        p.requires_grad = False

        if ('.lora_A.' in n) or ('.lora_B.' in n):
            if len(names) == 0:
                p.requires_grad = True
            else:
                if n.split('.')[-1] in names:
                    p.requires_grad = True


def _lora_state_dict(model: nn.Module, names=None):
    names = set(names or [])

    for k, v in model.state_dict().items():
        if ('.lora_A.' in k) or ('.lora_B.' in k):
            if len(names) == 0:
                yield k, v
            else:
                if k.split('.')[-1] in names:
                    yield k, v


def lora_state_dict(model: nn.Module, names=None) -> Dict[str, torch.Tensor]:
    return dict(_lora_state_dict(model, names))
