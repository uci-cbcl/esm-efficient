import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from esme.lora import LoRA
from esme.rotary import RotaryEmbedding


class FlashMultiheadAttention(nn.Module):
    '''
    FlashMultiheadAttention is a PyTorch module implementing a Flash Multihead 
        Attention Layer with Variable Length Support (Flash-Attn).

    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability. Default is 0.0.
        dtype (torch.dtype): Data type for the linear layers. Default is torch.bfloat16.
        rotary_embedding (bool): Whether to use rotary embeddings. Default is True.

    Attributes:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability. Default is 0.0.
        dtype (torch.dtype): Data type for the linear layers. Default is torch.bfloat16.
        rotary_embedding (bool): Whether to use rotary embeddings. Default is True.

    Methods:
        reset_parameters():
            Initializes the parameters of the linear layers using Xavier uniform initialization.
        _qkv(x: Tensor) -> Tensor:
            Computes the query, key, and value matrices from the input tensor.
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
            Returns:
                Tensor: A tensor containing the stacked query, key, and value matrices.
        _attn(qkv: Tensor, cu_lens: Tensor, max_len: int) -> Tensor:
            Computes the attention output using the Flash-Attn mechanism.
            Args:
                qkv (Tensor): Tensor containing the stacked query, key, and value matrices.
                cu_lens (Tensor): Cumulative sequence lengths.
                max_len (int): Maximum sequence length.
            Returns:
                Tensor: The attention output tensor.
        forward(x: Tensor, cu_lens: Tensor, max_len: int) -> Tensor:
            Forward pass of the FlashMultiheadAttention layer.
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
                cu_lens (Tensor): Cumulative sequence lengths.
                max_len (int): Maximum sequence length.
            Returns:
                Tensor: The output tensor after applying multihead attention and the output linear layer.
    '''

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.0,
        pre_layernorm=True,
        rotary_embedding=True,
        bias=False,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(embed_dim, dtype=dtype)
        self.q = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.k = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.v = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.out = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)

        if rotary_embedding:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)
        else:
            self.rot_emb = None

        self.pre_layernorm = pre_layernorm
        if self.pre_layernorm:
            self.layernorm_q = nn.LayerNorm(embed_dim, bias=bias, dtype=dtype)
            self.layernorm_k = nn.LayerNorm(embed_dim, bias=bias, dtype=dtype)

    def _qkv(self, x, lora_names=None):
        x = self.norm(x)

        if lora_names is not None:
            q = self.q(x, lora_names) if isinstance(self.q, LoRA) else self.q(x)
            k = self.k(x, lora_names) if isinstance(self.k, LoRA) else self.k(x)
            v = self.v(x, lora_names) if isinstance(self.v, LoRA) else self.v(x)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)

        if self.pre_layernorm:
            q, k = self.layernorm_q(q), self.layernorm_k(k)

        return map(
            lambda t: rearrange(t, 'b (h d) -> b h d', h=self.num_heads),
            (q, k, v)
        )

    def _attn(self, q, k, v, cu_lens, max_len):
        '''
        '''
        x = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_lens,
            cu_seqlens_k=cu_lens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            dropout_p=self.dropout,
            causal=False,
        )
        return rearrange(x, 't h d -> t (h d)')

    def forward(self, x: Tensor, cu_lens, max_len, lora_names=None) -> Tensor:
        '''
        '''
        q, k, v = self._qkv(x, lora_names)

        if self.rot_emb:
            q, k = self.rot_emb(q, k, cu_lens, max_len)

        x = self._attn(q, k, v, cu_lens, max_len)

        if (lora_names is not None) and isinstance(self.out, LoRA):
            return self.out(x, lora_names)
        else:
            return self.out(x)


class FlashTransformerLayer(nn.Module):
    """
    FlashTransformerLayer is a custom layer that implements a transformer layer with flash attention.

    Args:
        embed_dim (int): The dimension of the embedding.
        ffn_embed_dim (int): The dimension of the feed-forward network embedding.
        attention_heads (int): The number of attention heads.
        rotary_embedding (bool, optional): Whether to use rotary embeddings. Default is True.
        dtype (torch.dtype, optional): The data type for the layer. Default is torch.bfloat16.

    Attributes:
        embed_dim (int): The dimension of the embedding.
        ffn_embed_dim (int): The dimension of the feed-forward network embedding.
        attention_heads (int): The number of attention heads.
        self_attn (FlashMultiheadAttention): The multi-head attention mechanism.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization for the attention mechanism.
        fc1 (nn.Linear): The first linear layer in the feed-forward network.
        fc2 (nn.Linear): The second linear layer in the feed-forward network.
        final_layer_norm (nn.LayerNorm): Layer normalization for the output of the feed-forward network.

    Methods:
        _attn(x, cu_lens, max_len):
            Applies layer normalization and self-attention to the input tensor.
            Args:
                x (torch.Tensor): The input tensor.
                cu_lens (torch.Tensor): Cumulative lengths tensor for packed sequences.
                max_len (int): The maximum length of the sequences.
            Returns:
                torch.Tensor: The output tensor after applying self-attention.
        _head(x):
            Applies layer normalization and the feed-forward network to the input tensor.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying the feed-forward network.
        forward(x, cu_lens, max_len):
            The forward pass of the transformer layer.
            Args:
                x (torch.Tensor): The input tensor.
                cu_lens (torch.Tensor): Cumulative lengths tensor for packed sequences.
                max_len (int): The maximum length of the sequences.
            Returns:
                torch.Tensor: The output tensor after applying the transformer layer.
    """

    def __init__(
        self,
        embed_dim,
        expand_dim,
        attention_heads,
        rotary_embedding=True,
        pre_layernorm=False,
        bias=False,
        residue_scaling=1.,
        final_activation='swiglu',
        dropout=0.0,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_dim = expand_dim
        self.attention_heads = attention_heads
        self.residue_scaling = residue_scaling

        self.self_attn = FlashMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            pre_layernorm=pre_layernorm,
            bias=bias,
            dropout=dropout,
            rotary_embedding=rotary_embedding,
            dtype=dtype
        )

        if final_activation == 'swiglu':
            expanded_embed_dim = int(
                ((expand_dim * self.embed_dim) + 255) // 256 * 256)

            self.final = nn.Sequential(
                nn.LayerNorm(self.embed_dim, dtype=dtype),
                SwiGLU(self.embed_dim, expanded_embed_dim,
                       bias=bias, dtype=dtype),
                nn.Linear(expanded_embed_dim, self.embed_dim,
                          bias=bias, dtype=dtype)
            )
        elif final_activation == 'gelu':
            self.final = nn.Sequential(
                nn.LayerNorm(self.embed_dim, dtype=dtype),
                nn.Linear(self.embed_dim, embed_dim * self.expand_dim,
                          bias=bias, dtype=dtype),
                nn.GELU(),
                nn.Linear(embed_dim * self.expand_dim, self.embed_dim,
                          bias=bias, dtype=dtype)
            )
        else:
            raise ValueError(
                'Invalid final activation function. Must be "swiglu" or "gelu".')

    def forward(self, x, cu_lens, max_len, lora_names=None):
        '''
        Forward pass of the transformer layer.

        Args:
            x (torch.Tensor): The input tensor.
            cu_lens (torch.Tensor): Cumulative lengths tensor for packed sequences.
            max_len (int): The maximum length of the sequences.

        Returns:
            torch.Tensor: The output tensor after applying the transformer layer 
        '''
        x = x + self.self_attn(x, cu_lens, max_len, lora_names) / self.residue_scaling
        return x + self.final(x) / self.residue_scaling


class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.bfloat16):
        """
        SwiGLU activation.

        Args:
            input_dim (int): The input feature size.
            hidden_dim (int): The intermediate hidden size
        """
        super(SwiGLU, self).__init__()

        self.activation = nn.Linear(
            in_features, out_features, bias=bias, dtype=dtype)
        self.fc = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        """
        Forward pass for SwiGLU.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, hidden_dim].
        """
        return F.silu(self.activation(x)) * self.fc(x)
