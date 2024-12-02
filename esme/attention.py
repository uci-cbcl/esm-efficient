import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func
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
        dtype=torch.bfloat16,
        rotary_embedding=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.k = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.v = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.out = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)

        if rotary_embedding:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)
        else:
            self.rot_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight, gain=1 / 2**.5)
        nn.init.xavier_uniform_(self.v.weight, gain=1 / 2**.5)
        nn.init.xavier_uniform_(self.q.weight, gain=1 / 2**.5)
        nn.init.xavier_uniform_(self.out.weight)

    def _qkv(self, x):
        q = self.q(x) * self.scaling
        k = self.k(x)
        v = self.v(x)

        qkv = torch.stack(tensors=(q, k, v), dim=1)
        # qkv = qkv.contiguous().view(-1, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 't l (h d) -> t l h d', h=self.num_heads)

        return qkv

    def _attn(self, qkv, cu_lens, max_len):
        '''
        '''
        x = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_lens,
            max_seqlen=max_len,
            dropout_p=self.dropout,
            softmax_scale=1,
            causal=False,
        )
        return rearrange(x, 't h d -> t (h d)')

    def forward(self, x: Tensor, cu_lens, max_len) -> Tensor:
        '''
        '''
        qkv = self._qkv(x)

        if self.rot_emb:
            qkv = self.rot_emb(qkv, cu_lens, max_len)

        x = self._attn(qkv, cu_lens, max_len)
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
        ffn_embed_dim,
        attention_heads,
        rotary_embedding=True,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.self_attn = FlashMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            rotary_embedding=rotary_embedding,
            dtype=dtype
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, dtype=dtype)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim,
                             bias=True, dtype=dtype)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim,
                             bias=True, dtype=dtype)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, dtype=dtype)

    def _attn(self, x, cu_lens, max_len):
        x = self.self_attn_layer_norm(x)
        return self.self_attn(x, cu_lens, max_len)

    def _head(self, x):
        x = self.final_layer_norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x

    def forward(self, x, cu_lens, max_len):
        '''
        Forward pass of the transformer layer.

        Args:
            x (torch.Tensor): The input tensor.
            cu_lens (torch.Tensor): Cumulative lengths tensor for packed sequences.
            max_len (int): The maximum length of the sequences.

        Returns:
            torch.Tensor: The output tensor after applying the transformer layer 
        '''
        x = x + self._attn(x, cu_lens, max_len)
        return x + self._head(x)
