import os
import math
import accelerate
import torch
import torch.nn as nn
from tqdm import tqdm
import safetensors.torch as safetensors
from flash_attn.bert_padding import pad_input, unpad_input
from esme.download import download_model, model_names
from esme.head import RobertaLMHead
from esme.quantization import Linear8bit
from esme.attention import FlashTransformerLayer
from esme.alphabet import Alphabet, Alphabet3
from esme.embedding import LearnedPositionalEmbedding
from esme.lora import LoRA, mark_only_lora_as_trainable, lora_state_dict
from esme.deepspeed import import_deepseed, DEEPSPEED_STAGE2_CONFIG


def import_checkpointing(deepspeed=False):
    if deepspeed:
        _checkpoint = import_deepseed().checkpointing.checkpoint
    else:
        from torch.utils.checkpoint import checkpoint
        _checkpoint = lambda *args: checkpoint(*args, use_reentrant=False)
    return _checkpoint


class ESM(nn.Module):

    @staticmethod
    def from_pretrained(path, quantization=None, checkpointing=False, device='cpu'):
        '''
        Load a pretrained model from a safetensors file.

        Args:
            path: str - path to the safetensor model file to load from or name of the model.
            quantization: str - the quantization to use for the model weights.
                One of(None, '8bit', '4bit').
            checkpointing: bool - whether to offload the model to the cpu for training.
            device: str - the device to load the model.
        '''
        if not os.path.isfile(path):
            try:
                path = download_model(path)
            except ValueError:
                raise ValueError(
                    f'Invalid model name: {path}. Must be one of {model_names}'
                )

        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            name = f.metadata()['name'].split('_')[0]

        if name == 'esm1b':
            model = ESM1b.from_pretrained(
                path, quantization, checkpointing, device)
        elif name == 'esm1v':
            model = ESM1v.from_pretrained(
                path, quantization, checkpointing, device)
        elif name == 'esm2':
            model = ESM2.from_pretrained(
                path, quantization, checkpointing, device)
        elif name == 'esmc':
            model = ESMC.from_pretrained(
                path, quantization, checkpointing, device)
        else:
            raise ValueError(
                f'Invalid model name: {name}. Must be one of {model_names}'
            )
        return model


class ESM2(nn.Module):
    '''
    Efficient implementation of the ESM-2 model. Leverages the flash-attn library
    for efficient attention computation. Partition-wise attention is used to
    reduce the memory footprint of the model.

    Args:
        num_layers: int - the number of transformer layers
        embed_dim: int - the embedding dimension
        attention_heads: int - the number of attention heads
        checkpointing: bool - whether to use checkpointing for memory optimization
        rotary_embedding: bool - whether to use rotary embeddings
        dtype: torch.dtype - the datatype of the

    Attributes:
        num_layers: int - the number of transformer layers
        embed_dim: int - the embedding dimension
        attention_heads: int - the number of attention heads
        checkpointing: bool - whether to use checkpointing for memory optimization
        embed_scale: float - the scale of the embeddings
        embed_tokens: nn.Embedding - the embedding layer
        layers: nn.ModuleList - the transformer layers
        emb_layer_norm_after: nn.LayerNorm - the layer norm after the embeddings
        lm_head: RobertaLMHead - the head of the model

    Methods:
        embedding: Get the embeddings of the tokens with the specified scale
        forward_representation: Forward pass through the model without the head
        forward: Forward pass through the model with the head
        predict_log_prob: Predict the log probabilities of the tokens
        predict_prob: Predict the probabilities of the tokens
        create_model: Create a model from a safetensors file with empty weights
        from_pretrained: Load a pretrained model from a safetensors file
        trainable_parameters: Get the trainable parameters of the model
        add_lora: Add LoRA adapters to the model
        mark_only_lora_as_trainable: Mark only the LoRA adapters as trainable
        lora_state_dict: Get the state dict of the LoRA adapters
        save_lora: Save the LoRA adapters to a safetensors file
        load_lora: Load LoRA adapters from a safetensors file
        mark_lmhead: Mark the head of the model as trainable

    Examples:
        >> > model = ESM2.from_pretrained('8M.safetenors')
        >> > model.forward(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.forward_representation(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3, ...], [0.2, 0.3, 0.4, ...], [0.3, 0.4, 0.5 ...]])

        >> > model.predict_log_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.predict_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.add_lora(rank=16, alpha=16, layers=('query', 'value', 'output'))
        >> > model.

    '''

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        checkpointing: bool = False,
        rotary_embedding: bool = True,
        dropout: float = 0.,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.checkpointing = checkpointing

        if self.checkpointing:
            self.checkpoint = import_checkpointing(
                deepspeed=(checkpointing == 'deepspeed'))
            self.checkpointing = True

        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(33, self.embed_dim, dtype=dtype,
                                         padding_idx=Alphabet.padding_idx)
        layers = [
            FlashTransformerLayer(
                self.embed_dim,
                4,
                self.attention_heads,
                rotary_embedding=rotary_embedding,
                pre_layernorm=False,
                bias=True,
                final_activation='gelu',
                dropout=dropout,
                dtype=dtype
            )
            for _ in range(self.num_layers)
        ]

        self.layers = nn.ModuleList(layers)
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim, dtype=dtype)

        self.lm_head = RobertaLMHead(self.embed_dim, 33, dtype=dtype)

    def embedding(self, tokens):
        '''
        Get the embeddings given the tokens.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
                or (batch * seq_len)

        Returns:
            x: torch.Tensor - the embeddings of the tokens with shape
                (batch, seq_len, embed_dim) or (batch * seq_len, embed_dim)
        '''
        x = self.embed_scale * self.embed_tokens(tokens)
        x.masked_fill_((tokens == Alphabet.mask_idx).unsqueeze(-1), .0)

        if tokens.ndim == 2:
            x = torch.where(~tokens.eq(Alphabet.padding_idx).unsqueeze(-1),
                            x, torch.zeros_like(x))
        elif tokens.ndim == 1:
            pass
        else:
            raise ValueError('tokens must be 1D or 2D')

        return x

    def forward_representation(self, tokens, pad_args=None, pad_output=False,
                               pad_indices=None, lora_names=None):
        '''
        Forward pass through the model without the head to get the representation
        per token in the sequence.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to the maximum length
            pad_indices: torch.Tensor - the indices of the padded tokens
        '''
        x = self.embedding(tokens)

        if pad_args is not None:
            assert tokens.ndim == 1, \
                'tokens are expected to be unpadded with shape (batch * seq_len)'
            cu_lens, max_len = pad_args
        else:
            assert tokens.ndim == 2, \
                'tokens are expected to be padded with shape (batch, seq_len, embed_dim)'
            x, pad_indices, cu_lens, max_len, _ = unpad_input(
                hidden_states=x, attention_mask=~tokens.eq(Alphabet.padding_idx))

        for layer in self.layers:
            if self.checkpointing:
                x = self.checkpoint(layer, x, cu_lens, max_len, lora_names)
            else:
                x = layer(x, cu_lens, max_len, lora_names)

        x = self.emb_layer_norm_after(x)

        if pad_output or (pad_args is None):
            x = pad_input(x, pad_indices, len(cu_lens) - 1, max_len)

        return x

    def forward(self, tokens, pad_args=None, pad_output=False,
                pad_indices=None, lora_names=None):
        '''
        Forward pass through the model with the head to get the logits
        of the tokens.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        return self.lm_head(self.forward_representation(
            tokens, pad_args, pad_output, pad_indices, lora_names))

    def predict_log_prob(self, tokens, pad_args=None, pad_output=False,
                         pad_indices=None, lora_names=None):
        '''
        Forward pass through the model with the head to get the log probabilities
        of the tokens.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        return torch.log_softmax(self(tokens, pad_args, pad_output,
                                      pad_indices, lora_names), dim=-1)

    def predict_prob(self, tokens, log=False, pad_args=None, pad_output=False,
                     pad_indices=None, lora_names=None):
        '''
        Forward pass through the model with the head to get the probabilities
        of the tokens.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
            log: bool - whether to return the log probabilities
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        args = (tokens, pad_args, pad_output, pad_indices, lora_names)
        if log:
            return self.predict_log_prob(*args)
        return torch.softmax(self(*args), dim=-1)

    @classmethod
    def create_model(cls, path, checkpointing=False):
        '''
        Create a model from a safetensors file with empty weights.

        Args:
            path: str - path to the safetensor model file
            checkpointing: bool - whether to use checkpointing for memory optimization
        '''
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            name = metadata['name'].split('_')[0]
            assert name == cls.__name__.lower(), \
                f'Invalid weight for the {cls.__name__} model. ' \
                f'You are trying to load a {name} model weights to a {cls.__name__} model.'
            model = cls(
                num_layers=int(metadata['num_layers']),
                embed_dim=int(metadata['embed_dim']),
                attention_heads=int(metadata['attention_heads']),
                checkpointing=checkpointing
            )
        return model

    @classmethod
    def from_pretrained(cls, path, quantization=None, checkpointing=False, device='cpu') -> 'ESM2':
        '''
        Load a pretrained model from a safetensors file.

        Args:
            path: str - path to the safetensor model file to load from
            quantization: str - the quantization to use for the model weights.
                One of(None, '8bit', '4bit').
            checkpointing: bool - whether to use checkpointing for memory optimization
            device: str - the device to load the model.
        '''
        assert quantization in {None, '8bit', '4bit', '8bitexperimental'}, \
            f'load_in must be one of [None, "8bit", "4bit"] but got {quantization}'

        if quantization is not None:
            assert device != 'cpu', \
                'Quantized model cannot be loaded on cpu provide CUDA gpu device'

        with accelerate.init_empty_weights():
            model = cls.create_model(path, checkpointing=checkpointing)

        if quantization == '8bit':
            model = cls._load_8bit(model, path, device)
        elif quantization == '8bitexperimental':
            model = cls._load_8bit_experimental(model, path, device)
        elif quantization == '4bit':
            model = cls._load_4bit(model, path, device)
        else:
            model = accelerate.load_checkpoint_and_dispatch(
                model, path, {'': device})

        return model

    @classmethod
    def _load_lm_head(cls, model, f):
        model.emb_layer_norm_after.weight = nn.Parameter(
            f.get_tensor('emb_layer_norm_after.weight'))
        if 'emb_layer_norm_after.bias' in f.keys():
            model.emb_layer_norm_after.bias = nn.Parameter(
                f.get_tensor('emb_layer_norm_after.bias'))

        model.lm_head.dense.weight = nn.Parameter(
            f.get_tensor('lm_head.dense.weight'))
        model.lm_head.dense.bias = nn.Parameter(
            f.get_tensor('lm_head.dense.bias'))

        model.lm_head.layer_norm.weight = nn.Parameter(
            f.get_tensor('lm_head.layer_norm.weight'))
        model.lm_head.layer_norm.bias = nn.Parameter(
            f.get_tensor('lm_head.layer_norm.bias'))

        model.lm_head.final.weight = nn.Parameter(
            f.get_tensor('lm_head.final.weight'))
        model.lm_head.final.bias = nn.Parameter(
            f.get_tensor('lm_head.final.bias'))

    @classmethod
    def _load_layer_norm(cls, layer, idx, sf):
        getattr(layer.final, '0').weight = nn.Parameter(
            sf.get_tensor(f'layers.{idx}.final.0.weight'))
        if f'layers.{idx}.final.0.bias' in sf.keys():
            getattr(layer.final, '0').bias = nn.Parameter(
                sf.get_tensor(f'layers.{idx}.final.0.bias'))

        getattr(layer.self_attn, 'norm').weight = nn.Parameter(
            sf.get_tensor(f'layers.{idx}.self_attn.norm.weight'))
        if f'layers.{idx}.self_attn.norm.bias' in sf.keys():
            getattr(layer.self_attn, 'norm').bias = nn.Parameter(
                sf.get_tensor(f'layers.{idx}.self_attn.norm.bias'))

    @classmethod
    def _load_linear8bit_experimental(cls, sf, device, weight_key, bias_key=None):
        return Linear8bit(
            sf.get_tensor(weight_key),
            sf.get_tensor(bias_key) if bias_key is not None else None,
            device=device, threshold=6)

    @classmethod
    def _load_linear8bit(cls, sf, device, weight_key, bias_key=None):
        from bitsandbytes.nn import Linear8bitLt
        weight = sf.get_tensor(weight_key).to(torch.float16)
        state_dict = {'weight': weight}

        if bias_key is not None:
            state_dict['bias'] = sf.get_tensor(bias_key).to(torch.float16)

        layer = Linear8bitLt(weight.shape[1], weight.shape[0],
                             has_fp16_weights=False, bias=bias_key is not None)
        layer.load_state_dict(state_dict)
        return layer

    @classmethod
    def _load_linear4bit(cls, sf, device, weight_key, bias_key=None):
        from bitsandbytes.nn import Linear4bit
        weight = sf.get_tensor(weight_key)
        state_dict = {'weight': weight}

        if bias_key is not None:
            state_dict['bias'] = sf.get_tensor(bias_key)

        layer = Linear4bit(weight.shape[1], weight.shape[0],
                           bias=bias_key is not None)
        layer.load_state_dict(state_dict)
        return layer.to(device)

    @classmethod
    def _load_quantize(cls, model, path, device, fn_layer):

        with safetensors.safe_open(path, framework="pt", device='cpu') as sf:
            model.embed_tokens.weight = nn.Parameter(
                sf.get_tensor('embed_tokens.weight'))

            for i, _ in enumerate(tqdm(model.layers, desc='Loading layers')):
                layer = model.layers[i]
                for j in ['q', 'k', 'v', 'out']:
                    setattr(layer.self_attn, j, fn_layer(
                        sf, device,
                        f'layers.{i}.self_attn.{j}.weight',
                        f'layers.{i}.self_attn.{j}.bias',
                    ))
                for j in ['1', '3']:
                    setattr(layer.final, j, fn_layer(
                        sf, device,
                        f'layers.{i}.final.{j}.weight',
                        f'layers.{i}.final.{j}.bias',
                    ))
                cls._load_layer_norm(layer, i, sf)
            cls._load_lm_head(model, sf)

        return model.to(device)

    @classmethod
    def _load_8bit_experimental(cls, model, path, device):
        return cls._load_quantize(model, path, device, cls._load_linear8bit_experimental)

    @classmethod
    def _load_8bit(cls, model, path, device):
        return cls._load_quantize(model, path, device, cls._load_linear8bit)

    @classmethod
    def _load_4bit(cls, model, path, device):
        return cls._load_quantize(model, path, device, cls._load_linear4bit)

    def trainable_parameters(self):
        '''
        Return the trainable parameters of the model.
        '''
        return [
            p for p in self.parameters()
            if p.requires_grad
        ]

    def add_lora(self, rank=16, alpha=16, layers=('query', 'value', 'output'),
                 dropout_p=0., adapter_names=None):
        '''
        Add LoRA adapters to the model.

        Args:
            rank: int - the rank of the LoRA adapters
            alpha: int - the alpha of the LoRA adapters
            layers: list - the layers to add the LoRA adapters to
            dropout_p: float - the dropout probability of the LoRA adapters
            adapter_names: list - the names of the adapters to add to the model
                enabling adding multiple adapters to the same layer.
        '''
        _layers = set(layers)
        assert len(_layers.difference({'query', 'value', 'key', 'output'})) == 0, \
            'layers must be a subset of {"query", "value", "key", "output"}'

        self.lora_kwargs = {
            'rank': rank,
            'alpha': alpha,
            'dropout_p': dropout_p,
            'layers': list(_layers),
            'names': adapter_names
        }

        target_modules = list()

        if 'query' in _layers:
            target_modules.append('q')
        if 'value' in _layers:
            target_modules.append('v')
        if 'key' in _layers:
            target_modules.append('k')
        if 'output' in _layers:
            target_modules.append('out')

        for i, _ in enumerate(tqdm(self.layers, desc='Adding LoRA adapters')):
            layer = self.layers[i]

            for j in target_modules:
                module = getattr(layer.self_attn, j)
                dtype = None if isinstance(module, Linear8bit) else None
                setattr(
                    layer.self_attn, j,
                    LoRA(module, rank=rank, alpha=alpha, dtype=dtype,
                         dropout_p=dropout_p, names=adapter_names)
                )
        self.mark_only_lora_as_trainable(adapter_names)
        return self

    def mark_only_lora_as_trainable(self, adapter_names=None):
        '''
        Mark only the LoRA adapters as trainable.

        Args:
            adapter_names: list - the names of the adapters to mark as trainable.
        '''
        mark_only_lora_as_trainable(self, adapter_names)
        return self

    def lora_state_dict(self, adapter_names=None):
        '''
        Get the state dict of the LoRA adapters.

        Args:
            adapter_names: list - the names of the adapters to get the state dict of.
        '''
        return lora_state_dict(self, adapter_names)

    def save_lora(self, path: str, adapter_names=None):
        '''
        Save the LoRA adapters to a safetensors file.

        Args:
            path: str - path to the safetensor file
            names: list - list of names of the adapters to save
        '''
        state = self.lora_state_dict(adapter_names)
        assert len(state) > 0, 'No LoRA adapters found to save'

        lora_kwargs = self.lora_kwargs
        metadata = {
            'rank': str(lora_kwargs['rank']),
            'alpha': str(lora_kwargs['alpha']),
            'dropout_p': str(lora_kwargs['dropout_p']),
            'layers': ','.join(lora_kwargs['layers']),
            'names': ','.join(adapter_names or lora_kwargs['names']),
            'format': 'pt'
        }
        safetensors.save_file(state, path, metadata)
        return self

    def load_lora(self, path: str, names=None):
        """
        Load LoRA adapters from a safetensors file.

        Args:
            path: str - path to the safetensor file
        """
        with safetensors.safe_open(path, 'pt') as f:
            metadata = f.metadata()
            names = names or metadata['names']
            self.add_lora(
                rank=int(metadata['rank']),
                alpha=float(metadata['alpha']),
                dropout_p=float(metadata['dropout_p']),
                layers=metadata['layers'].split(','),
                adapter_names=metadata.get('names').split(',')
            )
        _, expected = safetensors.load_model(self, path, strict=False)
        assert len(expected) == 0, \
            f"Expected LoRA keys in the model missing state_dict: {expected}"
        return self

    def mark_lmhead(self, trainable=True):
        '''
        Mark the head of the model as trainable.
        '''
        for param in self.lm_head.parameters():
            param.requires_grad_(trainable)
        return self


class ESM1b(ESM2):
    '''
    ESM-1b model with 33 transformer layers, 1280 embedding dimension, and 20 attention heads.

    Args:
        checkpointing: bool - whether to use checkpointing for memory optimization
        dtype: torch.dtype - the datatype of the model
    '''

    def __init__(self, checkpointing: bool = False, dtype=torch.bfloat16):
        super().__init__(num_layers=33, embed_dim=1280, attention_heads=20,
                         checkpointing=checkpointing, rotary_embedding=False, dtype=dtype)

        self.emb_layer_norm_before = nn.LayerNorm(self.embed_dim, dtype=dtype)
        max_seq_len = 4096
        self.embed_positions = LearnedPositionalEmbedding(
            max_seq_len, self.embed_dim)

    def embedding(self, tokens):
        x = self.embed_scale * self.embed_tokens(tokens)
        x.masked_fill_((tokens == Alphabet.mask_idx).unsqueeze(-1), .0)

        if tokens.ndim != 2:
            raise ValueError('tokens must be 2D for esm1b')

        x = self.emb_layer_norm_before(x + self.embed_positions(tokens))
        x = torch.where(~tokens.eq(Alphabet.padding_idx).unsqueeze(-1),
                        x, torch.zeros_like(x))

        return x

    @classmethod
    def _load_emb_layer(cls, model, path, device):
        with safetensors.safe_open(path, framework="pt", device='cpu') as sf:
            model.emb_layer_norm_before.weight = nn.Parameter(
                sf.get_tensor('emb_layer_norm_before.weight'))
            model.emb_layer_norm_before.bias = nn.Parameter(
                sf.get_tensor('emb_layer_norm_before.bias'))
            model.embed_positions.weight = nn.Parameter(
                sf.get_tensor('embed_positions.weight'))

    @classmethod
    def _load_8bit(cls, model, path, device):
        cls._load_emb_layer(model, path, device)
        return super()._load_8bit(model, path, device)

    @classmethod
    def _load_4bit(cls, model, path, device):
        cls._load_emb_layer(model, path, device)
        return super()._load_4bit(model, path, device)

    @classmethod
    def create_model(cls, path, checkpointing=False):
        return cls(checkpointing=checkpointing)


class ESM1v(ESM2):
    '''
    ESM-1v model with 33 transformer layers, 1280 embedding dimension, and 20 attention heads.

    Args:
        checkpointing: bool - whether to use checkpointing for memory optimization
        dtype: torch.dtype - the datatype of the model
    '''

    def __init__(self, checkpointing: bool = False, dtype=torch.bfloat16):
        super().__init__(num_layers=33, embed_dim=1280, attention_heads=20,
                         checkpointing=checkpointing, rotary_embedding=False, dtype=dtype)
        max_seq_len = 4096
        self.embed_positions = LearnedPositionalEmbedding(
            max_seq_len, self.embed_dim)

    def embedding(self, tokens):
        x = self.embed_scale * self.embed_tokens(tokens)
        x.masked_fill_((tokens == Alphabet.mask_idx).unsqueeze(-1), .0)

        if tokens.ndim != 2:
            raise ValueError('tokens must be 2D for esm1b')

        x += self.embed_positions(tokens)
        x = torch.where(~tokens.eq(Alphabet.padding_idx).unsqueeze(-1),
                        x, torch.zeros_like(x))
        return x

    @classmethod
    def create_model(cls, path, checkpointing=False):
        return cls(checkpointing=checkpointing)

    @classmethod
    def _load_emb_layer(cls, model, path, device):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            model.embed_positions.weight = nn.Parameter(
                f.get_tensor('embed_positions.weight'))

    @classmethod
    def _load_8bit(cls, model, path, device):
        cls._load_emb_layer(model, path, device)
        return super()._load_8bit(model, path, device)

    @classmethod
    def _load_4bit(cls, model, path, device):
        cls._load_emb_layer(model, path, device)
        return super()._load_4bit(model, path, device)


class ESMC(ESM2):
    '''
    Efficient implementation of the ESM-3 model. Leverages the flash-attn library
    for efficient attention computation. Partition-wise attention is used to
    reduce the memory footprint of the model.

    Args:
        num_layers: int - the number of transformer layers
        embed_dim: int - the embedding dimension
        attention_heads: int - the number of attention heads
        checkpointing: bool - whether to use checkpointing for memory optimization
        rotary_embedding: bool - whether to use rotary embeddings
        dtype: torch.dtype - the datatype of the

    Attributes:
        num_layers: int - the number of transformer layers
        embed_dim: int - the embedding dimension
        attention_heads: int - the number of attention heads
        checkpointing: bool - whether to use checkpointing for memory optimization
        embed_scale: float - the scale of the embeddings
        embed_tokens: nn.Embedding - the embedding layer
        layers: nn.ModuleList - the transformer layers
        emb_layer_norm_after: nn.LayerNorm - the layer norm after the embeddings
        lm_head: RobertaLMHead - the head of the model

    Methods:
        embedding: Get the embeddings of the tokens with the specified scale
        forward_representation: Forward pass through the model without the head
        forward: Forward pass through the model with the head
        predict_log_prob: Predict the log probabilities of the tokens
        predict_prob: Predict the probabilities of the tokens
        create_model: Create a model from a safetensors file with empty weights
        from_pretrained: Load a pretrained model from a safetensors file
        trainable_parameters: Get the trainable parameters of the model
        add_lora: Add LoRA adapters to the model
        mark_only_lora_as_trainable: Mark only the LoRA adapters as trainable
        lora_state_dict: Get the state dict of the LoRA adapters
        save_lora: Save the LoRA adapters to a safetensors file
        load_lora: Load LoRA adapters from a safetensors file
        mark_lmhead: Mark the head of the model as trainable

    Examples:
        >> > model = ESMC.from_pretrained('esmc_300m')
        >> > model.forward(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.forward_representation(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3, ...], [0.2, 0.3, 0.4, ...], [0.3, 0.4, 0.5 ...]])

        >> > model.predict_log_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.predict_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >> > model.add_lora(rank=16, alpha=16, layers=('query', 'value', 'output'))
        >> > model.

    '''

    def __init__(
        self,
        num_layers: int = 30,
        embed_dim: int = 960,
        attention_heads: int = 15,
        checkpointing: bool = False,
        dropout: float = 0.,
        dtype=torch.bfloat16,
    ):
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            checkpointing=checkpointing,
            rotary_embedding=True,
            dropout=dropout,
            dtype=dtype
        )
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.checkpointing = checkpointing

        if self.checkpointing:
            self.checkpoint = import_checkpointing(
                deepspeed=(checkpointing == 'deepspeed'))
            self.checkpointing = True

        # TODO
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(64, self.embed_dim, dtype=dtype,
                                         padding_idx=Alphabet3.padding_idx)
        layers = [
            FlashTransformerLayer(
                self.embed_dim,
                8 / 3,
                self.attention_heads,
                rotary_embedding=True,
                pre_layernorm=True,
                bias=False,
                final_activation='swiglu',
                residue_scaling=math.sqrt(num_layers / 36),
                dropout=dropout,
                dtype=dtype
            )
            for _ in range(self.num_layers)
        ]

        self.layers = nn.ModuleList(layers)
        self.emb_layer_norm_after = nn.LayerNorm(
            self.embed_dim, dtype=dtype, bias=False)

        self.lm_head = RobertaLMHead(self.embed_dim, 64, dtype=dtype)

    def forward_representation(self, tokens, pad_args=None, pad_output=False,
                               pad_indices=None, lora_names=None):
        '''
        Forward pass through the model without the head to get the representation
        per token in the sequence.

        Args:
            tokens: torch.Tensor - the input tokens with shape(batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to the maximum length
            pad_indices: torch.Tensor - the indices of the padded tokens
        '''
        x = self.embed_tokens(tokens)

        if pad_args is not None:
            assert tokens.ndim == 1, \
                'tokens are expected to be unpadded with shape (batch * seq_len)'
            cu_lens, max_len = pad_args
        else:
            assert tokens.ndim == 2, \
                'tokens are expected to be padded with shape (batch, seq_len, embed_dim)'
            x, pad_indices, cu_lens, max_len, _ = unpad_input(
                hidden_states=x, attention_mask=~tokens.eq(Alphabet3.padding_idx))

        for layer in self.layers:
            if self.checkpointing:
                x = self.checkpoint(layer, x, cu_lens, max_len, lora_names)
            else:
                x = layer(x, cu_lens, max_len, lora_names)

        x = self.emb_layer_norm_after(x)

        if pad_output or (pad_args is None):
            x = pad_input(x, pad_indices, len(cu_lens) - 1, max_len)

        return x

    @classmethod
    def _load_quantize(cls, model, path, device, fn_layer):

        with safetensors.safe_open(path, framework="pt", device='cpu') as sf:
            model.embed_tokens.weight = nn.Parameter(
                sf.get_tensor('embed_tokens.weight'))

            for i, _ in enumerate(tqdm(model.layers, desc='Loading layers')):
                layer = model.layers[i]
                for j in ['q', 'k', 'v', 'out']:
                    setattr(layer.self_attn, j, fn_layer(
                        sf, device,
                        f'layers.{i}.self_attn.{j}.weight',
                    ))
                for j in ['layernorm_q', 'layernorm_k']:
                    getattr(layer.self_attn, j).weight = nn.Parameter(
                        sf.get_tensor(f'layers.{i}.self_attn.{j}.weight'))
                setattr(layer.final[1], 'activation', fn_layer(
                    sf, device,
                    f'layers.{i}.final.1.activation.weight',
                ))
                setattr(layer.final[1], 'fc', fn_layer(
                    sf, device,
                    f'layers.{i}.final.1.fc.weight',
                ))
                setattr(layer.final, '2', fn_layer(
                    sf, device,
                    f'layers.{i}.final.2.weight',
                ))
                cls._load_layer_norm(layer, i, sf)
            cls._load_lm_head(model, sf)
        return model.to(device)
