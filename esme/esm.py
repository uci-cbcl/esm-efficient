import accelerate
import torch
import torch.nn as nn
from tqdm import tqdm
import safetensors.torch as safetensors
from flash_attn.bert_padding import pad_input, unpad_input
from esme.head import RobertaLMHead
from esme.quantization import Linear8bit
from esme.attention import FlashTransformerLayer
from esme.alphabet import padding_idx
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
        cpuoffload: bool - whether to offload the model to the cpu
        rotary_embedding: bool - whether to use rotary embeddings
        dtype: torch.dtype - the datatype of the
        
    Attributes:
        num_layers: int - the number of transformer layers
        embed_dim: int - the embedding dimension
        attention_heads: int - the number of attention heads
        checkpointing: bool - whether to use checkpointing for memory optimization
        cpuoffload: bool - whether to offload the model to the cpu
        embed_scale: float - the scale of the embeddings
        embed_tokens: nn.Embedding - the embedding layer
        layers: nn.ModuleList - the transformer layers
        emb_layer_norm_after: nn.LayerNorm - the layer norm after the embeddings
        lm_head: RobertaLMHead - the head of the model
    
    Methods:
        tie_weights: Tie the weights of the head to the embedding
        untie_weights: Untie the weights of the head from the embedding
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
        >>> model = ESM2.from_pretrained('8M.safetenors')
        >>> model.forward(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        
        >>> model.forward_representation(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3, ...], [0.2, 0.3, 0.4, ...], [0.3, 0.4, 0.5 ...]])

        >>> model.predict_log_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        
        >>> model.predict_prob(torch.tensor([1, 2, 3]))
        ... tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

        >>> model.add_lora(rank=16, alpha=16, layers=('query', 'value', 'output'))
        >>> model.
        
    '''

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        checkpointing: bool = False,
        cpuoffload: bool = False,
        rotary_embedding: bool = True,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.checkpointing = checkpointing
        self.cpuoffload = cpuoffload

        if self.checkpointing:
            self.checkpoint = import_checkpointing(
                deepspeed=(checkpointing == 'deepspeed'))
            self.checkpointing = True

        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(33, self.embed_dim, dtype=dtype,
                                         padding_idx=padding_idx)
        layers = [
            FlashTransformerLayer(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                rotary_embedding=rotary_embedding,
                dtype=dtype
            )
            for _ in range(self.num_layers)
        ]

        if self.cpuoffload:
            hook = None
            _layers = list()
            for layer in layers:
                layer, hook = accelerate.cpu_offload_with_hook(
                    layer, layer.device, hook)
                _layers.append(layer)
            layers = _layers

        self.layers = nn.ModuleList(layers)
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim, dtype=dtype)

        self.lm_head = RobertaLMHead.from_embedding(
            self.embed_tokens, dtype=dtype)

    def tie_weights(self):
        '''
        Tie the weights of the head to the embedding.
        '''
        self.lm_head.weight = self.embed_tokens.weight

    def untie_weights(self):
        '''
        Untie the weights of the head from the embedding.
        Might be needed for checkpointing if saving tied weights might cause issues.
        '''
        self.lm_head.weight = nn.Parameter(
            torch.clone(self.embed_tokens.weight.data))

    def embedding(self, tokens):
        '''
        Get the embeddings given the tokens.
        
        Args:
            tokens: torch.Tensor - the input tokens with shape (batch, seq_len)
                or (batch * seq_len)
                
        Returns:
            x: torch.Tensor - the embeddings of the tokens with shape 
                (batch, seq_len, embed_dim) or (batch * seq_len, embed_dim)
        '''
        x = self.embed_scale * self.embed_tokens(tokens)

        if tokens.ndim == 2:
            x = torch.where(~tokens.eq(padding_idx).unsqueeze(-1),
                            x, torch.zeros_like(x))
        elif tokens.ndim == 1:
            pass
        else:
            raise ValueError('tokens must be 1D or 2D')

        return x

    def forward_representation(self, tokens, pad_args=None, pad_output=False,
                               pad_indices=None):
        '''
        Forward pass through the model without the head to get the representation
        per token in the sequence.

        Args:
            tokens: torch.Tensor - the input tokens with shape (batch, seq_len)
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
            x, pad_indices, cu_lens, max_len = unpad_input(
                hidden_states=x, attention_mask=~tokens.eq(padding_idx))

        for layer in self.layers:
            if self.checkpointing:
                x = self.checkpoint(layer, x, cu_lens, max_len)
            else:
                x = layer(x, cu_lens, max_len)

        x = self.emb_layer_norm_after(x)

        if pad_output or (pad_args is None):
            x = pad_input(x, pad_indices, len(cu_lens) - 1, max_len)

        return x

    def forward(self, tokens, pad_args=None, pad_output=False, pad_indices=None):
        '''
        Forward pass through the model with the head to get the logits
        of the tokens. 

        Args:
            tokens: torch.Tensor - the input tokens with shape (batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        return self.lm_head(self.forward_representation(
            tokens, pad_args, pad_output, pad_indices))

    def predict_log_prob(self, tokens, pad_args=None, pad_output=False, pad_indices=None):
        '''
        Forward pass through the model with the head to get the log probabilities
        of the tokens.
        
        Args:
            tokens: torch.Tensor - the input tokens with shape (batch, seq_len)
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        return torch.log_softmax(self(tokens, pad_args, pad_output, pad_indices), dim=-1)

    def predict_prob(self, tokens, log=False, pad_args=None, pad_output=False, pad_indices=None):
        '''
        Forward pass through the model with the head to get the probabilities
        of the tokens.
        
        Args:
            tokens: torch.Tensor - the input tokens with shape (batch, seq_len)
            log: bool - whether to return the log probabilities
            pad_args: Tuple[torch.Tensor, int] - (cu_lens, max_len) the cumulative lengths and the
                maximum length of the sequences
            pad_output: bool - whether to pad the output to output.
            pad_indices: torch.Tensor - the indices of the padded tokens.
        '''
        if log:
            return self.predict_log_prob(tokens, pad_args, pad_output, pad_indices)
        return torch.softmax(self(tokens, pad_args, pad_output, pad_indices), dim=-1)

    @classmethod
    def create_model(cls, path, cpuoffload=False, checkpointing=False):
        '''
        Create a model from a safetensors file with empty weights.

        Args:
            path: str - path to the safetensor model file
            cpuoffload: bool - whether to offload the model to the cpu for training
            checkpointing: bool - whether to use checkpointing for memory optimization
        '''
        with safetensors.safe_open(path, "pt") as f:
            metadata = f.metadata()
            model = cls(
                num_layers=int(metadata['num_layers']),
                embed_dim=int(metadata['embed_dim']),
                attention_heads=int(metadata['attention_heads']),
                cpuoffload=cpuoffload, checkpointing=checkpointing
            )
        return model

    @classmethod
    def from_pretrained(cls, path, quantization=None, cpuoffload=False,
                        checkpointing=False, device='cpu') -> 'ESM2':
        '''
        Load a pretrained model from a safetensors file.

        Args:
            path: str - path to the safetensor model file to load from
            quantization: str - the quantization to use for the model weights.
                One of (None, '8bit', '4bit').
            cpuoffload: bool - whether to offload the model to the cpu for training
            checkpointing: bool - whether to use checkpointing for memory optimization
            device: str - the device to load the model.
        '''
        assert quantization in {None, '8bit', '4bit', '8bitexperimental'}, \
            f'load_in must be one of [None, "8bit", "4bit"] but got {quantization}'

        if quantization is not None:
            assert device != 'cpu', \
                'Quantized model cannot be loaded on cpu provide CUDA gpu device'

        with accelerate.init_empty_weights():
            model = cls.create_model(path, cpuoffload=cpuoffload,
                                     checkpointing=checkpointing)
        # needed to avoid bug bcs of meta device for shared weights not same
        # tie weights of the head to the embedding
        model.tie_weights()

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
    def _load_8bit_experimental(cls, model, path, device):
        with safetensors.safe_open(path, framework="pt", device=device) as f:
            model.embed_tokens.weight = nn.Parameter(
                f.get_tensor('embed_tokens.weight'))

            for i, _ in enumerate(tqdm(model.layers, desc='Loading layers')):
                layer = model.layers[i]
                for j in ['q', 'k', 'v', 'out']:
                    linaer = Linear8bit(
                        f.get_tensor(f'layers.{i}.self_attn.{j}.weight'),
                        f.get_tensor(f'layers.{i}.self_attn.{j}.bias'),
                        device=device, threshold=6)
                    setattr(layer.self_attn, j, linaer)
                for j in ['fc1', 'fc2']:
                    linear = Linear8bit(
                        f.get_tensor(f'layers.{i}.{j}.weight'),
                        f.get_tensor(f'layers.{i}.{j}.bias'),
                        device=device, threshold=6)
                    setattr(layer, j, linear)
                for j in ['final_layer_norm', 'self_attn_layer_norm']:
                    getattr(layer, j).weight = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.weight'))
                    getattr(layer, j).bias = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.bias'))

            cls._load_lm_head(model, f)

        return model.to(device)

    @classmethod
    def _load_lm_head(cls, model, f):
        model.emb_layer_norm_after.weight = nn.Parameter(
            f.get_tensor('emb_layer_norm_after.weight'))
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
        model.lm_head.bias = nn.Parameter(
            f.get_tensor('lm_head.bias'))

        model.tie_weights()

    @classmethod
    def _load_8bit(cls, model, path, device):
        from bitsandbytes.nn import Linear8bitLt

        with safetensors.safe_open(path, framework="pt", device='cpu') as f:
            model.embed_tokens.weight = nn.Parameter(
                f.get_tensor('embed_tokens.weight'))

            for i, _ in enumerate(tqdm(model.layers, desc='Loading layers')):
                layer = model.layers[i]
                for j in ['q', 'k', 'v', 'out']:
                    weight = f.get_tensor(
                        f'layers.{i}.self_attn.{j}.weight').to(torch.float16)
                    bias = f.get_tensor(
                        f'layers.{i}.self_attn.{j}.bias').to(torch.float16)

                    l = Linear8bitLt(
                        weight.shape[1], weight.shape[0], has_fp16_weights=False)
                    l.load_state_dict({'weight': weight, 'bias': bias})
                    setattr(layer.self_attn, j, l)
                for j in ['fc1', 'fc2']:
                    weight = f.get_tensor(
                        f'layers.{i}.{j}.weight').to(torch.float16)

                    bias = f.get_tensor(
                        f'layers.{i}.{j}.bias').to(torch.float16)

                    l = Linear8bitLt(
                        weight.shape[1], weight.shape[0], has_fp16_weights=False)
                    l.load_state_dict({'weight': weight, 'bias': bias})
                    setattr(layer, j, l)
                for j in ['final_layer_norm', 'self_attn_layer_norm']:
                    getattr(layer, j).weight = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.weight'))
                    getattr(layer, j).bias = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.bias'))

            cls._load_lm_head(model, f)

        return model.to(device)

    @classmethod
    def _load_4bit(cls, model, path, device):
        from bitsandbytes.nn import Linear4bit

        with safetensors.safe_open(path, framework="pt", device='cpu') as f:
            model.embed_tokens.weight = nn.Parameter(
                f.get_tensor('embed_tokens.weight'))

            for i, _ in enumerate(tqdm(model.layers, desc='Loading layers')):
                layer = model.layers[i]
                for j in ['q', 'k', 'v', 'out']:
                    weight = f.get_tensor(f'layers.{i}.self_attn.{j}.weight')
                    bias = f.get_tensor(f'layers.{i}.self_attn.{j}.bias')
                    l = Linear4bit(weight.shape[1], weight.shape[0])
                    l.load_state_dict({'weight': weight, 'bias': bias})
                    l = l.to(device)
                    setattr(layer.self_attn, j, l)
                for j in ['fc1', 'fc2']:
                    weight = f.get_tensor(f'layers.{i}.{j}.weight')
                    bias = f.get_tensor(f'layers.{i}.{j}.bias')
                    l = Linear4bit(weight.shape[1], weight.shape[0])
                    l.load_state_dict({'weight': weight, 'bias': bias})
                    l = l.to(device)
                    setattr(layer, j, l)
                for j in ['final_layer_norm', 'self_attn_layer_norm']:
                    getattr(layer, j).weight = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.weight'))
                    getattr(layer, j).bias = nn.Parameter(
                        f.get_tensor(f'layers.{i}.{j}.bias'))

            cls._load_lm_head(model, f)

        return model.to(device)

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
        cpuoffload: bool - whether to offload the model to the cpu
        dtype: torch.dtype - the datatype of the model
    '''

    def __init__(self, checkpointing: bool = False, cpuoffload: bool = False,
                 dtype=torch.bfloat16):
        super().__init__(num_layers=33, embed_dim=1280, attention_heads=20,
                         checkpointing=checkpointing, cpuoffload=cpuoffload,
                         rotary_embedding=False, dtype=dtype)

        self.emb_layer_norm_before = nn.LayerNorm(self.embed_dim, dtype=dtype)
        max_seq_len = 4096
        self.embed_positions = LearnedPositionalEmbedding(
            max_seq_len, self.embed_dim)

    def embedding(self, tokens):
        x, attention_mask = super().embedding(tokens)
        x = self.emb_layer_norm_before(x + self.embed_positions(tokens))

        if attention_mask is not None:
            x = torch.where(attention_mask.unsqueeze(-1),
                            x, torch.zeros_like(x))

        return x, attention_mask

    @classmethod
    def _load_emb_layer(cls, model, path, device):
        with safetensors.safe_open(path, framework="pt", device='cpu') as f:
            model.emb_layer_norm_before.weight = nn.Parameter(
                f.get_tensor('emb_layer_norm_before.weight'))
            model.emb_layer_norm_before.bias = nn.Parameter(
                f.get_tensor('emb_layer_norm_before.bias'))
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

    @classmethod
    def create_model(cls, path, cpuoffload=False, checkpointing=False):
        return cls(cpuoffload=cpuoffload, checkpointing=checkpointing)


class ESM1v(ESM2):
    '''
    ESM-1v model with 33 transformer layers, 1280 embedding dimension, and 20 attention heads.
    
    Args:
        checkpointing: bool - whether to use checkpointing for memory optimization
        cpuoffload: bool - whether to offload the model to the cpu
        dtype: torch.dtype - the datatype of the model
    '''

    def __init__(self, checkpointing: bool = False, cpuoffload: bool = False,
                 dtype=torch.bfloat16):
        super().__init__(num_layers=33, embed_dim=1280, attention_heads=20,
                         checkpointing=checkpointing, cpuoffload=cpuoffload,
                         rotary_embedding=False, dtype=dtype)
        max_seq_len = 4096
        self.embed_positions = LearnedPositionalEmbedding(
            max_seq_len, self.embed_dim)

    def embedding(self, tokens):
        x, attention_mask = super().embedding(tokens)
        x += self.embed_positions(tokens)

        x = torch.where(attention_mask.unsqueeze(-1), x, torch.zeros_like(x))

        return x, attention_mask

    @classmethod
    def create_model(cls, path, cpuoffload=False, checkpointing=False):
        return cls(cpuoffload=cpuoffload, checkpointing=checkpointing)

    @classmethod
    def _load_emb_layer(cls, model, path, device):
        with safetensors.safe_open(path, framework="pt", device='cpu') as f:
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
