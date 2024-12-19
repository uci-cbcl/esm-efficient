from pathlib import Path
import torch
from esme import ESMC
from safetensors.torch import save_model

model_name = snakemake.wildcards['model']

if model_name == 'c300m':
    model = ESMC(num_layers=30, embed_dim=960,
                 attention_heads=15).to(torch.bfloat16)
elif model_name == 'c600m':
    model = ESMC(num_layers=36, embed_dim=1152,
                 attention_heads=18).to(torch.bfloat16)
else:
    raise ValueError(f"Unknown model {wildcards.model}")

weights = torch.load(snakemake.input['model'])

state_dict = {
    'embed_tokens.weight': weights['embed.weight'],
    'emb_layer_norm_after.weight': weights['transformer.norm.weight'],
    'lm_head.dense.weight': weights['sequence_head.0.weight'],
    'lm_head.dense.bias': weights['sequence_head.0.bias'],
    'lm_head.layer_norm.weight': weights['sequence_head.2.weight'],
    'lm_head.layer_norm.bias': weights['sequence_head.2.bias'],
    'lm_head.final.weight': weights['sequence_head.3.weight'],
    'lm_head.final.bias': weights['sequence_head.3.bias'],
}

for i in range(model.num_layers):
    state_dict[f'layers.{i}.self_attn.norm.weight'] = weights[f'transformer.blocks.{i}.attn.layernorm_qkv.0.weight']
    state_dict[f'layers.{i}.self_attn.norm.bias'] = weights[f'transformer.blocks.{i}.attn.layernorm_qkv.0.bias']
    q, k, v = weights[f'transformer.blocks.{i}.attn.layernorm_qkv.1.weight'].chunk(
        3, dim=0)
    state_dict[f'layers.{i}.self_attn.q.weight'] = q
    state_dict[f'layers.{i}.self_attn.k.weight'] = k
    state_dict[f'layers.{i}.self_attn.v.weight'] = v

    state_dict[f'layers.{i}.self_attn.out.weight'] = weights[f'transformer.blocks.{i}.attn.out_proj.weight']
    state_dict[f'layers.{i}.self_attn.layernorm_q.weight'] = weights[f'transformer.blocks.{i}.attn.q_ln.weight']
    state_dict[f'layers.{i}.self_attn.layernorm_k.weight'] = weights[f'transformer.blocks.{i}.attn.k_ln.weight']
    state_dict[f'layers.{i}.final.0.weight'] = weights[f'transformer.blocks.{i}.ffn.0.weight']
    state_dict[f'layers.{i}.final.0.bias'] = weights[f'transformer.blocks.{i}.ffn.0.bias']

    _act, _weights = weights[f'transformer.blocks.{i}.ffn.1.weight'].chunk(
        2, dim=0)
    state_dict[f'layers.{i}.final.1.activation.weight'] = _act
    state_dict[f'layers.{i}.final.1.fc.weight'] = _weights
    state_dict[f'layers.{i}.final.2.weight'] = weights[f'transformer.blocks.{i}.ffn.3.weight']

missing, unexpected = model.load_state_dict(state_dict, strict=True)
model = model.to(torch.bfloat16)

model_name = model_name.replace('c', '')

metadata = {
    'format': 'pt',
    'name': f'esmc_{model_name}',
    'num_layers': str(model.num_layers),
    'embed_dim': str(model.embed_dim),
    'attention_heads': str(model.attention_heads)
}

save_model(model, snakemake.output['model'], metadata=metadata)

model = ESMC.from_pretrained(snakemake.output['model'])
