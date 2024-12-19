from pathlib import Path
import torch
from esme import ESM2, ESM1b
from safetensors.torch import save_model


model_path = Path(snakemake.input['model'])
params = torch.load(model_path)
weights = dict()

for k, v in params['model'].items():
    if k.endswith('.rot_emb.inv_freq'):
        continue
    for i in ['sentence_encoder.', 'encoder.', '_proj']:
        k = k.replace(i, '')
    weights[k] = v

is_esm2 = model_path.name.startswith('esm2')

if is_esm2:
    model = ESM2(
        num_layers=params['cfg']['model'].encoder_layers,
        embed_dim=params['cfg']['model'].encoder_embed_dim,
        attention_heads=params['cfg']['model'].encoder_attention_heads,
    )
else:
    model = ESM1b()

state_dict = {
    'embed_tokens.weight': weights['embed_tokens.weight'],
    'lm_head.layer_norm.bias': weights['lm_head.layer_norm.bias'],
    'lm_head.layer_norm.weight': weights['lm_head.layer_norm.weight'],
    'emb_layer_norm_after.weight': weights['emb_layer_norm_after.weight'],
    'emb_layer_norm_after.bias': weights['emb_layer_norm_after.bias'],
    'lm_head.dense.weight': weights['lm_head.dense.weight'],
    'lm_head.dense.bias': weights['lm_head.dense.bias'],
    'lm_head.final.weight': weights['lm_head.weight'],
    'lm_head.final.bias': weights['lm_head.bias'],
}

for i in range(model.num_layers):
    state_dict[f'layers.{i}.self_attn.norm.weight'] = weights[
        f'layers.{i}.self_attn_layer_norm.weight']
    state_dict[f'layers.{i}.self_attn.norm.bias'] = weights[
        f'layers.{i}.self_attn_layer_norm.bias']

    for j in ['q', 'k', 'v', 'out']:
        state_dict[f'layers.{i}.self_attn.{j}.weight'] = weights[
            f'layers.{i}.self_attn.{j}.weight']
        state_dict[f'layers.{i}.self_attn.{j}.bias'] = weights[
            f'layers.{i}.self_attn.{j}.bias']

    state_dict[f'layers.{i}.final.0.weight'] = weights[
        f'layers.{i}.final_layer_norm.weight']
    state_dict[f'layers.{i}.final.0.bias'] = weights[
        f'layers.{i}.final_layer_norm.bias']
    state_dict[f'layers.{i}.final.1.weight'] = weights[
        f'layers.{i}.fc1.weight']
    state_dict[f'layers.{i}.final.1.bias'] = weights[
        f'layers.{i}.fc1.bias']
    state_dict[f'layers.{i}.final.3.weight'] = weights[
        f'layers.{i}.fc2.weight']
    state_dict[f'layers.{i}.final.3.bias'] = weights[
        f'layers.{i}.fc2.bias']

model.load_state_dict(state_dict)
model = model.to(torch.bfloat16)

model_name = snakemake.wildcards['model'].lower()
metadata = {
    'format': 'pt',
    'name': f'esm2_{model_name}'
}
if is_esm2:
    metadata['num_layers'] = str(model.num_layers)
    metadata['embed_dim'] = str(model.embed_dim)
    metadata['attention_heads'] = str(model.attention_heads)

save_model(model, snakemake.output['model'], metadata=metadata)

if is_esm2:
    model = ESM2.from_pretrained(snakemake.output['model'])
else:
    model = ESM1b.from_pretrained(snakemake.output['model'])
