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

model.load_state_dict(weights)
model = model.to(torch.bfloat16)
model.lm_head.weight = model.embed_tokens.weight

metadata = {'format': 'pt'}
if is_esm2:
    metadata['num_layers'] = str(model.num_layers)
    metadata['embed_dim'] = str(model.embed_dim)
    metadata['attention_heads'] = str(model.attention_heads)

save_model(model, snakemake.output['model'], metadata=metadata)

if is_esm2:
    model = ESM2.from_pretrained(snakemake.output['model'])
else:
    model = ESM1b.from_pretrained(snakemake.output['model'])
