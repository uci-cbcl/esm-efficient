import pandas as pd
import torch
from tqdm import tqdm
from workflow.utils import benchmark_memory


device = 0
torch.cuda.set_device(device)

quantization = snakemake.wildcards['quantize']
quantization = None if quantization == 'none' else quantization

tokens = torch.load(snakemake.input['tokens'])

if snakemake.wildcards['model'].endswith('e'):
    from esme import ESM2
    model = ESM2.from_pretrained(snakemake.input['model'],
                                 quantization=quantization, device=device)
else:
    assert quantization is None
    import esm
    model, _ = esm.pretrained.load_model_and_alphabet(snakemake.input['model'])
    model = model.to(device).to(torch.bfloat16)

model.eval()


def fn(x):
    with torch.no_grad():
        out = model(x)


memory_usage = list()

for length, token in tqdm(tokens.items()):

    try:
        mem = benchmark_memory(fn, {'x': token.to(device)}, device)
    except torch.cuda.OutOfMemoryError:
        mem = -1  # oom

    memory_usage.append({'length': length, 'mem_gb': mem})
    pd.DataFrame(memory_usage).to_csv(snakemake.output['mem'], index=False)

    if mem == -1:
        break
