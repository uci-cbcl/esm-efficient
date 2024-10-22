import pandas as pd
import esm
import torch
import torch.utils.benchmark as benchmark
from tqdm import tqdm
from esme import ESM2


device = snakemake.params['device']
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
    model = model.to(torch.bfloat16).to(device)
model.eval()


def fn(x):
    with torch.no_grad():
        out = model(x)


runtimes = list()

for length, token in tqdm(tokens.items()):
    timer = benchmark.Timer(
        stmt='fn(x)',
        globals={'x': token.to(device), 'fn': fn},
        num_threads=torch.get_num_threads()
    )
    try:
        t = timer.timeit(10).mean
    except torch.cuda.OutOfMemoryError:
        t = -1  # oom

    runtimes.append({
        'length': length,
        'runtime': t
    })

    pd.DataFrame(runtimes).to_csv(snakemake.output['runtime'], index=False)
