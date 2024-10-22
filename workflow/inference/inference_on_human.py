from tqdm import tqdm
import time
import torch
from esme.data import FastaDataset, FastaTokenDataset

if snakemake.wildcards['model'].endswith('e'):
    dl = FastaTokenDataset(
        snakemake.input['fasta'],
        token_per_batch=100_000,
        max_len=3500,
    ).to_dataloader(num_workers=snakemake.threads)
else:
    dl = FastaDataset(
        snakemake.input['fasta'],
        max_len=3500,
    ).to_dataloader(batch_size=16, num_workers=snakemake.threads)

device = 0

if snakemake.wildcards['model'].endswith('e'):
    from esme import ESM2
    model = ESM2.from_pretrained(snakemake.input['model'], device=device)
else:
    import esm
    model, _ = esm.pretrained.load_model_and_alphabet(snakemake.input['model'])
    model = model.to(torch.bfloat16).to(device)

model.eval()

t = time.time()

with torch.no_grad():
    for tokens in tqdm(dl):
        out = model(tokens.to(device))

t = t - time.time()

open(snakemake.output['runtime'], 'w').write(str(t))
