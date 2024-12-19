from tqdm import tqdm
import time
import torch
from esme.data import FastaDataset, FastaTokenDataset

efficient = snakemake.wildcards['model'].endswith('e')

if efficient:
    dl = FastaTokenDataset(
        snakemake.input['fasta'],
        token_per_batch=50_000,
        max_len=3500,
    ).to_dataloader(num_workers=snakemake.threads)
else:
    dl = FastaDataset(
        snakemake.input['fasta'],
        max_len=3500,
    ).to_dataloader(batch_size=8, num_workers=snakemake.threads)

device = 5

if efficient:
    if snakemake.wildcards['model'].startswith('c'):
        from esme import ESMC
        model = ESMC.from_pretrained(snakemake.input['model'], device=device)
    else:
        from esme import ESM2
        model = ESM2.from_pretrained(snakemake.input['model'], device=device)
else:
    if snakemake.wildcards['model'].startswith('c'):
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        if snakemake.wildcards['model'] == 'c300m':
            model = ESMC(
                d_model=960, n_heads=15, n_layers=30,
                tokenizer=get_esmc_model_tokenizers()
            )
        elif snakemake.wildcards['model'] == 'c600m':
            model = ESMC(
                d_model=1152, n_heads=18, n_layers=36,
                tokenizer=get_esmc_model_tokenizers()
            ).eval()

        model.load_state_dict(torch.load(snakemake.input['model']))
    else:
        import fair_esm
        model, _ = fair_esm.pretrained.load_model_and_alphabet(
            snakemake.input['model'])

    model = model.to(torch.bfloat16).to(device)

model.eval()

t = time.time()

with torch.no_grad():
    if efficient:
        for tokens, (cu_lens, max_len) in tqdm(dl):
            out = model(tokens.to(device), (cu_lens.to(device), max_len))
    else:
        for tokens in tqdm(dl):
            out = model(tokens.to(device))

t = t - time.time()

open(snakemake.output['runtime'], 'w').write(str(t))
