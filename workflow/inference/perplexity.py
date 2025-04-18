import random
from tqdm import tqdm
import time
import torch
import pandas as pd
from esme.fasta import Fasta
from esme.variant import predict_pseudoperplexity


model_name = snakemake.wildcards['model']
efficient = model_name.endswith('e') or model_name.startswith('1ve')
device = snakemake.params['device']
max_len = None

if efficient:
    batch_size = 16
    from esme import ESM
    model = ESM.from_pretrained(snakemake.input['model'], device=device)
else:
    batch_size = 1
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

        if snakemake.wildcards['model'] == '15B':
            max_len = 1024
        elif snakemake.wildcards['model'] == '1b':
            max_len = 1024
        elif snakemake.wildcards['model'].startswith('1v'):
            max_len = 1024

    model = model.to(device)

def predict_pseudoperplexity(model, seq: str, batch_size=32,
                             max_len=None, alphabet=Alphabet3):
    '''
    Predict the pseudo-perplexity of sequence.
    '''
    device = next(model.parameters()).device

    if isinstance(seq, str):
        dl = DataLoader(MaskMarginDataset(seq, max_len=max_len, alphabet=alphabet),
                        batch_size=batch_size, shuffle=False)
    elif isinstance(seq, DataLoader):
        dl = seq
    elif isinstance(seq, MaskMarginDataset):
        dl = DataLoader(seq, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError('seq must be str or DataLoader')

    perplexity = Perplexity().to(device)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            if isinstance(model, ESM2) or isinstance(model, ESMC): 
                probs = model(batch['token'].to(device), pad_output=True)
            elif isinstance(model, _ESMC):
                token = batch['token'].to(device)
                logits = model(token, token != 1).sequence_logits
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = model(batch['token'].to(device))['logits']

            batch_idx = torch.arange(probs.size(0))
            pos = batch['local_pos']
            wts = batch['wt_token'].unsqueeze(dim=0)

            probs = probs[batch_idx, pos, :].unsqueeze(0).cpu()
            perplexity.update(probs, wts)

    return perplexity.compute().item()

model.eval()

fasta = Fasta(snakemake.input['fasta'])
random.seed(43)
proteins = random.sample([
    row['id']
    for row in fasta.fai
    if row['length'] <= 2500
], snakemake.params['num_proteins'])

perplexities = dict()

for protein_id in proteins:
    seq = fasta[protein_id]

    if len(seq) > 1024:
        continue

    perplexities[protein_id] = predict_pseudoperplexity(
        model, seq, batch_size=batch_size, max_len=max_len)

    pd.Series(perplexities).to_csv(
        snakemake.output['perplexity'], sep='\t', header=False)