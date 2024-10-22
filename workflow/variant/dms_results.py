import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import spearmanr
from esme.variant import predict_mask_margin

device = snakemake.params['device']
batch_size = 16
quantization = snakemake.wildcards['quantize']
quantization = None if quantization == 'none' else quantization
efficient = 'e' in snakemake.wildcards['model']

if efficient:
    if snakemake.wildcards['model'] == '1be':
        from esme import ESM1b
        model = ESM1b.from_pretrained(snakemake.input['model'],
                                      quantization=quantization, device=device)
    elif snakemake.wildcards['model'].startswith('1ve'):
        from esme import ESM1v
        model = ESM1v.from_pretrained(snakemake.input['model'],
                                      quantization=quantization, device=device)
    else:
        from esme import ESM2
        model = ESM2.from_pretrained(snakemake.input['model'],
                                     quantization=quantization, device=device)
    max_len = None
else:
    assert quantization is None
    import esm
    model, _ = esm.pretrained.load_model_and_alphabet(snakemake.input['model'])
    model = model.to(device)
    model.predict_log_prob = lambda token: torch.log_softmax(
        model(token)['logits'], dim=-1)

    if snakemake.wildcards['model'] == '15B':
        max_len = 1024
    elif snakemake.wildcards['model'] == '1b':
        max_len = 1024
    elif snakemake.wildcards['model'].startswith('1v'):
        max_len = 1024
    else:
        max_len = None

mapping_study = pd.read_csv(snakemake.input['mapping']) \
    .set_index('DMS_id')['Uniprot_ID'].to_dict()

dms_dir = Path(snakemake.input['dms_substitutions_dir'])
proteins = snakemake.params['proteins']


def read_dms(dms_dir, proteins, mapping):

    print('Reading DMS data')
    ground_truth = {
        dms_path.stem: pd.read_csv(dms_path)
        .rename(columns={'mutant': 'variant_id'})
        .set_index('variant_id')

        for dms_path in tqdm(dms_dir.iterdir())
        if dms_path.suffix == '.csv'
    }
    # TODO: missing in uniprot-swiss uniprotdb
    for study_id in list(ground_truth):
        if study_id not in mapping:
            warnings.warn(f'TOFIX: add mapping for {study_id}')
            del ground_truth[study_id]
            continue

        if mapping[study_id] not in proteins:
            del ground_truth[study_id]

    return ground_truth


def predict_variant(df_dms):
    df_dms = df_dms[~df_dms.index.str.contains(':')]
    row = df_dms.iloc[0]
    dms_seq = row.mutated_sequence
    pos = int(row.name[1:-1])
    dms_seq = dms_seq[:pos - 1] + row.name[0] + dms_seq[pos:]
    return predict_mask_margin(model, dms_seq, batch_size, max_len=max_len)


ground_truth = read_dms(dms_dir, proteins, mapping_study)

df_corr = list()

scatter_dir = Path(snakemake.output['scatter_dir'])
scatter_dir.mkdir(exist_ok=True, parents=True)


for i, study_id in enumerate(ground_truth):

    print(f'{i+1}/{len(ground_truth)} {study_id}')
    protein_id = mapping_study[study_id]

    df = ground_truth[study_id]
    scores = predict_variant(df).to_dict()['score']

    df['score'] = df.index.map(
        lambda x: sum(scores.get(i, np.nan) for i in x.split(':'))
    ).tolist()

    missing = df['score'].isna().sum()
    if missing > 0:
        warnings.warn(f'{protein_id} {missing} missing variants')
        continue

    df = df[~df['score'].isna()]

    corr, pval = spearmanr(df['DMS_score'], df['score'])

    plt.figure(figsize=(5, 5), dpi=300)
    plt.title(f'{study_id} 'r'$\rho$'f'={corr:.2f}')
    alpha = 0.1 if df.shape[0] < 10_000 else 0.01
    sns.regplot(data=df, x='score', y='DMS_score',
                scatter_kws={'alpha': alpha}, line_kws={'color': 'black'})
    plt.xlabel('ESM')
    sns.despine()
    plt.savefig(scatter_dir / f'{study_id}.png')
    plt.close()

    df_corr.append({
        'study_id': study_id,
        'protein_id': protein_id,
        'correlation': corr,
        'pval': pval
    })

df_corr = pd.DataFrame(df_corr)
df_corr.to_csv(snakemake.output['stats'], index=False)
