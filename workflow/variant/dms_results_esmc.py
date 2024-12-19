import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from esme.alphabet import Alphabet3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.models.esmc import ESMC


device = 3
batch_size = 32
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
token_to_id = {tok: ind for ind, tok in enumerate(SEQUENCE_VOCAB)}
mask_idx = token_to_id['<mask>']
cls_idx = token_to_id['<cls>']
eos_idx = token_to_id['<eos>']
unk_idx = token_to_id['<unk>']

tokenizer = EsmSequenceTokenizer()

if snakemake.wildcards['model'] == 'c300m':
    model = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=tokenizer,
    )
elif snakemake.wildcards['model'] == 'c600m':
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer
    )
else:
    raise ValueError('Invalid model')

state_dict = torch.load(
    snakemake.input['model'],
    weights_only=True
)
model.load_state_dict(state_dict)
model = model.to(device).to(torch.bfloat16)
model.eval()


class MaskMarginDataset(Dataset):
    '''
    Dataset for predicting the mask margin of a given sequence.

    Args:
        seq (str): Protein sequence.
        max_len (int, optional): Maximum length of the sequence. (default: None)

    Returns:
        Dict[str, Tensor]: Dictionary with keys of 'token', 'local_pos', 'pos', 'wt', 'wt_token'.
            if max_len is not None, 'token' will be truncated to the maximum length and variant 
            will centered. 'local_pos' is the position of the variant in the truncated sequence.
            if max_len is None, 'token' will be the full sequence and 'local_pos' will be the same
            as 'pos'. 'wt' is the wild-type amino acid at the variant position. 'wt_token' is the
            token index of the wild-type amino acid. 'pos' is the position of the variant in the
            full sequence.

    Example:
    >>> ds = MaskMarginDataset('MPEAAPPVAPAPAAP')
    >>> len(ds)
    ... 16
    >>> ds[0]
    ... {'token': tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
    ...  'local_pos': 1,
    ...  'pos': 0,
    ...  'wt': 'M',
    ...  'wt_token': 1}
    '''

    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        self.token = torch.tensor([
            cls_idx,
            *(token_to_id.get(aa, unk_idx) for aa in seq),
            eos_idx
        ], dtype=torch.int64)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        token = self.token.clone()
        wt = self.seq[idx]
        idx += 1
        token[idx] = mask_idx

        return {
            'token': token,
            'pos': idx,
            'wt': wt,
            'wt_token': token_to_id[wt],
        }


def predict_mask_margin(model, seq: str, batch_size: int = 32):
    device = next(model.parameters()).device

    if isinstance(seq, str):
        dl = DataLoader(MaskMarginDataset(
            seq), batch_size=batch_size, shuffle=False)
    elif isinstance(seq, DataLoader):
        dl = seq
    elif isinstance(seq, MaskMarginDataset):
        dl = DataLoader(seq, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError('seq must be str or DataLoader')

    df = list()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            tokens = batch['token'].to(device)
            sequence_id = tokens == token_to_id['<pad>']
            logit = model(tokens, sequence_id=sequence_id).sequence_logits
            probs = torch.log_softmax(logit, dim=-1)

            batch_idx = torch.arange(probs.size(0))
            pos = batch['pos']
            wts = batch['wt_token']

            probs = probs[batch_idx, pos, :]
            mask_margin = probs - probs[batch_idx, wts].unsqueeze(1)

            pos = batch['pos']
            for p, wt, rl in zip(pos, batch['wt'], mask_margin):
                for aa in Alphabet3.amino_acids:
                    df.append({
                        'variant': f'{wt}{p}{aa}',
                        'score': rl[token_to_id[aa]].item()
                    })
    return pd.DataFrame(df).set_index('variant')


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
    return predict_mask_margin(model, dms_seq, batch_size)


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
