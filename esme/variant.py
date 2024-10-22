import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.text import Perplexity
from esme import ESM2
from esme.alphabet import tokenize, mask_idx, token_to_idx, amino_acids


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

    def __init__(self, seq, max_len=None):
        super().__init__()
        self.seq = seq
        self.max_len = max_len
        self.token = tokenize([seq])[0]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        token = self.token.clone()
        wt = self.seq[idx]
        idx += 1
        token[idx] = mask_idx

        if self.max_len is not None and token.size(0) > self.max_len:
            start = max(0, idx - self.max_len // 2)
            start = min(token.size(0) - self.max_len, start)
            end = min(token.size(0), start + self.max_len)
            token = token[start:end]
            pos = idx - start
        else:
            pos = idx

        return {
            'token': token,
            'local_pos': pos,
            'pos': idx,
            'wt': wt,
            'wt_token': token_to_idx[wt],
        }


class PseudoPerplexitiesMarginDataset(Dataset):

    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        self.token = tokenize([seq])[0]

        self.variants = list()

        for pos, wt in enumerate(seq):
            for mt in amino_acids:
                for mask_pos, mask_aa in enumerate(seq):
                    self.variants.append(
                        (wt, pos + 1, mt, mask_pos + 1, mask_aa)
                    )

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, idx):
        wt, pos, mt, mask_pos, mask_aa = self.variants[idx]

        token = self.token.clone()
        token[pos] = token_to_idx[mt]
        token[mask_pos] = mask_idx

        return {
            'token': token,
            'wt': wt,
            'pos': pos,
            'mt': mt,
            'wt_mask_idx': token_to_idx[mask_aa],
            'mask_pos': mask_pos,
        }


def predict_mask_margin(model: ESM2, seq: str, batch_size: int = 32, max_len=None):
    """Predicts the mask margin for a given sequence.

    Args:
        model (ESM): Model object to use for prediction.
        seq (str): Protein sequence or DataLoader of MaskMarginDataset
        batch_size (int, optional): batch size for prediction. (default: 32)
        max_len (_type_, optional): 

    Returns:
        pd.DataFrame: DataFrame with 'variant' as index and columns of 'score'.

    Example:
    >>> model = ESM2.from_pretrained('8M.safetensors')
    >>> predict_mask_margin(model, 'MPEAAPPVAPAPAAP', batch_size=32)
    ... pd.DataFrame({
    ...    'variant': ['M1A', 'M1C', ..., 'P16Y'],
    ...    'score': [0.1, 0.2, ..., -0.3]
    ... }).set_index('variant')
    """
    device = next(model.parameters()).device

    if isinstance(seq, str):
        dl = DataLoader(MaskMarginDataset(seq, max_len=max_len),
                        batch_size=batch_size, shuffle=False)
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
            probs = model.predict_log_prob(
                batch['token'].to(device), pad_output=True)

            batch_idx = torch.arange(probs.size(0))
            pos = batch['local_pos']
            wts = batch['wt_token']

            probs = probs[batch_idx, pos, :]
            mask_margin = probs - probs[batch_idx, wts].unsqueeze(1)

            pos = batch['pos']
            for p, wt, rl in zip(pos, batch['wt'], mask_margin):
                for aa in amino_acids:
                    df.append({
                        'variant': f'{wt}{p}{aa}',
                        'score': rl[token_to_idx[aa]].item()
                    })
    return pd.DataFrame(df).set_index('variant')


def predict_pseudoperplexity(model: ESM2, seq: str, batch_size=32, max_len=None):
    '''
    Predict the pseudo-perplexity of sequence.

    Args:
        model (ESM2): Model object to use for prediction.
        seq (str): Protein sequence.
        batch_size (int, optional): batch size for prediction. (default: 32)

    Returns:
        int: Pseudo-perplexity of the sequence.

    Example:
    >>> model = ESM2.from_pretrained('8M.safetensors')
    >>> predict_pseudoperplexity(model, 'MPEAAPPVAPAPAAP', batch_size=32)
    ... 10
    '''
    device = next(model.parameters()).device

    if isinstance(seq, str):
        dl = DataLoader(MaskMarginDataset(seq, max_len=max_len),
                        batch_size=batch_size, shuffle=False)
    elif isinstance(seq, DataLoader):
        dl = seq
    elif isinstance(seq, MaskMarginDataset):
        dl = DataLoader(seq, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError('seq must be str or DataLoader')

    perplexity = Perplexity()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            if isinstance(model, ESM2):
                # TODO: fix this
                probs = model(batch['token'].to(device), pad_output=True)
            else:
                probs = model(batch['token'].to(device))['logits']

            batch_idx = torch.arange(probs.size(0))
            pos = batch['local_pos']
            wts = batch['wt_token'].unsqueeze(dim=0)

            probs = probs[batch_idx, pos, :].unsqueeze(0).cpu()
            perplexity.update(probs, wts)

    return perplexity.compute().item()


def predict_pseudoperplexity_margin(model: ESM2, seq: str, batch_size):

    device = next(model.parameters()).device

    dl = DataLoader(PseudoPerplexitiesMarginDataset(seq),
                    batch_size=batch_size, shuffle=False)

    df = list()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            probs = model.predict_log_prob(batch['token'].to(device))
            probs = probs[
                torch.arange(probs.size(0)),
                batch['mask_pos'],
                batch['wt_mask_idx'],
            ]

            df.append(pd.DataFrame({
                'variant': [
                    f'{wt}{p}{aa}'
                    for wt, p, aa in zip(batch['wt'], batch['pos'], batch['mt'])
                ],
                'score': probs.cpu().half().numpy()
            }))

    return pd.concat(df).groupby('variant').mean()
