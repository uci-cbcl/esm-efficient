from typing import List, Union, Tuple
import re
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


alphabet = [
    '<cls>', '<pad>', '<eos>', '<unk>',
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
    'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O',
    '.', '-', '<null_1>', '<mask>'
]
amino_acids = alphabet[4:24]

idx_to_token = dict(enumerate(alphabet))
token_to_idx = {token: idx for idx, token in idx_to_token.items()}

cls_idx = token_to_idx['<cls>']
eos_idx = token_to_idx['<eos>']
padding_idx = token_to_idx['<pad>']
mask_idx = token_to_idx['<mask>']
unk_idx = token_to_idx['<unk>']
amino_acids_idx = [token_to_idx[aa] for aa in amino_acids]


def split_alphabet(seq: Union[str, List[str]]) -> List[List[str]]:
    '''
    Split a sequence into a list of token in alphabet.

    Args:
      seq: A string representing a sequence.

    Returns:
      List[str]: A list of token in alphabet.

    Example:
    >>> split_alphabet('MPEAAPPV<mask>APAPAAP')
    ['M', 'P', 'E', 'A', 'A', 'P', 'P', 'V', '<mask>', 'A', 'P', 'A', 'P', 'A', 'A', 'P']
    '''
    pattern = re.compile(r"<[^>]+>|.")

    if isinstance(seq, str):
        return re.findall(pattern, seq)
    else:
        return [re.findall(pattern, s) for s in seq]


def token_to_str(tokens: Tensor) -> List[str]:
    '''
    Convert a tensor of token indices to a list of sequences.

    Args:
      tokens: A tensor of token indices.

    Returns:
      List[str]: A list of sequences.
    '''
    return [''.join(idx_to_token[i] for i in seq) for seq in tokens.tolist()]


def tokenize(sequences: Union[List[str], str]) -> Tensor:
    '''
    Convert a list of sequences to a list of token indices.

    Args:
      sequences: A list of sequences, where each sequence is either a list of 
        tokens or a string.

    Returns:
      Tensor[int]: A tensor of token indices.

    '''
    if isinstance(sequences, str):
        sequences = [sequences]

    sequences = split_alphabet(sequences)
    max_len = max(len(seq) for seq in sequences) + 2

    tokens = torch.ones(len(sequences), max_len,
                        dtype=torch.int64) * padding_idx

    for i, seq in enumerate(sequences):
        tokens[i, :len(seq) + 2] = torch.tensor([
            cls_idx,
            *(token_to_idx.get(aa, unk_idx) for aa in seq),
            eos_idx
        ], dtype=torch.int64)

    return tokens


def tokenize_unpad(sequences: Union[List[str], str]) -> Tuple[Tensor, Tensor, Tensor, int]:
    '''
    Convert a list of sequences to a list of token indices without padding.

    Args:
        sequences: A list of sequences, where each sequence is either a list of 
            tokens or a string.

    Returns:
        Tensor[int]: A tensor of tokens.
        Tensor[int]: A tensor of indices.
        Tensor[int]: A tensor of cumulative lengths.
        int: The maximum length.
    '''
    if isinstance(sequences, str):
        sequences = [sequences]

    sequences = split_alphabet(sequences)
    lens = [len(seq) + 2 for seq in sequences]
    cu_lens = torch.tensor(np.cumsum([0] + lens), dtype=torch.int32)
    max_len = max(lens)

    tokens = torch.zeros(cu_lens[-1], dtype=torch.int64)
    indices  = torch.cat([
        torch.arange(i * max_len, i * max_len + l, dtype=torch.int64)
        for i, l in enumerate(lens)
    ])
    
    for cu_low, cu_high, seq in zip(cu_lens[:-1], cu_lens[1:], sequences):
        tokens[cu_low:cu_high] = torch.tensor([
            cls_idx,
            *(token_to_idx.get(aa, unk_idx) for aa in seq),
            eos_idx
        ], dtype=torch.int64)

    return tokens, indices, cu_lens, max_len


def pad_tokens(tokens: List[torch.Tensor]):
    '''
    Pad a list of tokens to the same length by adding padding tokens.
    
    Args:
        tokens (List[Tensor]): A list of tokens.
        
    Returns:
        Tensor: A tensor of padded tokens.
        
    Example:
        >>> pad_tokens([torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 5])])
        ... tensor([[1, 2, 3, 1, 1],
                    [1, 2, 3, 4, 5]])
    '''
    if len(tokens[0].shape) == 1:
        max_size = np.max([i.size(0) for i in tokens])
        return torch.stack([
            F.pad(i, (0, max_size - i.size(0)), value=padding_idx)
            for i in tokens
        ])

    max_size = np.max([i.size(1) for i in tokens])
    return torch.cat([
        F.pad(i, (0, max_size - i.size(1)), value=1)
        for i in tokens
    ], dim=0)


def mask_tokens(token: torch.Tensor, freq: float = 0.15, alter: float = 0.1) -> torch.Tensor:
    '''
    Mask tokens randomly tokens for bert training. 80% of the masked tokens are
    replaced with the mask token, 10% of the masked tokens are replaced with
    random tokens, and 10% of the masked tokens are replaced with the same token.

    Args:
        tokens (Tensor): A tensor of token indices.
        freq (float): The frequency of masking tokens.
        alter (float): The frequency of altering tokens and replacing them with
            random. Another same percentage of tokens are replaced with the original
            token.

    Returns:
        Tensor: A tensor of masked tokens.
    '''
    token = token.clone()
    valid = (token != cls_idx) & (token != eos_idx) & (token != padding_idx)
    mask = (torch.rand_like(token, dtype=torch.float32) < freq) & valid

    # at least one mask per row
    not_masked = mask.sum(axis=-1) == 0
    if not_masked.any():
        mask_pos = torch.multinomial(valid[not_masked].float(), 1).squeeze(1)
        if token.ndim == 1:
            mask[mask_pos] = True
        elif token.ndim == 2:
            mask[not_masked, mask_pos] = True
        else:
            raise ValueError('tokens must be 1D or 2D')

    _token = token.clone()
    # 80% of the masked tokens are replaced with the mask token
    token[mask] = mask_idx

    # 10% of the masked tokens are replaced with random tokens
    token = torch.where(
        (torch.rand_like(token, dtype=torch.float32) < alter) & mask,
        torch.randint_like(token, amino_acids_idx[0], amino_acids_idx[-1] + 1),
        token
    )
    # 10% of the masked tokens are replaced with the same token
    token = torch.where(
        (torch.rand_like(token, dtype=torch.float32) < alter) & mask,
        _token, token
    )

    return token, mask
