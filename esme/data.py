import torch
import sklearn
import lightning as L
import sklearn.impute
import sklearn.utils
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from esme.fasta import Fasta
from esme.alphabet import tokenize, pad_tokens, mask_tokens, tokenize_unpad, Alphabet3


class TokenSizeBatchSampler:
    '''
    Batch sampler based on the token size of the sequences in the dataset.

    Args:
        token_sizes (list): List of token sizes for each sequence in the dataset.
        token_per_batch (int): Maximum number of tokens per batch.
        drop_last (bool): If True, drop the last batch if it is smaller than
            `token_per_batch`.
    '''

    def __init__(self, token_sizes, token_per_batch, drop_last=False,
                 shuffle=True, random_state=None):
        self.token_sizes = token_sizes
        self.token_per_batch = token_per_batch
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.random_state = random_state
        self._batches = list(self.batches())

    def batches(self):
        indices = list(range(len(self.token_sizes)))

        if self.shuffle:
            indices = sklearn.utils.shuffle(
                indices, random_state=self.random_state)

        batch = list()
        max_len = 0

        for idx in indices:
            token_len = self.token_sizes[idx] + 2  # add 2 for start, end

            if max_len + token_len > self.token_per_batch:
                yield batch
                max_len = token_len
                batch = [idx]
            else:
                max_len += token_len
                batch.append(idx)

        if len(batch) > 0 and (not self.drop_last):
            yield batch

    def __getitem__(self, idx):
        return self._batches[idx]

    def __len__(self):
        return len(self._batches)


class BaseFastaDataset(Dataset):

    def __init__(self, fasta, fai=None, k_sample=None, max_len=None, alphabet=Alphabet3):
        self.max_len = max_len or float('inf')
        self.fasta = fasta
        self.alphabet = alphabet
        self.fasta = Fasta(fasta, fai=fai, max_len=max_len, k_sample=k_sample)

    def read_seq(self, idx):
        return self.fasta[idx]

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class FastaDataset(BaseFastaDataset):
    '''
    Dataset for reading fasta files into tokenized sequences.

    Args:
        fasta (str): Path to the fasta file.
        fai (str, optional): Path to the fasta index file. If not provided,
            it is assumed to be `fasta + ".fai"`.
        k_sample (int, optional): Number of sequences to sample from the fasta
            file. If None, all sequences are used.
        max_len (int, optional): Maximum length of the sequences to include.
            If None, all sequences are used.
    '''

    def __init__(self, fasta, fai=None, k_sample=None, max_len=None, alphabet=Alphabet3):
        super().__init__(fasta, fai=fai, k_sample=k_sample,
                         max_len=max_len, alphabet=alphabet)

    def __len__(self):
        return len(self.fasta)

    def __getitem__(self, idx):
        return tokenize(self.read_seq(idx), alphabet=self.alphabet)

    @staticmethod
    def collate_fn(batch):
        return pad_tokens(batch)

    def to_dataloader(self, batch_size, shuffle=False, num_workers=0, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=self.collate_fn,
                          **kwargs)


class FastaTokenDataset(BaseFastaDataset):
    '''
    Dataset for reading fasta files into tokenized sequences with a fixed
    number of tokens per batch.

    Args:
        fasta (str): Path to the fasta file.
        fai (str, optional): Path to the fasta index file. If not provided,
            it is assumed to be `fasta + ".fai"`.
        token_per_batch (int): Maximum number of tokens per batch.
        k_sample (int, optional): Number of sequences to sample from the fasta
            file. If None, all sequences are used.
        max_len (int, optional): Maximum length of the sequences to include.
            If None, all sequences are used.
        drop_last (bool): If True, drop the last batch if it is smaller than
            `token_per_batch`.
        shuffle (bool): If True, shuffle the sequences.
        random_state (int): Random seed for shuffling the sequences.
    '''

    def __init__(self, fasta, fai=None, token_per_batch=50_000, k_sample=None,
                 max_len=None, drop_last=False, shuffle=True, random_state=None, alphabet=Alphabet3):
        super().__init__(fasta, fai=fai, k_sample=k_sample, max_len=max_len)
        self.token_per_batch = token_per_batch
        self.alphabet = alphabet

        lengths = [row['length'] for row in self.fasta.fai]
        self.sampler = list(iter(TokenSizeBatchSampler(
            lengths, token_per_batch, drop_last=drop_last,
            shuffle=shuffle, random_state=random_state
        )))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        indices = self.sampler[idx]
        token, _, cu_lens, max_len = tokenize_unpad(
            [self.read_seq(i) for i in indices], alphabet=self.alphabet)
        return token, (cu_lens, max_len)

    def to_dataloader(self, num_workers=0, **kwargs):
        return DataLoader(
            self,
            num_workers=num_workers,
            batch_size=None,  # batch_size is calculated by TokenSizeBatchSampler
            **kwargs
        )


class MaskedFastaDataset(FastaDataset):
    '''
    Dataset for reading fasta files into tokenized sequences with masked tokens.

    Args:
        fasta (str): Path to the fasta file.
        fai (str, optional): Path to the fasta index file. If not provided,
            it is assumed to be `fasta + ".fai"`.
        k_sample (int, optional): Number of sequences to sample from the fasta
            file. If None, all sequences are used.
        max_len (int, optional): Maximum length of the sequences to include.
            If None, all sequences are used.
        mask_freq (float): Frequency of masked tokens.
        alter_freq (float): Frequency of altered
    '''

    def __init__(self, fasta, fai=None, max_len=None, k_sample=None, mask_freq=.15,
                 alter_freq=.1, alphabet=Alphabet3):
        super().__init__(fasta, fai=fai, k_sample=k_sample,
                         max_len=max_len, alphabet=alphabet)
        self.mask_freq = mask_freq
        self.alter_freq = alter_freq

    def __getitem__(self, idx):
        token = super().__getitem__(idx)
        mtokens, mask = mask_tokens(token, self.mask_freq, self.alter_freq,
                                    alphabet=self.alphabet)
        return token, mtokens, mask

    @staticmethod
    def collate_fn(batch):
        tokens = pad_tokens([i[0] for i in batch])
        mtokens = pad_tokens([i[1] for i in batch])
        mask = pad_sequence([i[2][0] for i in batch], batch_first=True,
                            padding_value=False)
        return tokens, mtokens, mask

    def to_dataloader(self, batch_size, shuffle=False, num_workers=0, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=self.collate_fn,
                          **kwargs)


class MaskedFastaTokenDataset(FastaTokenDataset):
    '''
    Dataset for reading fasta files into tokenized sequences with masked tokens
    and a fixed number of tokens per batch.

    Args:
        fasta (str): Path to the fasta file.
        fai (str, optional): Path to the fasta index file. If not provided,
            it is assumed to be `fasta + ".fai"`.
        token_per_batch (int): Maximum number of tokens per batch.
        k_sample (int, optional): Number of sequences to sample from the fasta
            file. If None, all sequences are used.
        max_len (int, optional): Maximum length of the sequences to include.
            If None, all sequences are used.
        mask_freq (float): Frequency of masked tokens.
        alter_freq (float): Frequency of altered tokens and same frequency of
            masked tokens will be replace with original tokens.
        drop_last (bool): If True, drop the last batch if it is smaller than
            `token_per_batch`.
        shuffle (bool): If True, shuffle the sequences.
        random_state (int): Random seed for shuffling the sequences
    '''

    def __init__(self, fasta, fai=None, token_per_batch=50_000, k_sample=None,
                 max_len=None, mask_freq=.15, alter_freq=.1, drop_last=False,
                 shuffle=True, random_state=None, alphabet=Alphabet3):
        super().__init__(fasta, fai=fai, token_per_batch=token_per_batch,
                         k_sample=k_sample, max_len=max_len, drop_last=drop_last,
                         shuffle=shuffle, random_state=random_state, alphabet=alphabet)
        self.mask_freq = mask_freq
        self.alter_freq = alter_freq

    def __getitem__(self, idx):
        tokens, unpad_args = super().__getitem__(idx)
        mtokens, mask = mask_tokens(
            tokens, self.mask_freq, self.alter_freq, alphabet=self.alphabet)
        return tokens, unpad_args, mtokens, mask


class MaskedFastaDataModule(L.LightningDataModule):
    '''
    DataModule for reading fasta files into tokenized sequences with masked tokens.

    Args:
        train_fasta (str): Path to the training fasta file.
        val_fasta (str): Path to the validation fasta file.
        test_fasta (str, optional): Path to the test fasta file.
        train_fai (str, optional): Path to the training fasta index file.
        val_fai (str, optional): Path to the validation fasta index file.
        test_fai (str, optional): Path to the test fasta index file.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the dataloaders.
        mask_freq (float): Frequency of masked tokens.
        alter_freq (float): Frequency of altered
        max_len (int): Maximum length of the sequences to include.
    '''

    def __init__(
        self, train_fasta, val_fasta, test_fasta=None,
        train_fai=None, val_fai=None, test_fai=None,
        batch_size=16, num_workers=0, mask_freq=.15, alter_freq=.1, max_len=None, alphabet=Alphabet3
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fasta = train_fasta
        self.val_fasta = val_fasta
        self.test_fasta = test_fasta

        self.train_fai = train_fai
        self.val_fai = val_fai
        self.test_fai = test_fai

        self.mask_freq = mask_freq
        self.alter_freq = alter_freq
        self.max_len = max_len
        self.alphabet = alphabet

    def _dataloder(self, fasta, fai=None, shuffle=False):
        return MaskedFastaDataset(
            fasta, fai=fai, max_len=self.max_len, mask_freq=self.mask_freq,
            alter_freq=self.alter_freq, alphabet=self.alphabet
        ).to_dataloader(shuffle=shuffle, batch_size=self.batch_size,
                        num_workers=self.num_workers)

    def train_dataloader(self):
        return self._dataloder(self.train_fasta, self.train_fai, shuffle=True)

    def val_dataloader(self):
        return self._dataloder(self.val_fasta, self.val_fai)

    def test_dataloader(self):
        return self._dataloder(self.test_fasta, self.test_fai)


class MaskedFastaTokenDataModule(L.LightningDataModule):
    '''
    DataModule for reading fasta files into tokenized sequences with masked tokens
    and a fixed number of tokens per batch.

    Args:
        train_fasta (str): Path to the training fasta file.
        val_fasta (str): Path to the validation fasta file.
        test_fasta (str, optional): Path to the test fasta file.
        train_fai (str, optional): Path to the training fasta index file.
        val_fai (str, optional): Path to the validation fasta index file.
        test_fai (str, optional): Path to the test fasta index file.
        token_per_batch (int): Maximum number of tokens per batch.
        num_workers (int): Number of workers for the dataloaders.
        mask_freq (float): Frequency of masked tokens.
        alter_freq (float): Frequency of altered tokens and same frequency of
            masked tokens will be replace with original tokens.
        max_len (int): Maximum length of the sequences to include.
    '''

    def __init__(
        self, train_fasta, val_fasta, test_fasta=None,
        train_fai=None, val_fai=None, test_fai=None,
        token_per_batch=100_000, num_workers=0,
        mask_freq=.15, alter_freq=.1, max_len=None, alphabet=Alphabet3
    ):
        super().__init__()
        self.token_per_batch = token_per_batch
        self.num_workers = num_workers
        self.train_fasta = train_fasta
        self.val_fasta = val_fasta
        self.test_fasta = test_fasta

        self.train_fai = train_fai
        self.val_fai = val_fai
        self.test_fai = test_fai

        self.mask_freq = mask_freq
        self.alter_freq = alter_freq
        self.max_len = max_len

        self.alphabet = alphabet
        self.current_epoch = 0

    def _dataloder(self, fasta, fai=None, shuffle=False):
        return MaskedFastaTokenDataset(
            fasta, fai=fai, token_per_batch=self.token_per_batch, max_len=self.max_len,
            mask_freq=self.mask_freq, alter_freq=self.alter_freq,
            shuffle=shuffle, random_state=self.current_epoch, alphabet=self.alphabet
        ).to_dataloader(num_workers=self.num_workers)

    def train_dataloader(self):
        return self._dataloder(self.train_fasta, self.train_fai, shuffle=True)

    def val_dataloader(self):
        return self._dataloder(self.val_fasta, self.val_fai)

    def test_dataloader(self):
        return self._dataloder(self.test_fasta, self.test_fai)

    def set_epoch(self, epoch):
        self.current_epoch = epoch


class SetEpochCallback(L.pytorch.callbacks.Callback):
    '''
    Callback to set the current epoch in the datamodule. This is useful when
    the datamodule needs to shuffle the sequences at the beginning of each epoch.
    '''

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.set_epoch(trainer.current_epoch)


class LabeledDataset(Dataset):

    def __init__(self, seqs, labels, token_per_batch, shuffle=True, 
                 random_state=None, truncate_len=None):
        self.seqs = seqs
        self.labels = labels
        self.truncate_len = truncate_len

        self.sampler = list(iter(TokenSizeBatchSampler(
            [len(seq) for seq in seqs], token_per_batch,
            shuffle=shuffle, random_state=random_state)))

    def truncate(self, seq):
        if (self.truncate_len is not None) and (len(seq) > self.truncate_len):
            return seq[:self.truncate_len]
        return seq

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        indices = self.sampler[idx]
        tokens, _indices, cu_lens, max_len = tokenize_unpad(
            [self.truncate(self.seqs[i]) for i in indices])
        return {
            'token': tokens,
            'cu_lens': cu_lens,
            'max_len': max_len,
            'indices': _indices,
            'label': torch.Tensor([self.labels[i] for i in indices])
        }


class LabeledDataModule(L.LightningDataModule):

    def __init__(self, train_seqs, train_labels, val_seqs, val_labels,
                 test_seqs=None, test_labels=None, token_per_batch=None,
                 truncate_len=None, num_workers=0):
        super().__init__()
        self.train_seqs = train_seqs
        self.train_labels = train_labels
        self.val_seqs = val_seqs
        self.val_labels = val_labels
        self.test_seqs = test_seqs
        self.test_labels = test_labels

        self.token_per_batch = token_per_batch
        self.truncate_len = truncate_len
        self.num_workers = num_workers
        self.current_epoch = None

    def _dataloder(self, seqs, labels, shuffle=True, **kwargs):
        return DataLoader(
            dataset=LabeledDataset(
                seqs,
                labels,
                token_per_batch=self.token_per_batch,
                truncate_len=self.truncate_len,
                shuffle=shuffle,
                random_state=self.current_epoch
            ),
            num_workers=self.num_workers,
            batch_size=None,
            **kwargs
        )

    def train_dataloader(self):
        return self._dataloder(self.train_seqs, self.train_labels, shuffle=True)

    def val_dataloader(self):
        return self._dataloder(self.val_seqs, self.val_labels, shuffle=False)

    def test_dataloader(self):
        return self._dataloder(self.test_seqs, self.test_labels, shuffle=False)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
