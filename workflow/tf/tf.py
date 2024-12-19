from tqdm import tqdm
import random
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
import torchmetrics
from esme import ESM2
from esme.alphabet import tokenize, pad_tokens, Alphabet


padding_idx = Alphabet.padding_idx


class TfDataset(Dataset):

    def __init__(self, tfs, non_tfs, random_seed=41, balance=None):
        assert balance in [None, 'upsample', 'downsample']
        self.tfs = tfs
        self.non_tfs = non_tfs
        self.balance = balance

        if balance == 'upsample':
            imbalance = len(non_tfs) // len(tfs)
            self.tfs = tfs * imbalance
        elif balance == 'downsample':
            random.seed(random_seed)

    def __len__(self):
        if self.balance == 'downsample':
            return 2 * len(self.tfs)
        else:
            return len(self.tfs) + len(self.non_tfs)

    def __getitem__(self, idx):
        if idx < len(self.tfs):
            protein = self.tfs[idx]
            label = 1.
        else:
            if self.balance == 'downsample':
                protein = random.choice(self.non_tfs)
            else:
                protein = self.non_tfs[idx - len(self.tfs)]
            label = 0.

        return {
            'token': protein,
            'label': torch.tensor([label])
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'token': pad_tokens([b['token'] for b in batch]),
            'label': default_collate([b['label'] for b in batch]),
        }


class TfDataModule:

    def __init__(self, train, val, test, batch_size=32, num_workers=16, balance=None):
        super().__init__()

        self.df_train = pd.read_parquet(train)
        self.df_val = pd.read_parquet(val)
        self.df_test = pd.read_parquet(test)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance

    def _dataloder(self, df, shuffle=False, balance=None):
        tfs = [
            tokenize([row.seq])[0]
            for row in tqdm(df.query('tf').itertuples(), desc='Loading TFs')
        ]
        non_tfs = [
            tokenize([row.seq])[0]
            for row in tqdm(df.query('~tf').itertuples(), desc='Loading non-TFs')
        ]
        return DataLoader(
            TfDataset(tfs, non_tfs, balance=balance),
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=TfDataset.collate_fn, shuffle=shuffle)

    def train_dataloader(self):
        return self._dataloder(self.df_train, shuffle=True, balance=self.balance)

    def val_dataloader(self):
        return self._dataloder(self.df_val)

    def test_dataloader(self):
        return self._dataloder(self.df_test)


class TfModel(nn.Module):

    def __init__(self, model_path, checkpointing=True, quantization=None,
                 lora_kwargs=None, device=0):
        super().__init__()

        self.plm = ESM2.from_pretrained(
            model_path, checkpointing=checkpointing,
            quantization=quantization, device=device)

        if lora_kwargs is not None:
            self.plm.add_lora(**lora_kwargs)

        self.head = nn.Sequential(
            nn.Linear(self.plm.embed_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        ).to(torch.bfloat16).to(device)

    def forward(self, token):
        emb = self.plm.forward_representation(token)
        emb = emb.masked_fill((token == padding_idx).unsqueeze(-1), 0)
        return self.head(emb.mean(dim=1))
