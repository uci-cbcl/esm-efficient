import pandas as pd
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from esme import ESM
from esme.alphabet import tokenize_unpad
from esme.data import TokenSizeBatchSampler, LabeledDataset


# class MeltomeDataset(Dataset):

#     def __init__(self, seqs, labels, token_per_batch, shuffle=True, random_state=None):
#         self.seqs = seqs
#         self.labels = labels

#         self.sampler = list(iter(TokenSizeBatchSampler(
#             [len(seq) for seq in seqs], token_per_batch,
#             shuffle=shuffle, random_state=random_state)))

#     def __len__(self):
#         return len(self.sampler)

#     def __getitem__(self, idx):
#         indices = self.sampler[idx]
#         tokens, _indices, cu_lens, max_len = tokenize_unpad(
#             [self.seqs[i] for i in indices])
#         return {
#             'token': tokens,
#             'cu_lens': cu_lens,
#             'max_len': max_len,
#             'indices': _indices,
#             'label': torch.tensor([self.labels[i] for i in indices], dtype=torch.bfloat16)
#         }


class MeltomeDataModule(L.LightningDataModule):

    def __init__(self, path, token_per_batch=None, num_workers=0, truncate_len=None):
        super().__init__()
        df = pd.read_csv(path)
        self.df_test = df.query('set == "test"')

        df = df.query('set == "train"')
        trainset = df['validation'].isna()
        self.df_train = df[trainset]
        self.df_val = df[~trainset]

        self.token_per_batch = token_per_batch
        self.num_workers = num_workers
        self.truncate_len = truncate_len
        self.current_epoch = None

    def _dataloder(self, df, shuffle=True, **kwargs):
        return DataLoader(
            dataset=LabeledDataset(
                df['sequence'].values.tolist(),
                df['target'].values.tolist(),
                token_per_batch=self.token_per_batch,
                shuffle=shuffle,
                random_state=self.current_epoch,
                truncate_len=self.truncate_len,
            ),
            num_workers=self.num_workers,
            batch_size=None,
            **kwargs
        )

    def train_dataloader(self):
        return self._dataloder(self.df_train, shuffle=True)

    def test_dataloader(self):
        return self._dataloder(self.df_val, shuffle=False)

    def val_dataloader(self):
        return self._dataloder(self.df_test, shuffle=False)

    def set_epoch(self, epoch):
        self.current_epoch = epoch


class MeltomeModel(L.LightningModule):

    def __init__(self, model_path, checkpointing=True, lr=1e-5, lr_head=1e-4,
                 lora_kwargs=None, device=0):
        super().__init__()
        self.lr = lr
        self.lr_head = lr_head

        self.plm = ESM.from_pretrained(model_path, checkpointing=checkpointing, 
                                       device=device)

        if lora_kwargs is not None:
            self.plm.add_lora(**lora_kwargs)

        self.head = nn.Sequential(
            nn.Linear(self.plm.embed_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        ).to(torch.bfloat16).to(device)

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.val_spearman = torchmetrics.SpearmanCorrCoef()

    def forward(self, token, pad_args, indices):
        embed = self.plm.forward_representation(
            token, pad_args, pad_output=True, pad_indices=indices)
        return self.head(embed.mean(dim=1)).squeeze(1) * 100

    def _loss(self, batch):
        y = self(batch['token'], (batch['cu_lens'],
                 batch['max_len']), batch['indices'])
        return F.mse_loss(y, batch['label'], reduction='sum'), y

    def training_step(self, batch, batch_idx):
        loss, _ = self._loss(batch)
        self.train_loss.update(loss.detach().clone())
        self.log('train_loss', self.train_loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y = self._loss(batch)
        self.val_loss.update(loss.detach().clone())
        self.val_spearman.update(y.detach().clone(), batch['label'])
        self.log('val_loss', self.val_loss, on_epoch=True)
        self.log('val_spearman_corr', self.val_spearman, on_epoch=True)

    def configure_optimizers(self):
        adam = torch.optim.Adam([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': self.plm.parameters()}
        ], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            adam, gamma=.9, step_size=10)
        return {
            "optimizer": adam,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }
