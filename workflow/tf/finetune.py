from pathlib import Path
import os
import random
import pandas as pd
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import safetensors.torch as safetensors
from esme import ESM2
from esme.alphabet import tokenize, padding_idx
from esme.data import TokenSizeBatchSampler


class TfDataset(Dataset):

    def __init__(self, tfs, non_tfs, random_seed=41, balance=None, token_per_batch=100_000):
        assert balance in [None, 'upsample', 'downsample']
        self.tfs = tfs
        self.non_tfs = non_tfs
        self.balance = balance

        if balance == 'upsample':
            imbalance = len(non_tfs) // len(tfs)
            _tfs = tfs * imbalance
            _non_tfs = non_tfs
        elif balance == 'downsample':
            random.seed(random_seed)
            _tfs = tfs
            _non_tfs = random.sample(non_tfs, len(tfs))
        else:
            _tfs = tfs
            _non_tfs = non_tfs

        self.seqs = _tfs + _non_tfs
        self.labels = [1] * len(_tfs) + [0] * len(_non_tfs)

        self.sampler = TokenSizeBatchSampler(
            [len(seq) for seq in self.seqs], token_per_batch)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        indices = self.sampler[idx]
        return {
            'token': tokenize([self.seqs[i] for i in indices]),
            'label': torch.tensor([self.labels[i] for i in indices], dtype=torch.bfloat16)
        }


class TfDataModule(L.LightningDataModule):

    def __init__(self, train, val, test, token_per_batch, num_workers=16, balance=None):
        super().__init__()

        self.df_train = pd.read_parquet(train)
        self.df_val = pd.read_parquet(val)
        self.df_test = pd.read_parquet(test)

        self.token_per_batch = token_per_batch
        self.num_workers = num_workers
        self.balance = balance

    def _dataloder(self, df, balance=None):
        tfs = df.query('tf').seq.tolist()
        non_tfs = df.query('~tf').seq.tolist()
        return DataLoader(
            TfDataset(tfs, non_tfs, balance=balance,
                      token_per_batch=self.token_per_batch),
            num_workers=self.num_workers,
            batch_size=None,
        )

    def train_dataloader(self):
        return self._dataloder(self.df_train, balance=self.balance)

    def val_dataloader(self):
        return self._dataloder(self.df_val)

    def test_dataloader(self):
        return self._dataloder(self.df_test)


class TfModel(L.LightningModule):

    def __init__(self, model_path, checkpointing=True, lr=1e-4,
                 lora_kwargs=None, device=0):
        super().__init__()
        self.lr = lr

        self.plm = ESM2.from_pretrained(
            model_path, checkpointing=checkpointing, device=device)

        if lora_kwargs is not None:
            self.plm.add_lora(**lora_kwargs)

        self.head = nn.Sequential(
            nn.Linear(self.plm.embed_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        ).to(torch.bfloat16).to(device)

        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, token):
        emb = self.plm.forward_representation(token, pad_output=True)
        return self.head(emb.mean(dim=1)).squeeze(-1)

    def _loss(self, batch):
        return F.binary_cross_entropy_with_logits(
            self(batch['token']), batch['label'])

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.val_loss.update(loss)
        self.log('val_loss', self.val_loss)

    def configure_optimizers(self):
        adam = torch.optim.Adam([*self.parameters()], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=.99)
        return {
            "optimizer": adam,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }


devices = [0]
torch.cuda.set_device(devices[0])
torch.set_float32_matmul_precision('medium')

lora_kwargs = None

wld_lora = snakemake.wildcards['lora']
if wld_lora != 'none':
    rank, layers = wld_lora.split(';')
    layers = layers.split(',')
    lora_kwargs = dict(rank=int(rank), layers=layers, adapter_names='tf')

lr = 1e-3
cls_model = TfModel

checkpoint_dir = Path(snakemake.params['checkpoint_dir'])
if checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())) > 0:
    last_checkpoint = max(checkpoint_dir.iterdir(), key=os.path.getctime)

    model = cls_model.load_from_checkpoint(
        checkpoint_path=last_checkpoint, model_path=snakemake.input['model'],
        lora_kwargs=lora_kwargs, device=devices[0])
else:
    last_checkpoint = None
    model = cls_model(snakemake.input['model'], lr=lr,
                      lora_kwargs=lora_kwargs, device=devices[0])

model = model.to(devices[0])

if wld_lora == 'none':
    for p in model.plm.parameters():
        p.requires_grad = False

class_balance = 'upsample'
datamodule = TfDataModule(
    snakemake.input['train'],
    snakemake.input['val'],
    snakemake.input['test'],
    token_per_batch=50_000,
    num_workers=snakemake.threads // len(devices),
    balance=class_balance
)

checkpoint_callback = ModelCheckpoint(
    dirpath=snakemake.params['checkpoint_dir'],
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

wandb_logger = WandbLogger(project='esm2e-tf')
wandb_logger.log_hyperparams(
    {**snakemake.params, 'lr': lr, 'balance': class_balance})

trainer = L.Trainer(
    devices=devices,
    max_epochs=snakemake.params['epochs'],
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            min_delta=1e-4,
        ),
        checkpoint_callback
    ],
    logger=wandb_logger,
    accumulate_grad_batches=16,
    precision="bf16-mixed",
    reload_dataloaders_every_n_epochs=1,
    log_every_n_steps=1,
    val_check_interval=0.1,
)
trainer.fit(model=model, datamodule=datamodule, ckpt_path=last_checkpoint)

best_model = cls_model.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    map_location=f'cuda:{devices[0]}',
    model_path=snakemake.input['model'],
    lora_kwargs=lora_kwargs, device=devices[0]
)

if wld_lora != 'none':
    best_model.plm.save_lora(snakemake.output['lora_weights'])
else:
    Path(snakemake.output['lora_weights']).touch()

safetensors.save_file(best_model.head.state_dict(),
                      snakemake.output['head_weights'])
