import os
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import safetensors.torch as safetensors
from esme.data import SetEpochCallback
from esme import ESM
from esme.pooling import BinaryLearnedAggregation
from esme.trainer import RegressionTrainer
from workflow.gb1_aav.aav import AavDataModule


torch.set_float32_matmul_precision('medium')

truncate_len = None
if snakemake.wildcards['model'].startswith('1ve') or snakemake.wildcards['model'].startswith('1be'):
    truncate_len = 4096

datamodule = AavDataModule(
    snakemake.input['fasta'],
    token_per_batch=snakemake.params['token_per_batch'],
    truncate_len=truncate_len,
    num_workers=snakemake.threads,
)

devices = snakemake.params['devices']
lora_kwargs = None

_model = ESM.from_pretrained(
    snakemake.input['model'], checkpointing=True, device=devices[0])

wld_lora = snakemake.wildcards['lora']
if wld_lora != 'none':
    rank, layers = wld_lora.split(';')
    layers = layers.split(',')
    lora_kwargs = dict(rank=int(rank), layers=layers, adapter_names=['gb1'])
    _model.add_lora(**lora_kwargs)
else:
    for p in _model.parameters():
        p.requires_grad = False

head = BinaryLearnedAggregation(
    _model.attention_heads, _model.embed_dim, dtype=torch.float32).to(devices[0])

lr = 1e-5
lr_head = 1e-4

model = RegressionTrainer(
    _model, head, lr=lr, lr_head=lr_head, reduction=None).to(devices[0])

checkpoint_callback = ModelCheckpoint(
    dirpath=snakemake.params['checkpoint_dir'],
    save_top_k=5,
    monitor='val_loss',
    mode='min'
)

wandb_logger = WandbLogger(project='esme-aav')
wandb_logger.log_hyperparams({
    **snakemake.params, **snakemake.wildcards, 'lr': lr
})

trainer = L.Trainer(
    precision="bf16-mixed",
    devices=devices,
    max_epochs=snakemake.params['epochs'],
    callbacks=[
        SetEpochCallback(),
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min',
            min_delta=1e-5,
        ),
        checkpoint_callback
    ],
    logger=wandb_logger,
    log_every_n_steps=1,
    reload_dataloaders_every_n_epochs=1
)
trainer.fit(model=model, datamodule=datamodule)

best_model = RegressionTrainer.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model=_model, head=head, device=devices[0]
)

if wld_lora != 'none':
    best_model.plm.save_lora(snakemake.output['lora_weights'])
else:
    safetensors.save_file({}, snakemake.output['lora_weights'])

safetensors.save_file(best_model.head.state_dict(),
                      snakemake.output['head_weights'])
