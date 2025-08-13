import os
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import safetensors.torch as safetensors
from esme.data import SetEpochCallback
from workflow.meltome.meltome import MeltomeDataModule, MeltomeModel


torch.set_float32_matmul_precision('medium')

devices = snakemake.params['devices']

quantization = snakemake.wildcards['quantize']
quantization = None if quantization == 'none' else quantization
lora_kwargs = None

wld_lora = snakemake.wildcards['lora']
if wld_lora != 'none':
    rank, layers = wld_lora.split(';')
    layers = layers.split(',')
    lora_kwargs = dict(rank=int(rank), layers=layers, adapter_names='meltome')

cls_model = MeltomeModel
lr = 1e-4
lr_head = 1e-3

checkpoint_dir = Path(snakemake.params['checkpoint_dir'])
if checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())) > 0:
    last_checkpoint = max(checkpoint_dir.iterdir(), key=os.path.getctime)

    model = cls_model.load_from_checkpoint(
        checkpoint_path=last_checkpoint, model_path=snakemake.input['model'],
        lr=lr, lr_head=lr_head, lora_kwargs=lora_kwargs,
        device=devices[0])
else:
    last_checkpoint = None
    model = cls_model(snakemake.input['model'], lr=lr, lr_head=lr_head,
                      lora_kwargs=lora_kwargs, device=devices[0])

if wld_lora == 'none':
    for p in model.plm.parameters():
        p.requires_grad = False

truncate_len = None
if snakemake.wildcards['model'].startswith('1ve') or snakemake.wildcards['model'].startswith('1be'):
    truncate_len = 4096 - 2

datamodule = MeltomeDataModule(
    snakemake.input['dataset'],
    token_per_batch=50_000,
    num_workers=0,
    truncate_len=truncate_len,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=snakemake.params['checkpoint_dir'],
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

wandb_logger = WandbLogger(project='esme-meltome')
wandb_logger.log_hyperparams({
    **snakemake.params, **snakemake.wildcards, 'lr': lr
})

trainer = L.Trainer(
    devices=devices,
    max_epochs=snakemake.params['epochs'],
    callbacks=[
        SetEpochCallback(),
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            mode='min',
            min_delta=1e-5,
        ),
        checkpoint_callback
    ],
    logger=wandb_logger,
    accumulate_grad_batches=16,
    precision="bf16-mixed",
    reload_dataloaders_every_n_epochs=1
)
trainer.fit(model=model, datamodule=datamodule, ckpt_path=last_checkpoint)

best_model = cls_model.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model_path=snakemake.input['model'],
    lora_kwargs=lora_kwargs, device=devices[0]
)

if wld_lora != 'none':
    best_model.plm.save_lora(snakemake.output['lora_weights'])
else:
    safetensors.save_file({}, snakemake.output['lora_weights'])

safetensors.save_file(best_model.head.state_dict(),
                      snakemake.output['head_weights'])
