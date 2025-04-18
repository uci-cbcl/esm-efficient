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
from esme.data import SetEpochCallback, LabeledDataModule
from esme import ESM2
from esme.pooling import BinaryLearnedAggregation
from esme.trainer import RegressionTrainer
# from workflow.gb1.gb1 import Gb1DataModule


torch.set_float32_matmul_precision('medium')

class Gb1DataModule(LabeledDataModule):

    def __init__(self, fasta_path, token_per_batch=None, num_workers=0):
        ''''''
        self.token_per_batch = token_per_batch
        self.num_workers = num_workers
        df = list()

        with open(fasta_path, 'r') as f:
            seq = list()
            for line in f:
                if line.startswith('>'):
                    if seq:
                        df.append({
                            'seq': ''.join(seq),
                            'label': label,
                            'split': 'val' if validation else split
                        })
                        seq = list()
                    _, label, split, validation = line.split()
                    label = float(label.split('=')[1])
                    split = split.split('=')[1]
                    validation = validation.split('=')[1] == 'True'
                else:
                    seq.append(line.strip())
            if seq:
                df.append({
                    'seq': ''.join(seq),
                    'label': label,
                    'split': 'val' if validation else split
                })

        df = pd.DataFrame(df)
        df_train = df[df['split'] == 'train']
        df_val = df[df['split'] == 'val']
        df_test = df[df['split'] == 'test']

        super().__init__(
            train_seqs=df_train['seq'].tolist(),
            train_labels=(df_train['label'] / 10 + .5).tolist(),
            val_seqs=df_val['seq'].tolist(),
            val_labels=(df_val['label'] / 20 + .5).tolist(),
            test_seqs=df_test['seq'].tolist(),
            test_labels=(df_test['label'] / 20 + .5).tolist(),
            token_per_batch=token_per_batch,
            num_workers=self.num_workers
        )

datamodule = Gb1DataModule(
    snakemake.input['fasta'],
    token_per_batch=snakemake.params['token_per_batch'],
    num_workers=snakemake.threads,
)

devices = snakemake.params['devices']
lora_kwargs = None

_model = ESM2.from_pretrained(
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

head = BinaryLearnedAggregation(_model.attention_heads, _model.embed_dim, dtype=torch.float32).to(devices[0])

lr = 1e-5
lr_head = 1e-4

model = RegressionTrainer(
    _model, head, lr=lr, lr_head=lr_head, reduction=None).to(devices[0])

# for batch in datamodule.train_dataloader():

#     pad_args = (batch['cu_lens'].to(devices[0]), batch['max_len'])

#     embed = _model.forward_representation(
#         batch['token'].to(devices[0]),
#         pad_args,
#         pad_output=False,
#         pad_indices=batch['indices'].to(devices[0]),
#     )
#     # print(embed.shape)

#     x = head(embed, pad_args)
#     print(x.shape)
#     breakpoint()

#     pred = model.training_step({
#         'token': batch['token'].to(devices[0]),
#         'cu_lens': batch['cu_lens'].to(devices[0]),
#         'max_len': batch['max_len'],
#         'indices': batch['indices'].to(devices[0]),
#         'label': batch['label'].to(devices[0]),
#     }, batch_idx=0)

checkpoint_callback = ModelCheckpoint(
    dirpath=snakemake.params['checkpoint_dir'],
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

wandb_logger = WandbLogger(project='esme-aav')
wandb_logger.log_hyperparams({
    **snakemake.params, **snakemake.wildcards, 'lr': lr
})

trainer = L.Trainer(
    precision="bf16-true",
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
