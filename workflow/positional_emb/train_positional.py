import os
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from safetensors.torch import save_model
from esme.esm import ESM1b, ESM1v
from esme.data import MaskedFastaTokenDataModule
from esme.embedding import LearnedPositionalEmbedding
from esme.trainer import MaskedPLM


lr = 1e-6
torch.set_float32_matmul_precision('medium')

# extend sequence length
max_seq_len = snakemake.params['max_seq_len']

checkpoint_dir = Path(snakemake.params['checkpoint_dir'])

if checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())) > 0:
    last_checkpoint = max(checkpoint_dir.iterdir(), key=os.path.getctime)

    masked_model = MaskedPLM.load_from_checkpoint(
        checkpoint_path=last_checkpoint, model=ESM1v(checkpointing=True))

else:
    params = torch.load(snakemake.input['model'])
    weights = dict()

    for k, v in params['model'].items():
        for i in ['sentence_encoder.', 'encoder.', '_proj']:
            k = k.replace(i, '')
        weights[k] = v

    if '1b' in snakemake.input['model']:
        model = ESM1b(checkpointing=True)
    else
    model = ESM1v(checkpointing=True)

    model.embed_positions = LearnedPositionalEmbedding(1024, 1280)
    model.load_state_dict(weights)
    model = model.to(torch.bfloat16)
    model.lm_head.weight = model.embed_tokens.weight

    positional_weight = model.embed_positions.weight.data
    model.embed_positions = LearnedPositionalEmbedding(max_seq_len, 1280)
    model.embed_positions.weight.data[:1026] = positional_weight

    # only train positional embeddings
    for param in model.parameters():
        param.requires_grad = False

    model.embed_positions.weight.requires_grad = True
    masked_model = MaskedPLM(model, lr=lr)
    last_checkpoint = None

datamodule = MaskedFastaTokenDataModule(
    train_fasta=snakemake.input['train'],
    val_fasta=snakemake.input['val'],
    token_per_batch=50_000, num_workers=snakemake.threads
)

checkpoint_callback = ModelCheckpoint(
    dirpath=snakemake.params['checkpoint_dir'],
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

wandb_logger = WandbLogger(project='esm1b-emdedding_size')
wandb_logger.log_hyperparams({**snakemake.params, 'lr': lr})

trainer = L.Trainer(
    devices=[0, 1, 2, 3],
    strategy='ddp',
    log_every_n_steps=1,
    max_epochs=snakemake.params['epochs'],
    val_check_interval=.01,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            mode='min',
            min_delta=1e-4,
        ),
        checkpoint_callback
    ],
    logger=wandb_logger,
    accumulate_grad_batches=64,
    precision="bf16-mixed",
    reload_dataloaders_every_n_epochs=1
)
trainer.fit(model=masked_model, datamodule=datamodule,
            ckpt_path=last_checkpoint)

# save model as safetensor
model = ESM1v()
model.embed_positions = LearnedPositionalEmbedding(max_seq_len, 1280)

best_model = MaskedPLM.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model=model,
)

metadata = {
    'format': 'pt',
    'name': 'esm1v',
    'num_layers': str(model.num_layers),
    'embed_dim': str(model.embed_dim),
    'attention_heads': str(model.attention_heads),
}
save_model(model, snakemake.output['model'], metadata=metadata)
