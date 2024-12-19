import torch
import lightning as L
from esme import ESM2
from esme.loss import nll_loss
from esme.data import MaskedFastaDataset, MaskedFastaTokenDataset


torch.set_float32_matmul_precision('medium')
quantization = snakemake.wildcards['quantize']
quantization = None if quantization == 'none' else quantization
lora_kwargs = None

efficient = snakemake.wildcards['model'].endswith('e')
checkpointing = snakemake.wildcards['checkpointing'] == 'True'

wld_lora = snakemake.wildcards['lora']
if wld_lora != 'none':
    rank, layers = wld_lora.split(';')
    layers = layers.split(',')
    lora_kwargs = dict(rank=int(rank), layers=layers, adapter_names='nll')


if efficient:
    strategy = 'ddp'
    device = 0
    model = ESM2.from_pretrained(
        snakemake.input['model'],
        quantization=quantization,
        checkpointing=checkpointing,
        device=0 if quantization is not None else 'cpu'
    )
    if lora_kwargs is not None:
        model.add_lora(**lora_kwargs)

    dl = MaskedFastaTokenDataset(
        snakemake.input['fasta'],
        token_per_batch=10_000,
        max_len=2000,
    ).to_dataloader(num_workers=snakemake.threads)

    class Model(L.LightningModule):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, token, unpad_args):
            return self.model.log_prob(token, unpad_args)

        def training_step(self, batch, batch_idx):
            target, unpad_args, token, mask = batch
            log_probs = self(token, unpad_args)
            return nll_loss(log_probs, target, mask)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

else:
    strategy = 'ddp_find_unused_parameters_true'
    import esm
    model, _ = esm.pretrained.load_model_and_alphabet(snakemake.input['model'])
    model = model.to(dtype=torch.bfloat16)  # .to(device)

    batch_size = 1

    if '8M' == snakemake.wildcards['model']:
        batch_size = 4
    elif '35M' == snakemake.wildcards['model']:
        batch_size = 4
    elif '150M' == snakemake.wildcards['model']:
        batch_size = 2

    dl = MaskedFastaDataset(
        snakemake.input['fasta'], max_len=2000
    ).to_dataloader(
        batch_size=batch_size,
        num_workers=snakemake.threads
    )

    class Model(L.LightningModule):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            out = out['logits']
            return torch.log_softmax(out, dim=-1)

        def training_step(self, batch, batch_idx):
            target, token, mask = batch
            log_probs = self(token)
            return nll_loss(log_probs, target, mask)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())


_model = Model(model)

trainer = L.Trainer(
    devices=[0, 1, 2, 3],
    strategy=strategy,
    max_epochs=1,
    accumulate_grad_batches=16,
    precision="bf16-mixed"
)
trainer.fit(model=_model, train_dataloaders=dl)
