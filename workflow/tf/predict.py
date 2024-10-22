from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from .tf import TfDataModule, TfModel
from more_itertools import collapse
from sklearn.metrics import precision_recall_curve, roc_curve
import safetensors.torch as safetensors


device = snakemake.params['device']
torch.cuda.set_device(device)

quantization = snakemake.wildcards['quantize']
quantization = quantization if quantization != 'none' else None

model = TfModel(snakemake.input['model'], device=device,
                quantization=quantization)

wld_lora = snakemake.wildcards['lora']
if wld_lora != 'none':
    model.plm.load_lora(snakemake.input['lora_weights'])

safetensors.load_model(model.head, snakemake.input['head_weights'])

datamodule = TfDataModule(
    snakemake.input['train'],
    snakemake.input['val'],
    snakemake.input['test'],
    batch_size=snakemake.params['batch_size'],
    num_workers=4
)

preds, targets = [], []

model.eval()
with torch.no_grad():
    for batch in tqdm(datamodule.test_dataloader()):
        preds.append(model(batch['token'].to(device)).cpu().float().numpy())
        targets.append(batch['label'].cpu().float().numpy())

preds = list(collapse(preds))
targets = list(collapse(targets))

precision, recall, thresholds = precision_recall_curve(targets, preds)
df_pr = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'threshold': np.concatenate([thresholds, [thresholds[-1]]])
})
df_pr.to_csv(snakemake.output['precision_recall'], index=False)

fpr, tpr, thresholds = roc_curve(targets, preds)
df_roc = pd.DataFrame({
    'fpr': fpr,
    'tpr': tpr,
    'threshold': thresholds,
})
df_roc.to_csv(snakemake.output['roc'], index=False)
