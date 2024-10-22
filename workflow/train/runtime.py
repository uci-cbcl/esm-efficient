import argparse
import torch
import deepspeed
from esme.loss import nll_loss
from esme.data import MaskedFastaTokenDataset
from esme.esm import ESM2, DEEPSPEED_STAGE2_CONFIG

torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--fasta')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--token_per_batch', type=int)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

device = args.local_rank
model = ESM2.from_pretrained(
    args.model, checkpointing='deepseed', device=device)

ds = MaskedFastaTokenDataset(
    args.fasta,
    token_per_batch=args.token_per_batch,
    max_len=2000,
)

config = DEEPSPEED_STAGE2_CONFIG.copy()
config["train_micro_batch_size_per_gpu"] = 1
config["gradient_accumulation_steps"] = 16
config["data_sampling"] = {"num_workers": 0}


model_engine, optimizer, dl, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=ds,
    config=config
)


for target, (cu_lens, max_len), token, mask in dl:
    token = token[0].to(device)
    target = target[0].to(device) 
    cu_lens = cu_lens[0].to(device)
    max_len = max_len[0].to(device)
    mask = mask[0].to(device)

    log_probs = model_engine.predict_log_prob(token, (cu_lens, max_len))
    loss = nll_loss(log_probs, target, mask)
    model_engine.backward(loss)
    model_engine.step()
