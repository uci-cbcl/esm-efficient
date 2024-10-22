from workflow.utils import benchmark_memory
import pandas as pd
import torch
import esm
from tqdm import tqdm
from esme.esm import DEEPSPEED_STAGE2_CONFIG
from esme.loss import nll_loss
from esme.alphabet import mask_tokens


device = snakemake.params['device']
torch.cuda.set_device(device)

batch_size = int(snakemake.wildcards['batch_size'])
assert batch_size <= 32, \
    "32 is the maximum batch size supported for the bechmark."

quantization = snakemake.wildcards['quantize']
quantization = None if quantization == 'none' else quantization
checkpointing = snakemake.wildcards['checkpointing'] == 'True'
use_deepspeed = snakemake.wildcards['deepspeed'] == 'True'
efficient = snakemake.wildcards['model'].endswith('e')

if efficient:
    from esme import ESM2
    model = ESM2.from_pretrained(
        snakemake.input['model'], quantization=quantization,
        device=device, checkpointing=checkpointing)
else:
    assert quantization is None
    import esm
    model, _ = esm.pretrained.load_model_and_alphabet(snakemake.input['model'])
    model = model.to(torch.bfloat16).to(device)


if snakemake.wildcards['lora'] != 'none':
    assert efficient
    rank, layers = snakemake.wildcards['lora'].split(';')
    layers = layers.split(',')
    model.add_lora(rank=int(rank), layers=layers)

if use_deepspeed:
    def make_inputs_require_grads(module, input, output):
        output.requires_grad_(True)

    model._require_grads_hook = model.embed_tokens.register_forward_hook(
        make_inputs_require_grads)


tokens = torch.load(snakemake.input['tokens'])

if use_deepspeed:
    config = DEEPSPEED_STAGE2_CONFIG.copy()
    config["train_micro_batch_size_per_gpu"] = batch_size
    config["gradient_accumulation_steps"] = 1
    model_engine, _ = model.to_deepspeed(
        local_rank=device, deepspeed_config=config)

    def fn(token, mtoken):
        log_probs = model.predict_log_prob(mtoken)
        loss = nll_loss(log_probs, token, mtoken)
        # loss.requires_grad_(True)
        model_engine.backward(loss)
        model_engine.step()
else:
    model_engine = model
    optimizer = torch.optim.Adam(model_engine.parameters())

    if efficient:
        def fn(token, mtoken):
            log_probs = model.predict_log_prob(mtoken)
            loss = nll_loss(log_probs, token, mtoken)
            loss.backward()
            optimizer.step()
    else:
        def fn(token, mtoken):
            log_probs = torch.log_softmax(
                model_engine(mtoken)['logits'], dim=-1)
            loss = nll_loss(log_probs, token, mtoken)
            loss.backward()
            optimizer.step()


memory_usage = list()


for length, tokens in tqdm(tokens.items()):
    tokens = tokens[:batch_size]
    mtokens, mask = mask_tokens(tokens, .15)
    try:
        mem = benchmark_memory(fn, {
            'token': tokens.to(device),
            'mtoken': mtokens.to(device),
        }, device)
    except torch.cuda.OutOfMemoryError:
        mem = -1  # oom

    memory_usage.append({'length': length, 'mem_gb': mem})
    pd.DataFrame(memory_usage).to_csv(snakemake.output['mem'], index=False)

    if mem == -1:
        break
