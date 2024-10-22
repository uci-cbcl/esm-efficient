import pandas as pd
import torch
from tqdm import tqdm
from esme import ESM2
from esme.alphabet import tokenize
from workflow.utils import benchmark_memory


device = 0
torch.cuda.set_device(device)

model = ESM2.from_pretrained(snakemake.input['model'], device=device)
model.eval()


def fn(x):
    with torch.no_grad():
        out = model(x)


memory_usage = list()

token_sizes = [10_000, 25_000, 50_000, 75_000, 100_000]

# protein with 250 aa
seq = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRP'

for length in tqdm(token_sizes):
    token = tokenize(seq * (length // 250))

    try:
        mem = benchmark_memory(fn, {'x': token.to(device)}, device)
    except torch.cuda.OutOfMemoryError:
        mem = -1  # oom

    memory_usage.append({'length': length, 'mem_gb': mem})
    pd.DataFrame(memory_usage).to_csv(snakemake.output['mem'], index=False)

    if mem == -1:
        break
