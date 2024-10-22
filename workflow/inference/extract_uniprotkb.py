import torch
import random
from pyfaidx import Fasta
from esme import tokenize


random.seed(41)

fasta = Fasta(snakemake.input['fasta'])

seq_lengths = {k: len(seq) for k, seq in fasta.items()}

tokens = {
    i + 100: tokenize(
        random.sample([
            str(fasta[k])
            for k, length in seq_lengths.items()
            if (i + 100) > length > i
        ], snakemake.params['batch_size'])
    )
    for i in range(0, 3500, 100)
}

torch.save(tokens, snakemake.output['tokens'])
