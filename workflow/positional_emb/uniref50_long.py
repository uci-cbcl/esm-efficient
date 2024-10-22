import polars as pl
from tqdm import tqdm


cols = ["id", "length", "offset", "line_bases", "line_width"]
df = pl.scan_csv(snakemake.input['fai'], separator="\t",
                 has_header=False, new_columns=cols)

df = df.filter(pl.col("length") > 1026 - 2) \
    .filter(pl.col("length") < snakemake.params['max_seq_len'] - 1) \
    .collect().to_pandas().sort_values('length', ascending=False)

fasta = open(snakemake.input['fasta'])
fasta_train = open(snakemake.output['train'], 'w')
fasta_val = open(snakemake.output['val'], 'w')

for i, row in tqdm(enumerate(df.itertuples()), total=df.shape[0]):
    fasta.seek(row.offset)

    f = fasta_train if i % 1000 != 0 else fasta_val
    f.write(f'>{row.id}\n')

    while True:
        line = fasta.readline()

        if line.startswith('>'):
            break

        f.write(line)

fasta.close()
fasta_train.close()
fasta_val.close()
