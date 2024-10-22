from datetime import datetime, date
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO


np.random.seed(41)

tf_go_terms = set([i.strip() for i in open(snakemake.input['tf_go_terms'])])

go_terms = list()

for record in tqdm(SeqIO.parse(snakemake.input['uniprot'], "swiss")):
    terms = [
        i[3:]
        for i in record.dbxrefs
        if i.startswith('GO:')
    ]
    date = datetime.strptime(record.annotations['date'], "%d-%b-%Y")

    if len(terms) > 0:
        go_terms.append((record.id, terms, date, str(record.seq)))

df = pd.DataFrame(go_terms)
df.columns = ['entry', 'go_terms', 'date', 'seq']
df['tf'] = df['go_terms'].map(lambda x: any(i in tf_go_terms for i in x))

df_test = df[df['date'] >= datetime(2021, 1, 1)]
df_train = df.loc[df.index.difference(df_test.index)]
df_val = df_train.sample(frac=0.1)
df_train = df_train.loc[df_train.index.difference(df_val.index)]

df_train.to_parquet(snakemake.output['train'])
df_val.to_parquet(snakemake.output['val'])
df_test.to_parquet(snakemake.output['test'])
