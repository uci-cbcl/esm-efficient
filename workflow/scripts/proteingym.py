from pathlib import Path
from warnings import warn
import pandas as pd
from tqdm import tqdm
from io import StringIO
from bioservices import UniProt
from pyfaidx import Fasta


db = UniProt()
df = list()

# not in uniprot
exclude_protein = {
    'ANCSZ_Hobbs',
    'PSAE_SYNP2'
}

substitutions = 'data/resources/proteingym/ProteinGym_substitutions/'

for path in tqdm(Path(substitutions).iterdir()):
    if not str(path).endswith('.csv'):
        continue

    dms_id = path.stem
    protein_name = dms_id.split('_')
    protein_name = '_'.join(protein_name[:2])

    df_dms = pd.read_csv(path)
    df_dms = df_dms[~df_dms['mutant'].str.contains(':')]
    row = df_dms.iloc[0]
    dms_seq = row.mutated_sequence
    pos = int(row.mutant[1:-1])
    dms_seq = dms_seq[:pos - 1] + row.mutant[0] + dms_seq[pos:]

    if protein_name in exclude_protein:
        continue

    columns = "id,length,accession,gene_names,reviewed,sequence"
    response = db.search(protein_name, limit=5, columns=columns)

    _df = pd.read_csv(StringIO(response), sep='\t')
    _df = _df[_df['Entry Name'] == protein_name]

    if _df.shape[0] > 1:
        _df = _df[_df['Reviewed'] == 'reviewed']

    if _df.shape[0] > 1:
        warn(f'More than one entry found for {protein_name}')
        continue

    if _df.shape[0] == 0:
        warn(f'No entry found for {protein_name}')
        continue

    seq = _df.iloc[0].Sequence

    if dms_seq not in seq:
        warn(f'Sequence mismatch sequence for {protein_name}')
        continue

    _df['seq_index'] = seq.index(dms_seq)
    _df['DMS_id'] = dms_id
    df.append(_df)

df = pd.concat(df).rename(columns={
    'Entry': 'Uniprot_ID',
    'Entry Name': 'Uniprot_Name',
    'Length': 'protein_length',
    'Reviewed': 'uniprotkb'
})
df['protein_length'] = df['protein_length'].astype(int)
df['uniprotkb'] = df['uniprotkb'] == 'reviewed'
df[['DMS_id', 'Uniprot_Name', 'Uniprot_ID', 'uniprotkb', 'protein_length']] \
    .to_csv('_proteingym.csv', index=False)
