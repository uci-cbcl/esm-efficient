import pandas as pd
from esme.data import LabeledDataModule


class AavDataModule(LabeledDataModule):

    def __init__(self, fasta_path, token_per_batch=None, num_workers=0, truncate_len=None):
        ''''''
        self.token_per_batch = token_per_batch
        self.num_workers = num_workers
        df = list()

        with open(fasta_path, 'r') as f:
            seq = list()
            for line in f:
                if line.startswith('>'):
                    if seq:
                        df.append({
                            'seq': ''.join(seq),
                            'label': label,
                            'split': 'val' if validation else split
                        })
                        seq = list()
                    _, label, split, validation = line.split()
                    label = float(label.split('=')[1])
                    split = split.split('=')[1]
                    validation = validation.split('=')[1] == 'True'
                else:
                    seq.append(line.strip())
            if seq:
                df.append({
                    'seq': ''.join(seq),
                    'label': label,
                    'split': 'val' if validation else split
                })

        df = pd.DataFrame(df)
        df_train = df[df['split'] == 'train']
        df_val = df[df['split'] == 'val']
        df_test = df[df['split'] == 'test']

        super().__init__(
            train_seqs=df_train['seq'].tolist(),
            train_labels=(df_train['label'] / 10 + .5).tolist(),
            val_seqs=df_val['seq'].tolist(),
            val_labels=(df_val['label'] / 20 + .5).tolist(),
            test_seqs=df_test['seq'].tolist(),
            test_labels=(df_test['label'] / 20 + .5).tolist(),
            token_per_batch=token_per_batch,
            truncate_len=truncate_len,
            num_workers=self.num_workers
        )

