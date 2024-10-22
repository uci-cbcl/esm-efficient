from sklearn.metrics import precision_recall_curve, roc_curve
import pdb
from tqdm import tqdm
from more_itertools import collapse
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
import torch.nn as nn
from torch.utils.data import Dataset


device = snakemake.params['device']


class DeepTFactorDataset(Dataset):
    def __init__(self, data):
        self.getAAmap()
        self.data = list(self.preprocess(data))

    def __len__(self):
        return len(self.data)

    def preprocess(self, seqs, len_criteria=1000):
        for seq in seqs:
            if len(seq) <= len_criteria:
                seq += '_' * (len_criteria-len(seq))
            yield seq

    def getAAmap(self):
        aa_vocab = ['A', 'C', 'D', 'E',
                    'F', 'G', 'H', 'I',
                    'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S',
                    'T', 'V', 'W', 'X',
                    'Y', '_']
        map = {}
        for i, char in enumerate(aa_vocab):
            baseArray = np.zeros(len(aa_vocab)-1)
            if char != '_':
                baseArray[i] = 1
            map[char] = baseArray
        self.map = map
        return

    def convert2onehot(self, single_seq):
        single_onehot = []
        for x in single_seq:
            single_onehot.append(self.map[x])
        return np.asarray(single_onehot)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.convert2onehot(x)
        x = x.reshape((1,) + x.shape)
        return x


class CNN(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16
                  8 -> 12
                  16 -> 4
    '''

    def __init__(self, layer_info):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList()
        pooling_sizes = []
        for subnetwork in layer_info:
            pooling_size = 0
            self.layers += [self.make_subnetwork(subnetwork)]
            for kernel in subnetwork:
                pooling_size += (-kernel + 1)
            pooling_sizes.append(pooling_size)

        if len(set(pooling_sizes)) != 1:
            raise "Different kernel sizes between subnetworks"
        pooling_size = pooling_sizes[0]
        num_subnetwork = len(layer_info)

        self.conv = nn.Conv2d(in_channels=128*num_subnetwork,
                              out_channels=128*3, kernel_size=(1, 1))
        self.batchnorm = nn.BatchNorm2d(num_features=128*3)
        self.pool = nn.MaxPool2d(kernel_size=(1000+pooling_size, 1), stride=1)

    def make_subnetwork(self, subnetwork):
        subnetworks = []
        for i, kernel in enumerate(subnetwork):
            if i == 0:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=128,
                                  kernel_size=(kernel, 21)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
            else:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=128,
                                  kernel_size=(kernel, 1)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
        return nn.Sequential(*subnetworks)

    def forward(self, x):
        xs = []
        for layer in self.layers:
            xs.append(layer(x))
        x = torch.cat(xs, dim=1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)
        return x


class DeepTFactor(nn.Module):
    def __init__(self, out_features=[0]):
        super(DeepTFactor, self).__init__()
        self.explainECs = out_features
        self.layer_info = [[4, 4, 16], [12, 8, 4], [16, 4, 4]]
        self.cnn0 = CNN(self.layer_info)
        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.out_act(self.bn2(self.fc2(x)))
        return x


df_test = pd.read_parquet(snakemake.input['test'])
dataset = DeepTFactorDataset(df_test['seq'].tolist())
dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model = DeepTFactor(out_features=[1])
model.load_state_dict(torch.load(snakemake.input['weights'])['model'])
model = model.to(device).half()

row = list(dl)[73]

with torch.no_grad():
    model.eval()
    y = model(row.half().to(device)).cpu().numpy()

preds = []

with torch.no_grad():
    model.eval()
    for i, batch in tqdm(enumerate(dl)):
        preds.append(model(batch.half().to(device)).max().cpu().numpy())

preds = list(collapse(preds))
targets = list(df_test['tf'].values)

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
