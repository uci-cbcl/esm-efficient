import pytest
import torch
from esme.alphabet import tokenize
from esme.data import read_fai, TokenSizeBatchSampler, FastaDataset, \
    FastaTokenDataset, MaskedFastaDataset
from conftest import fai_path, fasta_path


def test_read_fai():
    df = read_fai(fai_path)
    assert df.shape == (16, 5)
    assert df['length'].to_list() == [256, 320, 458, 156, 438, 60, 217, 204,
                                      352, 75, 128, 447, 347, 948, 85, 137]


def test_token_size_batch_sampler():
    fai = read_fai(fai_path)
    lenghts = fai['length'].to_list()

    batch_sampler = TokenSizeBatchSampler(lenghts, 1500)

    for idx in batch_sampler:
        sizes = [lenghts[i] for i in idx]
        assert sum(sizes) <= 1500

    batch_sampler = TokenSizeBatchSampler(lenghts, 400, shuffle=False)
    assert list(iter(batch_sampler)) == [
        [0], [1], [2], [3], [4], [5, 6], [7], [8],
        [9, 10], [11], [12], [13], [14, 15]
    ]
    assert len(batch_sampler) == 13


@pytest.fixture
def fasta_dataset():
    return FastaDataset(fasta_path)


@pytest.fixture
def fasta_token_dataset():
    return FastaTokenDataset(fasta_path, token_per_batch=1500)


def test_FastaDataset_read_seq(fasta_dataset):
    seq = fasta_dataset.read_seq(0)

    assert seq == 'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNP'\
        'PSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAY'\
        'NLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVES'\
        'AHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL'

    seq_lens = [256, 320, 458, 156, 438, 60, 217, 204,
                352, 75, 128, 447, 347, 948, 85, 137]

    for i in range(16):
        seq = fasta_dataset.read_seq(i)
        assert len(seq) == seq_lens[i]

    dl = fasta_dataset.to_dataloader(
        batch_size=1, num_workers=4, shuffle=False)
    tokens = list(iter(dl))

    for i in range(16):
        assert tokens[i].shape[1] == seq_lens[i] + 2

    seq = fasta_dataset.read_seq(0)
    assert torch.all(tokens[0][0] == tokenize(seq))


def test_FastaDataset_len(fasta_dataset):
    assert len(fasta_dataset) == 16


def test_FastaDataset_max_len():
    dataset = FastaDataset(fasta_path, max_len=200)
    assert len(dataset) == 6

    for i in range(6):
        token = dataset[i]
        assert token.shape[1] <= 200


def test_FastaDataset_batch_size(fasta_dataset):
    dl = fasta_dataset.to_dataloader(batch_size=4)

    batch = next(iter(dl))
    assert batch.shape[0] == 4
    assert batch.shape[1] == 460


def test_FastaDataset_token_per_batch(fasta_token_dataset):
    dl = fasta_token_dataset.to_dataloader()

    for token, (cu_lens, max_len) in dl:
        assert token.shape[0] <= 1500


def test_MaskedFastaDataset():
    ds = MaskedFastaDataset(fasta_path)
    token, mtokens, mask = ds[0]
    assert token[~mask].eq(mtokens[~mask]).all()
    assert not token[mask].eq(mtokens[mask]).all()

    dl = ds.to_dataloader(batch_size=4)
    token, mtokens, mask = next(iter(dl))
    assert token[~mask].eq(mtokens[~mask]).all()
    assert not token[mask].eq(mtokens[mask]).all()
