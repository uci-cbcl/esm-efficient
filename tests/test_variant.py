import torch
from esme.alphabet import tokenize, mask_idx
from esme.variant import MaskMarginDataset, predict_pseudoperplexity
from esme import ESM2
from conftest import p53_human, esm2e_8M_path, calm1_human


def test_MaskMarginDataset():

    seq = p53_human

    dataset = MaskMarginDataset(seq)
    assert len(dataset) == len(seq)

    expected = {
        'pos': 1,
        'wt': 'M',
        'wt_token': 20,
    }
    token = tokenize([seq])[0]
    token[1] = mask_idx
    assert dataset[0]['pos'] == expected['pos']
    assert dataset[0]['wt'] == expected['wt']
    assert dataset[0]['wt_token'] == expected['wt_token']
    assert torch.equal(dataset[0]['token'], token)

    dataset = MaskMarginDataset(seq, max_len=50)
    assert len(dataset) == len(seq)
    expected = {
        'local_pos': 1,
        'wt': 'M',
        'wt_token': 20,
    }
    assert dataset[0]['local_pos'] == expected['local_pos']
    assert dataset[0]['wt'] == expected['wt']
    assert dataset[0]['wt_token'] == expected['wt_token']
    token = tokenize([seq])[0][:50]
    token[1] = mask_idx
    assert torch.equal(dataset[0]['token'], token)
    assert len(dataset[0]['token']) == 50

    expected = {
        'local_pos': 25,
        'wt': 'E',
        'wt_token': 9,
    }
    assert dataset[50]['local_pos'] == expected['local_pos']
    assert dataset[50]['wt'] == expected['wt']
    assert dataset[50]['wt_token'] == expected['wt_token']
    assert dataset[50]['token'][dataset[50]['local_pos']] == mask_idx

    token = tokenize([seq])[0][26:76]
    token[25] = mask_idx
    assert torch.equal(dataset[50]['token'], token)
    assert len(dataset[0]['token']) == 50

    dl = torch.utils.data.DataLoader(dataset, batch_size=32)

    for batch in dl:
        assert batch['token'].size(1) == 50


def test_predict_pseudoperplexity(flash_esm2, esm2_model):

    pps = predict_pseudoperplexity(flash_esm2, calm1_human)
    _pps = predict_pseudoperplexity(esm2_model, calm1_human)
    assert abs(_pps - pps) < .2

    pps = predict_pseudoperplexity(flash_esm2, p53_human)
    _pps = predict_pseudoperplexity(esm2_model, p53_human)
    assert abs(_pps - pps) < .2
