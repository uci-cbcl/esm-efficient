import torch
from flash_attn.bert_padding import unpad_input
from esme.alphabet import tokenize, mask_tokens, tokenize_unpad, \
    split_alphabet, token_to_str, Alphabet, pad_tokens
from conftest import p53_human, calm1_human


def test_tokenize(alphabet):
    tokens = tokenize(p53_human)
    assert torch.all(tokens == torch.tensor([
        0, 20,  9,  9, 14, 16,  8, 13, 14,  8,  7,  9, 14, 14,  4,  8, 16,  9,
        11, 18,  8, 13,  4, 22, 15,  4,  4, 14,  9, 17, 17,  7,  4,  8, 14,  4,
        14,  8, 16,  5, 20, 13, 13,  4, 20,  4,  8, 14, 13, 13, 12,  9, 16, 22,
        18, 11,  9, 13, 14,  6, 14, 13,  9,  5, 14, 10, 20, 14,  9,  5,  5, 14,
        14,  7,  5, 14,  5, 14,  5,  5, 14, 11, 14,  5,  5, 14,  5, 14,  5, 14,
        8, 22, 14,  4,  8,  8,  8,  7, 14,  8, 16, 15, 11, 19, 16,  6,  8, 19,
        6, 18, 10,  4,  6, 18,  4, 21,  8,  6, 11,  5, 15,  8,  7, 11, 23, 11,
        19,  8, 14,  5,  4, 17, 15, 20, 18, 23, 16,  4,  5, 15, 11, 23, 14,  7,
        16,  4, 22,  7, 13,  8, 11, 14, 14, 14,  6, 11, 10,  7, 10,  5, 20,  5,
        12, 19, 15, 16,  8, 16, 21, 20, 11,  9,  7,  7, 10, 10, 23, 14, 21, 21,
        9, 10, 23,  8, 13,  8, 13,  6,  4,  5, 14, 14, 16, 21,  4, 12, 10,  7,
        9,  6, 17,  4, 10,  7,  9, 19,  4, 13, 13, 10, 17, 11, 18, 10, 21,  8,
        7,  7,  7, 14, 19,  9, 14, 14,  9,  7,  6,  8, 13, 23, 11, 11, 12, 21,
        19, 17, 19, 20, 23, 17,  8,  8, 23, 20,  6,  6, 20, 17, 10, 10, 14, 12,
        4, 11, 12, 12, 11,  4,  9, 13,  8,  8,  6, 17,  4,  4,  6, 10, 17,  8,
        18,  9,  7, 10,  7, 23,  5, 23, 14,  6, 10, 13, 10, 10, 11,  9,  9,  9,
        17,  4, 10, 15, 15,  6,  9, 14, 21, 21, 9,  4, 14, 14,  6,  8, 11, 15,
        10,  5,  4, 14, 17, 17, 11,  8,  8,  8, 14, 16, 14, 15, 15, 15, 14,  4,
        13,  6,  9, 19, 18, 11,  4, 16, 12, 10,  6, 10,  9, 10, 18,  9, 20, 18,
        10,  9,  4, 17,  9,  5,  4,  9,  4, 15, 13,  5, 16,  5,  6, 15,  9, 14,
        6,  6,  8, 10,  5, 21,  8,  8, 21,  4, 15,  8, 15, 15,  6, 16,  8, 11,
        8, 10, 21, 15, 15,  4, 20, 18, 15, 11,  9,  6, 14, 13,  8, 13,  2
    ]))
    assert torch.all(tokens == tokenize([p53_human]))

    seqs = [p53_human, p53_human + p53_human]
    tokens = tokenize(seqs)

    batch_converter = alphabet.get_batch_converter()

    esm2_token = batch_converter((
        ('tp53', seqs[0]),
        ('tp53', seqs[1])
    ))[-1]
    assert torch.all(tokens == esm2_token)

    seq = p53_human[:10] + '<mask>' + p53_human[11:]
    tokens = tokenize(seq)
    assert torch.all(tokens[0, 11] == Alphabet.mask_idx)


def test_tokenize_unpad():

    seqs = [p53_human, p53_human + p53_human]
    tokens, indices, cu_lens, max_len = tokenize_unpad(seqs)

    assert list(tokens.shape) == [len(p53_human) * 3 + 4]
    assert torch.all(indices == torch.cat([
        torch.arange(0, len(p53_human) + 2),
        torch.arange(len(p53_human) * 2 + 2, len(p53_human) * 4 + 4),
    ]))
    assert cu_lens.tolist() == [0, len(p53_human) + 2, len(p53_human) * 3 + 4]
    assert max_len == len(p53_human) * 2 + 2
    
    seqs = [p53_human, p53_human + p53_human, calm1_human]
    tokens, indices, cu_lens, max_len = tokenize_unpad(seqs)

    tokens_pad = pad_tokens([
        tokenize(s)
        for s in seqs
    ])

    embedding_layer = torch.nn.Embedding(33, 32)
    embed = embedding_layer(tokens_pad)

    embed_unpad, _indices, _cu_lens, _max_len, _ = unpad_input(
        hidden_states=embed, attention_mask=~tokens_pad.eq(Alphabet.padding_idx))

    assert torch.all(embed_unpad == embedding_layer(tokens))
    assert torch.all(cu_lens == _cu_lens)
    assert max_len == _max_len
    assert torch.all(indices == _indices)


def test_mask_tokens():
    tokens = tokenize(p53_human)
    mtokens, mask = mask_tokens(tokens)
    assert (mtokens == Alphabet.mask_idx).sum() > 0

    tokens = tokenize(p53_human)
    mtokens, mask = mask_tokens(tokens, alter=.5)
    assert not torch.all(mtokens[mask] != tokens[mask])
    assert not torch.all(mtokens[mask] == tokens[mask])

    seqs = [p53_human, p53_human + p53_human]
    tokens = tokenize(seqs)
    mtokens, mask = mask_tokens(tokens)
    assert (mtokens == Alphabet.mask_idx).sum() > 0

    mtokens, mask = mask_tokens(tokens, .01)
    assert ((mtokens == Alphabet.mask_idx).sum(axis=1) > 0).all()

    mtokens, mask = mask_tokens(tokens, .0, .0)
    assert ((mtokens == Alphabet.mask_idx).sum(axis=1) == 1).all()

    tokens, _, cu_lens, max_len = tokenize_unpad(p53_human)
    mtokens, mask = mask_tokens(tokens, .0, .0)
    assert ((mtokens == Alphabet.mask_idx).sum(axis=-1) == 1).all()


def test_split_alphabet():
    seq = p53_human[:10] + '<mask>' + p53_human[11:]
    split = split_alphabet(seq)
    assert len(split) == len(p53_human)
    assert split[10] == '<mask>'


def test_token_to_str():
    seq = p53_human[:10] + '<mask>' + p53_human[11:]
    assert token_to_str(tokenize(seq)) == ['<cls>' + seq + '<eos>']
