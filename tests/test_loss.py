from esme.loss import nll_loss
from esme.alphabet import mask_tokens, tokenize_unpad
from conftest import p53_human, device


def test_nll_loss(flash_esm2):

    tokens, _, cu_lens, max_len = tokenize_unpad([p53_human, p53_human * 2])
    tokens = tokens.to(device)
    cu_lens = cu_lens.to(device)

    mtokens, mask = mask_tokens(tokens, .5)
    probs = flash_esm2.predict_log_prob(mtokens, (cu_lens, max_len))

    loss = nll_loss(probs, tokens, mask)
    assert loss.shape == ()
    assert loss.item() > 0
