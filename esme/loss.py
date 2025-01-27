import torch.nn.functional as F
from esme.alphabet import Alphabet3


def nll_loss(log_probs, tokens, mask, nll_loss_kwargs=None, alphabet=Alphabet3):
    '''
    Negative log likelihood loss from masked tokens.

    Args:
        log_probs (Tensor): Log probabilities from the model.
        tokens (Tensor): Target tokens.
        masked_tokens (Tensor): Masked tokens.
    Returns:
        Tensor: Negative log likelihood loss.
    Examples:
        >>> masked_tokens = mask_tokens(tokens)
        >>> log_probs = model.predict_log_prob(masked_tokens)
        >>> nll_loss(log_probs, tokens, masked_tokens)
    '''
    log_probs = log_probs.view(-1, log_probs.size(-1))
    targets = tokens.view(-1)
    mask = mask.view(-1)

    log_probs = log_probs[mask]
    targets = targets[mask]

    return F.nll_loss(log_probs, targets, ignore_index=alphabet.padding_idx,
                      **(nll_loss_kwargs or {}))


def cross_entropy(logits, tokens, mask, cross_entropy_loss_kwargs=None, alphabet=Alphabet3):
    '''
    Negative log likelihood loss from masked tokens.

    Args:
        log_probs (Tensor): Log probabilities from the model.
        tokens (Tensor): Target tokens.
        masked_tokens (Tensor): Masked tokens.
    Returns:
        Tensor: Negative log likelihood loss.
    Examples:
        >>> masked_tokens = mask_tokens(tokens)
        >>> logits = model(masked_tokens)
        >>> cross_entropy(log_probs, tokens, masked_tokens)
    '''
    logits = logits.view(-1, logits.size(-1))
    targets = tokens.view(-1)
    mask = mask.view(-1)

    logits = logits[mask]
    targets = targets[mask]

    return F.cross_entropy(logits, targets, ignore_index=alphabet.padding_idx,
                           **(cross_entropy_loss_kwargs or {}))
