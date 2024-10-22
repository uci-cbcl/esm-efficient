import torch
from esme.embedding import LearnedPositionalEmbedding


def test_LearnedPositionalEmbedding():

    num_embeddings = 33
    embedding_dim = 4096

    learned_positional_embedding = LearnedPositionalEmbedding(
        num_embeddings, embedding_dim, torch.bfloat16)

    assert learned_positional_embedding.num_embeddings == num_embeddings + 2
    assert learned_positional_embedding.embedding_dim == embedding_dim
    assert learned_positional_embedding.padding_idx == 1
    assert learned_positional_embedding.max_positions == num_embeddings

    x = torch.tensor([[20, 29, 28], [8, 13, 9]])
    pos_embed = learned_positional_embedding.positions(x)

    assert pos_embed.size() == x.size()
    assert torch.all(torch.tensor([[2, 3, 4], [2, 3, 4]]) == pos_embed)

    x = torch.tensor([20, 29, 28, 8, 13, 9])
    pad_args = (torch.tensor([0, 3, 6]), 3)
    pos_embed = learned_positional_embedding.position_unpad(x, pad_args)

    assert pos_embed.size() == x.size()
    assert torch.all(torch.tensor([2, 3, 4, 2, 3, 4]) == pos_embed)
