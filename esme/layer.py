from torch import nn


class FeedForward(nn.Module):
    """
    A simple feed forward neural network with a single hidden layer and ReLU activation.

    Args:
        embed_dim (int): The size of the input embedding.
        hidden_dim (int): The size of the hidden layer.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
