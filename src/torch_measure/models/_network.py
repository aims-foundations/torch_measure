# Copyright (c) 2026 AIMS Foundation. MIT License.

"""MLP network for amortized parameter prediction."""

from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron for amortizing IRT parameters from embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings.
    hidden_dim : int
        Dimension of hidden layers.
    output_dim : int
        Dimension of output (e.g., difficulty + discrimination + guessing).
    n_layers : int
        Number of layers (minimum 1).
    dropout : float
        Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_layers == 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ELU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
