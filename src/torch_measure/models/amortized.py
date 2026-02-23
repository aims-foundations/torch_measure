# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Amortized IRT model that predicts item parameters from embeddings.

Consolidated from agent-eval/model/amortized_irt.py and predictive-eval/train/amortized_irt/irt.py.
"""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel
from torch_measure.models._network import MLP


class AmortizedIRT(IRTModel):
    """Amortized IRT model.

    Instead of learning independent parameters for each item, this model
    learns a mapping from item embeddings to item parameters (difficulty,
    discrimination, guessing). This enables zero-shot prediction on new
    items given their embeddings.

    P(correct) = c + (1-c) * sigmoid(a * (theta - b))

    where b, a, c = f(embedding) are predicted by a neural network.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    embedding_dim : int
        Dimension of item embeddings.
    hidden_dim : int
        Hidden dimension for the embedding projection network.
    n_layers : int
        Number of layers in the projection network.
    pl : int
        Number of IRT parameters: 1 (Rasch), 2 (+discrimination), 3 (+guessing).
    dropout : float
        Dropout rate in the projection network.
    device : str
        Device to place parameters on.
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        pl: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.pl = pl
        self.embedding_dim = embedding_dim

        # Subject ability parameters (learned directly)
        self.ability = nn.Parameter(torch.randn(n_subjects, device=self._device))

        # Item parameter projection network
        # Output: difficulty + (discrimination if pl>=2) + (guessing if pl==3)
        output_dim = 1 + (1 if pl >= 2 else 0) + (1 if pl == 3 else 0)
        self.item_net = MLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

        self._embeddings: torch.Tensor | None = None

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        """Set item embeddings for parameter prediction.

        Parameters
        ----------
        embeddings : torch.Tensor
            Item embeddings of shape (n_items, embedding_dim).
        """
        if embeddings.shape[0] != self.n_items:
            raise ValueError(f"Expected {self.n_items} embeddings, got {embeddings.shape[0]}")
        self._embeddings = embeddings.to(self._device)

    def _compute_item_params(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute item parameters from embeddings via the projection network."""
        if self._embeddings is None:
            raise RuntimeError("Call set_embeddings() before predict()")

        params = self.item_net(self._embeddings)
        difficulty = params[:, 0]
        discrimination = torch.exp(params[:, 1]) if self.pl >= 2 else None
        guessing = torch.sigmoid(params[:, 2]) if self.pl == 3 else None
        return difficulty, discrimination, guessing

    @property
    def difficulty(self) -> torch.Tensor:
        """Predicted item difficulties from embeddings."""
        d, _, _ = self._compute_item_params()
        return d.detach()

    @property
    def discrimination(self) -> torch.Tensor | None:
        """Predicted item discriminations from embeddings (2PL/3PL only)."""
        _, a, _ = self._compute_item_params()
        return a.detach() if a is not None else None

    @property
    def guessing(self) -> torch.Tensor | None:
        """Predicted item guessing parameters from embeddings (3PL only)."""
        _, _, c = self._compute_item_params()
        return c.detach() if c is not None else None

    def predict(self) -> torch.Tensor:
        """Compute response probabilities using amortized item parameters.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        difficulty, discrimination, guessing = self._compute_item_params()
        return self._irt_probability(self.ability, difficulty, discrimination, guessing)

    def fit(
        self,
        response_matrix: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Fit the amortized IRT model.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Binary response matrix (n_subjects, n_items).
        embeddings : torch.Tensor
            Item embeddings (n_items, embedding_dim).
        mask : torch.Tensor | None
            Boolean mask for observed entries.
        max_epochs : int
            Maximum training epochs.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay for Adam optimizer.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history.
        """
        self.set_embeddings(embeddings)

        from torch_measure.fitting.mle import mle_fit

        response_matrix = response_matrix.to(self._device)
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)

        return mle_fit(
            self,
            response_matrix,
            mask,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            **kwargs,
        )
