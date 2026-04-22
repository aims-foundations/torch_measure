# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Ising model (pairwise Markov Random Field) for binary response data.

In network psychometrics the Ising model is the canonical tool for binary
item response data (Epskamp & Fried, 2018; van Borkulo et al., 2014).
Items are nodes in an undirected graph; edges encode pairwise conditional
dependencies after controlling for all other items.

The joint probability is

    P(x) ∝ exp( τᵀx + xᵀΘx )

where τ ∈ ℝⁿ are node thresholds and Θ ∈ ℝⁿˣⁿ is the symmetric weight
matrix (zero diagonal).

Because the partition function is intractable for large n, parameters are
estimated via **Maximum Pseudo-Likelihood (MPLE)**:

    L = Σᵢ Σⱼ log P(Xᵢⱼ = xᵢⱼ | Xᵢ,₋ⱼ)

where each conditional P(Xᵢⱼ | Xᵢ,₋ⱼ) = sigmoid(τⱼ + Σₖ≠ⱼ Θⱼₖ Xᵢₖ).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from torch_measure.models._network_base import NetworkModel


class IsingModel(NetworkModel):
    """Ising model for binary response data.

    Estimates the pairwise conditional dependence structure between items
    via Maximum Pseudo-Likelihood Estimation (MPLE).

    Parameters
    ----------
    n_items : int
        Number of items (nodes in the network).
    device : str
        Device to place parameters on.

    Attributes
    ----------
    thresholds : nn.Parameter
        Node threshold parameters τ, shape (n_items,).
    adjacency : torch.Tensor
        Estimated symmetric edge weight matrix Θ, shape (n_items, n_items),
        zero diagonal.

    Examples
    --------
    >>> model = IsingModel(n_items=10)
    >>> history = model.fit(binary_responses, max_epochs=500, verbose=False)
    >>> W = model.adjacency          # (10, 10) edge weights
    >>> s = model.centrality("strength")  # (10,) strength centrality

    References
    ----------
    .. [1] Epskamp, S., & Fried, E. I. (2018). A tutorial on regularized
           partial correlation networks. *Psychological Methods*, 23(4), 617.
    .. [2] van Borkulo, C. D., et al. (2014). A new method for constructing
           networks from binary data. *Scientific Reports*, 4, 5918.
    """

    def __init__(self, n_items: int, device: str = "cpu") -> None:
        super().__init__(n_items, device)
        self.thresholds = nn.Parameter(torch.zeros(n_items, device=self._device))
        # Full (n_items, n_items) raw weight matrix; symmetrised on the fly.
        # Initialise near zero so the network starts sparse.
        self._weights_raw = nn.Parameter(torch.zeros(n_items, n_items, device=self._device))

    def _get_weights(self) -> torch.Tensor:
        """Symmetric weight matrix with zero diagonal (gradient-tracked)."""
        W = (self._weights_raw + self._weights_raw.T) / 2.0
        # Zero diagonal via masking (keeps gradient)
        diag_mask = torch.eye(self._n_items, dtype=torch.bool, device=self._device)
        return W.masked_fill(diag_mask, 0.0)

    @property
    def adjacency(self) -> torch.Tensor:
        """Symmetric edge weight matrix (n_items, n_items), zero diagonal."""
        return self._get_weights().detach()

    def conditional_probs(self, response_matrix: torch.Tensor) -> torch.Tensor:
        """Compute item-conditional response probabilities given all other items.

        For each subject i and item j:
            P(Xᵢⱼ = 1 | Xᵢ,₋ⱼ) = sigmoid(τⱼ + Σₖ≠ⱼ Θⱼₖ Xᵢₖ)

        Parameters
        ----------
        response_matrix : torch.Tensor
            Binary response matrix (n_subjects, n_items).

        Returns
        -------
        torch.Tensor
            Conditional probabilities (n_subjects, n_items).
        """
        X = response_matrix.to(self._device).float()
        W = self._get_weights()
        logit = X @ W + self.thresholds.unsqueeze(0)  # (n_subjects, n_items)
        return torch.sigmoid(logit)

    def fit(
        self,
        response_matrix: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
        convergence_tol: float = 1e-6,
        **kwargs,
    ) -> dict:
        """Fit the Ising model via Maximum Pseudo-Likelihood Estimation.

        Minimises the summed binary cross-entropy of each item given all
        other items across all observed (subject, item) pairs.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Binary response matrix (n_subjects, n_items). NaN or -1 for missing.
        mask : torch.Tensor | None
            Boolean mask (n_subjects, n_items). Inferred if None.
        max_epochs : int
            Maximum optimisation epochs.
        lr : float
            Adam learning rate.
        verbose : bool
            Show tqdm progress bar.
        convergence_tol : float
            Stop early if |Δloss| < tol.

        Returns
        -------
        dict
            ``{"losses": [float, ...]}``.
        """
        response_matrix = response_matrix.to(self._device)
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)

        # Replace missing values with 0 for conditioning (excluded from loss)
        X = response_matrix.float().clone()
        X[~mask] = 0.0

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: dict[str, list] = {"losses": []}

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Ising MPLE")
            except ImportError:
                pass

        prev_loss = float("inf")

        for _epoch in iterator:
            optimizer.zero_grad()

            W = self._get_weights()  # (n_items, n_items)
            # logit[i,j] = τ[j] + Σ_k W[k,j] * X[i,k]  (W has zero diagonal)
            logit = X @ W + self.thresholds.unsqueeze(0)  # (n_subjects, n_items)

            loss = F.binary_cross_entropy_with_logits(logit[mask], X[mask], reduction="mean")
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history["losses"].append(loss_val)

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{loss_val:.6f}"})

            if abs(prev_loss - loss_val) < convergence_tol:
                break
            prev_loss = loss_val

        return history
