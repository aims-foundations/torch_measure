# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Amortized IRT model that predicts item parameters from embeddings.

Consolidated from agent-eval/model/amortized_irt.py and predictive-eval/train/amortized_irt/irt.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from torch_measure.models._base import IRTModel
from torch_measure.models._network import MLP
from torch_measure.models._predictor import Predictor


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

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute P(correct) at query rows using amortized item parameters."""
        s = query["subject_idx"]
        i = query["item_idx"]
        difficulty, discrimination, guessing = self._compute_item_params()
        return self._irt_probability(
            self.ability[s],
            difficulty[i],
            discrimination=discrimination[i] if discrimination is not None else None,
            guessing=guessing[i] if guessing is not None else None,
        )

    def fit(
        self,
        data,
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
        data : LongFormData | torch.Tensor
            Long-form dataset (preferred) or wide-form response tensor.
        embeddings : torch.Tensor
            Item embeddings ``(n_items, embedding_dim)``.
        mask : torch.Tensor | None
            Boolean mask for observed entries (only used with wide-form input).
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

        subject_idx, item_idx, response = self._normalize_fit_inputs(data, mask)
        return mle_fit(
            self,
            subject_idx,
            item_idx,
            response,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            **kwargs,
        )


class ARAF(Predictor):
    """Amortized Response/Agent Factor model with ARD-induced sparsity.

    A K-dimensional factor model where item loadings are amortized from
    pre-computed item embeddings via a learned projection matrix, and the
    effective dimensionality of the latent space is discovered automatically
    by an Automatic Relevance Determination (ARD) gate on the loadings::

        a_j   = (x_j @ W_normᵀ) ⊙ ReLU(τ)              (J, K)
        b_j   = Linear(x_j)                              (J,)
        μ_ij  = σ(θ_iᵀ a_j + b_j + θ_bias_i + g)         per cell

    Used by ``agent-eval`` to predict held-out agent performance on a
    benchmark from a partially observed (agents × tasks) score matrix and
    pre-computed item embeddings. Supports both Bernoulli (binary) and Beta
    (continuous-in-(0,1)) likelihoods at fit time.

    Forward computes the full ``(n_subjects, n_items)`` probability matrix
    in one shot. ``predict(query)`` gathers the queried cells from that
    matrix to satisfy the standard :class:`Predictor` per-row contract;
    use :meth:`dense_predict` when the full matrix is what you want.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (agents / models / test-takers).
    n_items : int
        Number of items (tasks / benchmark questions).
    embedding_dim : int
        Dimension ``d`` of the item embeddings supplied via
        :meth:`set_embeddings`.
    latent_dim : int
        Latent factor dimension ``K``. Set generously; ARD prunes inactive
        factors at fit time. Default 30.
    dropout : float
        Dropout applied to item loadings during training. Default 0.7.
    use_ard : bool
        If ``False``, disables the ARD gate (τ is fixed to ones). Useful
        for ablations. Default ``True``.
    device : str
        Device to place parameters on.

    References
    ----------
    .. [1] agent-eval research repository
       (https://github.com/aims-foundations/agent-eval).

    See Also
    --------
    AmortizedIRT : Scalar-ability amortized 1PL/2PL/3PL model.
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        embedding_dim: int,
        latent_dim: int = 30,
        dropout: float = 0.7,
        use_ard: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_ard = use_ard

        self.theta = nn.Parameter(torch.randn(n_subjects, latent_dim, device=self._device) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(n_subjects, device=self._device))
        self.global_bias = nn.Parameter(torch.zeros(1, device=self._device))
        self.W = nn.Parameter(torch.randn(latent_dim, embedding_dim, device=self._device) * 0.01)
        self.tau_raw = nn.Parameter(torch.full((latent_dim,), 0.5, device=self._device))
        self.difficulty_proj = nn.Linear(embedding_dim, 1).to(self._device)

        self.register_buffer("_x_j", torch.empty(0, device=self._device))

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        """Register item embeddings used to amortize loadings and difficulty.

        Parameters
        ----------
        embeddings : torch.Tensor
            Item embeddings, shape ``(n_items, embedding_dim)``.
        """
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {tuple(embeddings.shape)}")
        if embeddings.shape[0] != self.n_items:
            raise ValueError(f"Expected {self.n_items} embeddings, got {embeddings.shape[0]}")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: model was built with embedding_dim={self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        self._x_j = embeddings.to(self._device).float()

    def get_tau(self) -> torch.Tensor:
        """Non-negative ARD scales (``ReLU(τ_raw)``); ones if ARD is disabled."""
        if not self.use_ard:
            return torch.ones_like(self.tau_raw)
        return F.relu(self.tau_raw)

    def _check_embeddings(self) -> torch.Tensor:
        if self._x_j.numel() == 0:
            raise RuntimeError("Call set_embeddings() before forward()/predict()")
        return self._x_j

    def forward(self, query: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Return the full ``(n_subjects, n_items)`` probability matrix, or gather a query.

        With no argument, computes the dense response-probability matrix used
        by the ARAF training loop. With a long-form ``query`` dict (per the
        :class:`Predictor` contract), gathers the cells at
        ``(query["subject_idx"], query["item_idx"])``.
        """
        if query is None:
            return self.dense_predict()
        return self.predict(query)

    def dense_predict(self) -> torch.Tensor:
        """Compute the full ``(n_subjects, n_items)`` probability matrix.

        Returns
        -------
        torch.Tensor
            Probability matrix, shape ``(n_subjects, n_items)``.
        """
        x_j = self._check_embeddings()

        W_norm = F.normalize(self.W, dim=1)
        base_loadings = x_j @ W_norm.T
        a_j = base_loadings * self.get_tau().unsqueeze(0)

        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)

        diff = self.difficulty_proj(x_j).squeeze(-1)
        logits = (
            self.theta @ a_j.T
            + diff.unsqueeze(0)
            + self.theta_bias.unsqueeze(1)
            + self.global_bias
        )
        return torch.sigmoid(logits)

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute P(correct) at the queried (subject, item) rows."""
        s = query["subject_idx"]
        i = query["item_idx"]
        probs = self.dense_predict()
        return probs[s, i]

    @property
    def loadings(self) -> torch.Tensor:
        """Amortized item loadings ``a_j`` of shape ``(n_items, latent_dim)``."""
        x_j = self._check_embeddings()
        with torch.no_grad():
            W_norm = F.normalize(self.W, dim=1)
            return (x_j @ W_norm.T) * self.get_tau().unsqueeze(0)

    @property
    def difficulty(self) -> torch.Tensor:
        """Per-item difficulty ``b_j`` of shape ``(n_items,)``."""
        x_j = self._check_embeddings()
        with torch.no_grad():
            return self.difficulty_proj(x_j).squeeze(-1)

    @property
    def tau(self) -> torch.Tensor:
        """Current ARD scales (``ReLU(τ_raw)``), shape ``(latent_dim,)``."""
        with torch.no_grad():
            return self.get_tau().clone()

    @property
    def active_dims(self) -> torch.Tensor:
        """Indices of latent dims surviving ARD pruning (``τ > 1e-3``)."""
        with torch.no_grad():
            return torch.where(self.get_tau() > 1e-3)[0]

    def fit(
        self,
        data,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        likelihood: str = "bernoulli",
        beta_phi: float = 10.0,
        epochs: int = 1000,
        lambda_tau: float = 1.38,
        row_weights: torch.Tensor | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Fit ARAF with the ARD-regularised training loop.

        Thin wrapper around :func:`torch_measure.fitting.araf.araf_fit` that
        registers ``embeddings`` and forwards the response matrix and mask.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Long-form dataset or wide-form ``(n_subjects, n_items)`` tensor.
        embeddings : torch.Tensor
            Item embeddings, shape ``(n_items, embedding_dim)``.
        mask : torch.Tensor | None
            Boolean mask of observed cells (wide-form only). Inferred from
            NaN/-1 if omitted.
        likelihood : str
            ``"bernoulli"`` (binary responses) or ``"beta"`` (continuous in
            ``(0, 1)``).
        beta_phi : float
            Beta precision parameter (used when ``likelihood="beta"``).
        epochs : int
            Number of training epochs.
        lambda_tau : float
            Final L1 weight on τ after the warmup/ramp schedule.
        row_weights : torch.Tensor | None
            Optional per-subject sample weights, shape ``(n_subjects,)``.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history.
        """
        from torch_measure.fitting.araf import araf_fit

        self.set_embeddings(embeddings)

        response_matrix, observed_mask = self._normalize_to_matrix(data, mask)
        return araf_fit(
            self,
            response_matrix,
            observed_mask,
            likelihood=likelihood,
            beta_phi=beta_phi,
            epochs=epochs,
            lambda_tau=lambda_tau,
            row_weights=row_weights,
            verbose=verbose,
            **kwargs,
        )

    def adapt(
        self,
        data,
        mask: torch.Tensor | None = None,
        likelihood: str = "bernoulli",
        beta_phi: float = 10.0,
        epochs: int | None = None,
        row_weights: torch.Tensor | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Refit ``θ`` for new agents while keeping item parameters frozen.

        Used at inference time on a held-out response matrix: the projection
        ``W``, ARD ``τ``, difficulty head, and global bias are all frozen;
        only ``theta`` and ``theta_bias`` are updated.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Support responses. Must have the same ``n_items`` as the trained
            model. ``n_subjects`` may differ.
        mask : torch.Tensor | None
            Mask of observed support cells (wide-form only).
        likelihood, beta_phi : see :meth:`fit`.
        epochs : int | None
            Defaults to ``max(200, 500)``.
        row_weights : torch.Tensor | None
            Per-subject support weights.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history.
        """
        from torch_measure.fitting.araf import araf_adapt

        response_matrix, observed_mask = self._normalize_to_matrix(data, mask)
        return araf_adapt(
            self,
            response_matrix,
            observed_mask,
            likelihood=likelihood,
            beta_phi=beta_phi,
            epochs=epochs,
            row_weights=row_weights,
            verbose=verbose,
            **kwargs,
        )

    def _normalize_to_matrix(
        self,
        data,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coerce long-form / wide-form input into ``(matrix, mask)`` on device."""
        from torch_measure.datasets._long_form import LongFormData

        if isinstance(data, LongFormData):
            fit_inputs = data.to_fit_tensors(device=str(self._device))
            subject_idx = fit_inputs["subject_idx"]
            item_idx = fit_inputs["item_idx"]
            response = fit_inputs["response"].float()

            n_subjects = self._n_subjects
            n_items = self._n_items
            matrix = torch.zeros((n_subjects, n_items), dtype=torch.float32, device=self._device)
            counts = torch.zeros((n_subjects, n_items), dtype=torch.float32, device=self._device)
            matrix.index_put_((subject_idx, item_idx), response, accumulate=True)
            counts.index_put_(
                (subject_idx, item_idx),
                torch.ones_like(response, dtype=torch.float32),
                accumulate=True,
            )
            built_mask = counts > 0
            matrix[built_mask] = matrix[built_mask] / counts[built_mask]
            return matrix, built_mask

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"fit() expected LongFormData or torch.Tensor, got {type(data).__name__}")

        response_matrix = data.to(self._device).float()
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)
        response_matrix = torch.where(mask, response_matrix, torch.zeros_like(response_matrix))
        return response_matrix, mask
