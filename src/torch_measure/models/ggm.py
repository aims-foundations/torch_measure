# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Gaussian Graphical Model (GGM) for continuous response data.

In network psychometrics the GGM is the standard model for continuous or
polytomous (after polychoric correlation) item response data. Items are
modelled as a multivariate Gaussian:

    X ~ N(μ, Σ),   K = Σ⁻¹  (precision matrix)

The off-diagonal elements of K encode *partial correlations* between items
— the edge weights in the network — after controlling for all other items.

Fitting uses the **GraphicalLasso** objective (Friedman et al., 2008):

    L(K) = −log det K + tr(S K) + λ · Σᵢ≠ⱼ |Kᵢⱼ|

where S is the sample covariance matrix and λ controls sparsity. The
precision matrix is constrained to be positive definite throughout via a
Cholesky parameterisation (K = LLᵀ, L lower-triangular with positive
diagonal), making the objective fully differentiable and suitable for
gradient-based optimisation.

References
----------
.. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
       covariance estimation with the graphical lasso. *Biostatistics*, 9(3).
.. [2] Epskamp, S., & Fried, E. I. (2018). A tutorial on regularized partial
       correlation networks. *Psychological Methods*, 23(4), 617.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from torch_measure.models._network_base import NetworkModel


class GaussianGraphicalModel(NetworkModel):
    """Gaussian Graphical Model for continuous response data.

    Estimates a sparse precision matrix K via the GraphicalLasso objective,
    optimised with Adam using a Cholesky parameterisation to ensure K remains
    positive definite.

    Parameters
    ----------
    n_items : int
        Number of items (nodes in the network).
    lam : float
        L1 regularisation strength on off-diagonal precision entries.
        Larger values produce sparser networks.
    device : str
        Device to place parameters on.

    Attributes
    ----------
    precision : torch.Tensor
        Estimated precision matrix K = LLᵀ, shape (n_items, n_items).
    partial_correlations : torch.Tensor
        Partial correlation matrix, shape (n_items, n_items). Diagonal is 1.
    adjacency : torch.Tensor
        Partial correlations with zero diagonal (edge weights).

    Examples
    --------
    >>> model = GaussianGraphicalModel(n_items=10, lam=0.1)
    >>> history = model.fit(continuous_responses, max_epochs=500, verbose=False)
    >>> pcor = model.partial_correlations   # (10, 10)
    >>> s = model.centrality("strength")    # (10,) strength centrality

    References
    ----------
    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
           covariance estimation with the graphical lasso. *Biostatistics*, 9(3).
    """

    def __init__(self, n_items: int, lam: float = 0.1, device: str = "cpu") -> None:
        super().__init__(n_items, device)
        self.lam = lam
        # Cholesky factor L: lower-triangular, positive diagonal.
        # K = L @ L.T. Initialise to K = I (L = I).
        # We store the full matrix and apply tril + softplus-on-diagonal.
        self._L_raw = nn.Parameter(torch.eye(n_items, device=self._device))

    def _build_cholesky(self) -> torch.Tensor:
        """Construct the lower-triangular Cholesky factor with positive diagonal.

        Takes the lower triangle of ``_L_raw`` and replaces the diagonal with
        its softplus transform, guaranteeing K = LLᵀ ≻ 0.
        """
        L = torch.tril(self._L_raw)
        diag_pos = F.softplus(L.diagonal())  # (n_items,)
        # Replace diagonal: subtract old, add new (gradient-friendly)
        L = L - torch.diag(L.diagonal()) + torch.diag(diag_pos)
        return L

    @property
    def precision(self) -> torch.Tensor:
        """Estimated precision matrix K = LLᵀ, shape (n_items, n_items)."""
        L = self._build_cholesky()
        return (L @ L.T).detach()

    @property
    def partial_correlations(self) -> torch.Tensor:
        """Partial correlation matrix derived from the precision matrix.

        pcorᵢⱼ = −Kᵢⱼ / √(Kᵢᵢ · Kⱼⱼ)   for i ≠ j,   1 on the diagonal.

        Returns
        -------
        torch.Tensor
            Shape (n_items, n_items). Values in [−1, 1].
        """
        K = self.precision
        diag_inv_sqrt = 1.0 / K.diagonal().clamp(min=1e-10).sqrt()
        # pcor_ij = -K_ij / sqrt(K_ii * K_jj)
        pcor = -K * diag_inv_sqrt.unsqueeze(0) * diag_inv_sqrt.unsqueeze(1)
        pcor.fill_diagonal_(1.0)
        return pcor

    @property
    def adjacency(self) -> torch.Tensor:
        """Partial correlations with zero diagonal (edge weight matrix)."""
        pcor = self.partial_correlations
        pcor.fill_diagonal_(0.0)
        return pcor

    @staticmethod
    def _sample_covariance(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute sample covariance with mean imputation for missing values.

        Parameters
        ----------
        X : torch.Tensor
            Response matrix (n_subjects, n_items), float.
        mask : torch.Tensor
            Boolean mask of observed entries.

        Returns
        -------
        torch.Tensor
            Sample covariance matrix (n_items, n_items).
        """
        n_subjects, n_items = X.shape
        X_imp = X.clone()

        # Mean imputation per column
        for j in range(n_items):
            col_mask = mask[:, j]
            if col_mask.any():
                col_mean = X[col_mask, j].mean()
                X_imp[~col_mask, j] = col_mean

        # Centre each column
        X_c = X_imp - X_imp.mean(dim=0, keepdim=True)

        # Biased sample covariance (sufficient for optimisation)
        S = X_c.T @ X_c / n_subjects
        return S

    def fit(
        self,
        data,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 0.01,
        lam: float | None = None,
        verbose: bool = True,
        convergence_tol: float = 1e-6,
        **kwargs,
    ) -> dict:
        """Fit the GGM via the GraphicalLasso objective.

        Minimises ``−log det K + tr(S K) + λ · Σᵢ≠ⱼ |Kᵢⱼ|`` with K
        constrained to be positive definite via Cholesky parameterisation.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Long-form dataset (preferred) or wide-form continuous response
            tensor of shape ``(n_subjects, n_items)``. NaN or -1 marks missing.
        mask : torch.Tensor | None
            Only used with wide-form input — boolean mask. Inferred from
            NaNs if None.
        max_epochs : int
            Maximum optimisation epochs.
        lr : float
            Adam learning rate.
        lam : float | None
            Override the instance-level L1 regularisation strength.
        verbose : bool
            Show tqdm progress bar.
        convergence_tol : float
            Stop early if |Δloss| < tol.

        Returns
        -------
        dict
            ``{"losses": [float, ...]}``.
        """
        if lam is None:
            lam = self.lam

        response_matrix, mask = self._normalize_fit_inputs_to_matrix(data, mask)

        X = response_matrix.float()
        S = self._sample_covariance(X, mask).to(self._device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: dict[str, list] = {"losses": []}

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="GGM fitting")
            except ImportError:
                pass

        prev_loss = float("inf")

        for _epoch in iterator:
            optimizer.zero_grad()

            L = self._build_cholesky()
            K = L @ L.T  # (n_items, n_items) positive definite

            # log det(K) = 2 · Σ log(diag(L))
            log_det_K = 2.0 * L.diagonal().log().sum()

            # tr(S K)
            trace_SK = (S @ K).diagonal().sum()

            # L1 penalty on off-diagonal elements of K
            diag_mask = torch.eye(self._n_items, dtype=torch.bool, device=self._device)
            K_off = K.masked_fill(diag_mask, 0.0)
            l1_penalty = K_off.abs().sum()

            loss = -log_det_K + trace_SK + lam * l1_penalty
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history["losses"].append(loss_val)

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{loss_val:.4f}"})

            if abs(prev_loss - loss_val) < convergence_tol:
                break
            prev_loss = loss_val

        return history
