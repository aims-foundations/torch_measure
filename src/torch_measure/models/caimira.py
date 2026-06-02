# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Content-Aware Item Response Analysis (CAIMIRA).

A content-aware multidimensional IRT model that predicts per-item relevance
and difficulty vectors from question embeddings, complementing
:class:`~torch_measure.models.AmortizedIRT` (which predicts scalar 1PL/2PL/3PL
parameters) with a multidimensional MIRT-style response equation.

Reference
---------
Gor, M., Daumé III, H., Zhou, T., & Boyd-Graber, J.
"Do great minds think alike? Investigating Human-AI Complementarity in
Question Answering with CAIMIRA." EMNLP 2024. arXiv:2410.06524.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class CAIMIRA(IRTModel):
    r"""Content-Aware Item Response Analysis (CAIMIRA).

    A clean-room, paper-faithful implementation of the CAIMIRA model. Each
    item :math:`j` is summarised by two vectors derived linearly from its
    question embedding :math:`e_j \in \mathbb{R}^D`:

    * a **relevance** vector :math:`r_j \in \Delta^{K-1}` over :math:`K`
      latent skill dimensions, normalised via a softmax;
    * a **difficulty** vector :math:`d_j \in \mathbb{R}^K`, zero-centred
      across the training item bank.

    Each subject :math:`i` has a learned **skill** vector
    :math:`s_i \in \mathbb{R}^K`. The response probability is the
    paper's dot-product form:

    .. math::

        P(Y_{ij} = 1 \mid s_i, d_j, r_j) =
        \sigma\!\bigl( (s_i - d_j)^\top r_j \bigr).

    The fitting objective is Bernoulli negative log-likelihood plus L1
    regularisation on the centred difficulty matrix and on the subject
    skill matrix, propagated through :func:`~torch_measure.fitting.mle.mle_fit`
    via its ``loss_fn`` hook.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the training universe.
    n_items : int
        Number of items in the training universe (must match the first
        dimension of the embeddings passed to :meth:`set_embeddings`).
    embedding_dim : int
        Dimensionality of the question-embedding vectors.
    latent_dim : int, optional
        Number of latent skill dimensions :math:`K`. Defaults to 5, the
        value used in the CAIMIRA paper's reported experiments; the paper
        does not declare 5 as a universal default and tuning per task may
        help.
    device : str, optional
        Device for parameters and buffers. Defaults to ``"cpu"``.

    Notes
    -----
    **Bias asymmetry.** The relevance head is :class:`~torch.nn.Linear`
    with bias (the softmax normaliser absorbs constant shifts only along
    the latent axis, not the bias term). The difficulty head is
    :class:`~torch.nn.Linear` *without* bias, because any constant shift
    would be subtracted out by the zero-centering step.

    **Zero-centering protocol.** The paper specifies that difficulty is
    zero-centred over the training item bank but does not specify whether
    the centering mean is recomputed every batch / epoch or held fixed.
    This implementation:

    * uses a **dynamic** mean (recomputed each forward pass, with
      gradient flowing through the centering operation) while
      :attr:`training` is ``True``; and
    * uses a **frozen** snapshot of the training-bank mean (stored in
      the buffer :attr:`_difficulty_mean`) while :attr:`training` is
      ``False``, so cold-start items use the learned coordinate system.

    The snapshot is refreshed automatically at the end of :meth:`fit`.
    Callers handling cold-start items at inference should use
    :meth:`compute_item_params` directly with the new embeddings.

    Examples
    --------
    >>> import torch
    >>> from torch_measure.models import CAIMIRA
    >>> torch.manual_seed(0)
    >>> n_subjects, n_items, embed_dim = 20, 30, 16
    >>> model = CAIMIRA(n_subjects, n_items, embed_dim, latent_dim=3)
    >>> embeddings = torch.randn(n_items, embed_dim)
    >>> responses = torch.bernoulli(torch.full((n_subjects, n_items), 0.5))
    >>> history = model.fit(responses, embeddings, max_epochs=10, verbose=False)
    >>> # Cold-start: predict on new items with their embeddings
    >>> new_embeddings = torch.randn(5, embed_dim)
    >>> relevance, difficulty = model.compute_item_params(new_embeddings)
    >>> relevance.shape, difficulty.shape
    (torch.Size([5, 3]), torch.Size([5, 3]))
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        embedding_dim: int,
        latent_dim: int = 5,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # Subject skill vectors, learned directly (small init for stable training).
        self.skill = nn.Parameter(torch.randn(n_subjects, latent_dim, device=self._device) * 0.01)

        # Linear heads from question embeddings to (relevance_raw, difficulty_raw).
        # Bias asymmetry: relevance has bias (informative under softmax);
        # difficulty has no bias (would be subtracted out by zero-centering).
        self.relevance_head = nn.Linear(embedding_dim, latent_dim, bias=True).to(self._device)
        self.difficulty_head = nn.Linear(embedding_dim, latent_dim, bias=False).to(self._device)

        # Training-bank difficulty mean, refreshed by fit() and used in
        # frozen-mean centering for cold-start inference.
        self.register_buffer("_difficulty_mean", torch.zeros(latent_dim, device=self._device))

        self._embeddings: torch.Tensor | None = None

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        """Set the training-bank item embeddings used by :meth:`predict`.

        Parameters
        ----------
        embeddings : torch.Tensor
            Item embeddings of shape ``(n_items, embedding_dim)``.

        Raises
        ------
        ValueError
            If ``embeddings`` does not have shape ``(n_items, embedding_dim)``.
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2-D embeddings, got shape {tuple(embeddings.shape)}")
        if embeddings.shape[0] != self.n_items:
            raise ValueError(f"Expected {self.n_items} embeddings, got {embeddings.shape[0]}")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[1]}")
        self._embeddings = embeddings.to(self._parameter_device())

    def _parameter_device(self) -> torch.device:
        """Return the current module device from parameters, not constructor state."""
        return self.skill.device

    def compute_item_params(
        self,
        embeddings: torch.Tensor | None = None,
        center: Literal["auto", "dynamic", "frozen"] = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(relevance, difficulty)`` for a batch of item embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor or None
            Embeddings of shape ``(n, embedding_dim)``. If ``None``, uses
            ``self._embeddings`` (the training bank set by
            :meth:`set_embeddings`).
        center : {"auto", "dynamic", "frozen"}
            Difficulty-centering mode.

            * ``"dynamic"`` subtracts the mean of ``d_raw`` over the supplied
              ``embeddings`` (gradient flows through the mean — used during
              training).
            * ``"frozen"`` subtracts the stored :attr:`_difficulty_mean`
              buffer (the snapshot taken at the end of the last
              :meth:`fit` call — used for cold-start inference).
            * ``"auto"`` resolves to ``"dynamic"`` when :attr:`training`
              is ``True``, ``"frozen"`` otherwise.

        Returns
        -------
        relevance : torch.Tensor
            Shape ``(n, latent_dim)``, rows on the simplex (softmax over
            the latent axis).
        difficulty : torch.Tensor
            Shape ``(n, latent_dim)``, zero-centred per ``center``.
        """
        if embeddings is None:
            if self._embeddings is None:
                raise RuntimeError("Call set_embeddings() before computing item params.")
            embeddings = self._embeddings
        embeddings = embeddings.to(self._parameter_device())

        relevance = torch.softmax(self.relevance_head(embeddings), dim=-1)
        d_raw = self.difficulty_head(embeddings)

        if center == "auto":
            center = "dynamic" if self.training else "frozen"

        if center == "dynamic":
            difficulty = d_raw - d_raw.mean(dim=0, keepdim=True)
        elif center == "frozen":
            difficulty = d_raw - self._difficulty_mean
        else:
            raise ValueError(f"Unknown center mode {center!r}; expected 'auto', 'dynamic', or 'frozen'.")

        return relevance, difficulty

    def predict_embeddings(
        self,
        subject_idx: torch.Tensor,
        item_embeddings: torch.Tensor,
        center: str = "frozen",
    ) -> torch.Tensor:
        r"""Predict probabilities for paired subject indices and item embeddings.

        This is the cold-start item path: callers supply arbitrary item
        embeddings directly instead of indices into the training bank. The
        training-bank embeddings set by :meth:`set_embeddings` are not read or
        mutated.

        Parameters
        ----------
        subject_idx : torch.Tensor
            1-D subject indices of shape ``(N,)``.
        item_embeddings : torch.Tensor
            2-D item embeddings of shape ``(N, embedding_dim)``.
        center : {"auto", "dynamic", "frozen"}, optional
            Difficulty-centering mode. Defaults to ``"frozen"`` for
            cold-start inference.

        Returns
        -------
        torch.Tensor
            Probabilities, shape ``(N,)``.
        """
        if subject_idx.dim() != 1:
            raise ValueError(f"Expected 1-D subject_idx, got shape {tuple(subject_idx.shape)}")
        if item_embeddings.dim() != 2:
            raise ValueError(f"Expected 2-D item_embeddings, got shape {tuple(item_embeddings.shape)}")
        if item_embeddings.shape[0] != subject_idx.shape[0]:
            raise ValueError(
                "subject_idx and item_embeddings must have matching first dimension "
                f"({subject_idx.shape[0]} != {item_embeddings.shape[0]})"
            )
        if item_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {item_embeddings.shape[1]}")

        device = self._parameter_device()
        subject_idx = subject_idx.to(device=device, dtype=torch.long)
        relevance, difficulty = self.compute_item_params(item_embeddings, center=center)
        diff = self.skill[subject_idx] - difficulty
        logit = (diff * relevance).sum(dim=-1)
        return torch.sigmoid(logit)

    def _refresh_difficulty_mean(self) -> None:
        """Snapshot the current training-bank difficulty mean into the buffer.

        ``self._embeddings`` is a plain Python attribute (not a registered
        buffer), so ``model.to(device)`` does not carry it along. Without
        the explicit ``.to(self._parameter_device())`` below, calling this
        method after a ``model.to(...)`` would raise a cross-device
        ``RuntimeError`` because ``difficulty_head`` lives on the new device
        while ``self._embeddings`` still lives on the original one.
        """
        if self._embeddings is None:
            raise RuntimeError("Cannot snapshot difficulty mean before set_embeddings().")
        with torch.no_grad():
            embeddings = self._embeddings.to(self._parameter_device())
            d_raw = self.difficulty_head(embeddings)
            self._difficulty_mean.copy_(d_raw.mean(dim=0))

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Compute :math:`P(\text{correct})` per row of ``query``.

        Applies the CAIMIRA response equation
        :math:`\sigma((s_i - d_j)^\top r_j)` row-wise.

        Parameters
        ----------
        query : dict[str, torch.Tensor]
            Must contain ``"subject_idx"`` and ``"item_idx"`` — 1-D long
            tensors of equal length ``N``. Item indices reference the
            training bank set via :meth:`set_embeddings`.

        Returns
        -------
        torch.Tensor
            Probabilities, shape ``(N,)``.
        """
        device = self._parameter_device()
        s = query["subject_idx"].to(device=device, dtype=torch.long)
        i = query["item_idx"].to(device=device, dtype=torch.long)
        relevance, difficulty = self.compute_item_params()
        diff = self.skill[s] - difficulty[i]
        logit = (diff * relevance[i]).sum(dim=-1)
        return torch.sigmoid(logit)

    def fit(
        self,
        data,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        difficulty_reg: float = 1e-4,
        skill_reg: float = 1e-4,
        verbose: bool = True,
        method: str = "mle",
        **kwargs,
    ) -> dict:
        r"""Fit CAIMIRA by MLE with L1 regularisation on difficulty and skill.

        Minimises

        .. math::

            L = \mathrm{BernoulliNLL}(p, y)
              + \lambda_d \,\| d \|_1
              + \lambda_s \,\| s \|_1,

        where :math:`d` is the zero-centred per-item difficulty matrix
        evaluated over the full training item bank and :math:`s` is the
        subject skill matrix. Penalties use the mean absolute value so
        coefficients are dimension-invariant. Reuses the package's
        canonical :func:`~torch_measure.fitting.mle.mle_fit` loop via its
        ``loss_fn`` hook.

        Parameters
        ----------
        data : LongFormData or torch.Tensor
            Long-form dataset (preferred) or wide-form response tensor.
        embeddings : torch.Tensor
            Item embeddings of shape ``(n_items, embedding_dim)``.
        mask : torch.Tensor or None
            Boolean mask for observed entries when ``data`` is wide-form.
        max_epochs : int
            Maximum optimization epochs.
        lr : float
            Adam learning rate.
        weight_decay : float
            Adam weight decay (L2 on all parameters); orthogonal to the
            CAIMIRA L1 penalties.
        difficulty_reg : float
            L1 coefficient :math:`\lambda_d` on the centred difficulty.
        skill_reg : float
            L1 coefficient :math:`\lambda_s` on the subject skill matrix.
        verbose : bool
            Show a progress bar if ``tqdm`` is installed.
        method : str, optional
            Training algorithm. Only ``"mle"`` (the paper's MLE training,
            using :func:`~torch_measure.fitting.mle.mle_fit` under the
            hood) is supported. Any other value raises ``ValueError``.
            Defaults to ``"mle"``.
        **kwargs
            Forwarded to :func:`~torch_measure.fitting.mle.mle_fit`
            (e.g. ``optimizer_cls``, ``convergence_tol``).

        Returns
        -------
        dict
            Training history with ``"losses"`` key.

        Raises
        ------
        ValueError
            If ``method`` is not ``"mle"``. CAIMIRA does not implement EM
            or other estimators; the explicit guard prevents silent fall-
            through of an unknown ``method`` value into ``**kwargs``.
        """
        if method != "mle":
            raise ValueError(
                f"CAIMIRA.fit only supports method='mle' (the paper's MLE training). Got method={method!r}."
            )

        from torch_measure.fitting._losses import bernoulli_nll
        from torch_measure.fitting.mle import mle_fit

        self.set_embeddings(embeddings)
        self.train()
        # Seed the buffer so any post-init `predict` calls before training
        # finishes do not fall back to a stale zero mean.
        self._refresh_difficulty_mean()

        subject_idx, item_idx, response = self._normalize_fit_inputs(data, mask)
        device = self._parameter_device()
        subject_idx = subject_idx.to(device=device, dtype=torch.long)
        item_idx = item_idx.to(device=device, dtype=torch.long)
        response = response.to(device=device)

        def caimira_loss(probs: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
            base = bernoulli_nll(probs, observed)
            # Re-compute centered difficulty for the reg term so gradients
            # match the dynamic centering used in the forward pass.
            d_raw = self.difficulty_head(self._embeddings)
            d_centered = d_raw - d_raw.mean(dim=0, keepdim=True)
            reg = difficulty_reg * d_centered.abs().mean() + skill_reg * self.skill.abs().mean()
            return base + reg

        history = mle_fit(
            self,
            subject_idx,
            item_idx,
            response,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            loss_fn=caimira_loss,
            **kwargs,
        )

        self.eval()
        # Final snapshot so cold-start inference uses the learned coordinate system.
        self._refresh_difficulty_mean()
        return history
