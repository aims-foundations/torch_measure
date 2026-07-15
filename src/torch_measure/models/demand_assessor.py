# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Demand-based assessor: predicts P(success | subject_idx, item_features)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from torch_measure.fitting._losses import bernoulli_nll
from torch_measure.models._network import MLP
from torch_measure.models._predictor import Predictor

if TYPE_CHECKING:
    from torch_measure.datasets._long_form import LongFormData


def _to_long_form(
    data,
    mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coerce ``data`` to the long-form triple ``(subject_idx, item_idx, response)``.

    Mirrors :meth:`IRTModel._normalize_fit_inputs` as a free function.
    Returns all three tensors on ``device``.

    For :class:`~torch_measure.datasets.LongFormData` input, ``item_idx``
    values are indices into ``sorted(data.responses["item_id"].unique())``.
    The caller is responsible for aligning ``item_features`` rows to that
    ordering — use ``data.to_fit_tensors()["item_ids"]`` to retrieve it.
    """
    from torch_measure.datasets._long_form import LongFormData  # deferred — avoids circular import

    if isinstance(data, LongFormData):
        fit_inputs = data.to_fit_tensors(device=str(device))
        return fit_inputs["subject_idx"], fit_inputs["item_idx"], fit_inputs["response"]

    if not isinstance(data, torch.Tensor):
        raise TypeError(f"fit() expected LongFormData or torch.Tensor, got {type(data).__name__}")

    response_matrix = data.to(device)
    if mask is None:
        mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
    mask = mask.to(device)

    obs_indices = mask.nonzero(as_tuple=False)
    subject_idx = obs_indices[:, 0].to(device)
    item_idx = obs_indices[:, 1].to(device)
    response = response_matrix[mask].float().to(device)
    return subject_idx, item_idx, response


class DemandAssessor(Predictor):
    """Demand-based assessor for predicting AI system success from item features.

    Predicts ``P(response=1 | subject_idx, item_features)`` by concatenating a
    learned subject embedding with item feature vectors and passing the result
    through an MLP::

        P(success) = sigmoid(MLP([subject_embedding ‖ item_features]))

    Unlike IRT models, items are identified by their feature vectors rather than
    integer indices. The model generalises to items not seen during training as
    long as their feature vectors are provided — predicting on a new item requires
    no re-fitting, only its demand annotation vector.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (AI systems / models being evaluated).
    item_feature_dim : int
        Dimension of item feature vectors (demand annotations, benchmark metadata,
        or other task-level descriptors).
    subject_embedding_dim : int
        Dimension of the learned per-subject representation. Default 16.
    hidden_dim : int
        Width of the MLP hidden layer(s). Default 128.
    n_layers : int
        Total number of MLP layers (minimum 1). Default 2.
    dropout : float
        Dropout rate between MLP hidden layers. Default 0.0.
    device : str
        Device for all parameters. Default ``"cpu"``.

    Attributes
    ----------
    subject_embedding : nn.Embedding
        Learnable subject table, shape ``(n_subjects, subject_embedding_dim)``.
    net : MLP
        Maps ``[subject_embedding ‖ item_features]`` to a scalar logit.
        Input dim: ``subject_embedding_dim + item_feature_dim``. Output dim: 1.

    Notes
    -----
    This model does not require :meth:`fit` before :meth:`predict` — parameters
    are randomly initialised and produce valid (though uninformative) probabilities
    immediately. This differs from :class:`AmortizedIRT` and
    :class:`TabPFNPredictor`, which guard ``predict`` with a ``RuntimeError``
    until external state is supplied.

    :func:`predict_dense` is not applicable: it builds a Cartesian grid of integer
    ``item_idx`` values, but this model's :attr:`expected_keys` does not include
    ``"item_idx"``. To predict over a fixed item set, construct the query manually::

        n = n_subjects * n_items
        s = torch.arange(n_subjects).repeat_interleave(n_items)
        f = item_features.unsqueeze(0).expand(n_subjects, -1, -1).reshape(n, -1)
        probs = model.predict({"subject_idx": s, "item_features": f})
        dense = probs.view(n_subjects, n_items)

    This model is not compatible with :class:`~torch_measure.cat.runner.AdaptiveTester`,
    which expects IRT-style ``difficulty`` and ``discrimination`` parameters.

    When ``dropout > 0``, call ``model.eval()`` before inference to disable
    stochastic dropout. Call ``model.train()`` to re-enable dropout during
    further training.

    Examples
    --------
    >>> import torch
    >>> from torch_measure.models import DemandAssessor
    >>> model = DemandAssessor(n_subjects=50, item_feature_dim=8)
    >>> response_matrix = (torch.rand(50, 30) > 0.5).float()
    >>> item_features = torch.randn(30, 8)
    >>> history = model.fit(response_matrix, item_features, max_epochs=10, verbose=False)
    >>> query = {"subject_idx": torch.tensor([0, 1]), "item_features": torch.randn(2, 8)}
    >>> probs = model.predict(query)
    >>> probs.shape
    torch.Size([2])
    """

    expected_keys: ClassVar[tuple[str, ...]] = ("subject_idx", "item_features")

    def __init__(
        self,
        n_subjects: int,
        item_feature_dim: int,
        subject_embedding_dim: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        device: str = "cpu",
    ) -> None:
        # n_items=0: this model has no per-item parameters. Item count is
        # tracked via n_items_buf after fit() and exposed through n_items.
        super().__init__(n_subjects, n_items=0, device=device)
        self.item_feature_dim = item_feature_dim
        self.subject_embedding_dim = subject_embedding_dim
        # Scalar buffer so that n_items survives state_dict save/load.
        self.register_buffer("n_items_buf", torch.tensor(0, dtype=torch.long))
        self.subject_embedding = nn.Embedding(n_subjects, subject_embedding_dim).to(self._device)
        self.net = MLP(
            input_dim=subject_embedding_dim + item_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

    @property
    def n_items(self) -> int:
        """Number of rows in the ``item_features`` matrix supplied to the last :meth:`fit` call.

        Returns 0 before :meth:`fit`. Equal to ``item_features.shape[0]`` after
        fitting, which may exceed the number of items referenced in the training
        data if extra feature rows were supplied. Persists across ``state_dict``
        save/load via ``n_items_buf``.
        """
        return int(self.n_items_buf.item())

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute P(success) at each row of ``query``.

        Parameters
        ----------
        query : dict[str, torch.Tensor]
            Must contain:

            - ``"subject_idx"``: :class:`torch.LongTensor`, shape ``(N,)``
            - ``"item_features"``: :class:`torch.FloatTensor`, shape
              ``(N, item_feature_dim)``

        Returns
        -------
        torch.Tensor
            Predicted probabilities, shape ``(N,)``, values in ``[0, 1]``.

        Raises
        ------
        ValueError
            If the last dimension of ``item_features`` does not match
            :attr:`item_feature_dim`.
        """
        s = query["subject_idx"]
        f = query["item_features"]
        if f.ndim != 2 or f.shape[-1] != self.item_feature_dim:
            raise ValueError(
                f"item_features must be a 2-D tensor with last dim "
                f"{self.item_feature_dim}; got shape {tuple(f.shape)}"
            )
        e_s = self.subject_embedding(s)        # (N, subject_embedding_dim)
        x = torch.cat([e_s, f], dim=-1)        # (N, subject_embedding_dim + item_feature_dim)
        logit = self.net(x).squeeze(-1)         # (N,)
        return torch.sigmoid(logit)

    def fit(
        self,
        data: LongFormData | torch.Tensor,
        item_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        convergence_tol: float = 1e-6,
        verbose: bool = True,
    ) -> dict:
        """Fit the model on observed responses.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Either a :class:`~torch_measure.datasets.LongFormData` or a wide-form
            response tensor of shape ``(n_subjects, n_items)``. For wide-form,
            missing entries may be encoded as ``NaN`` or ``-1``.
        item_features : torch.Tensor
            Item feature matrix of shape ``(n_items, item_feature_dim)``.

            .. warning::

                Row ordering must match the item ordering used by ``item_idx``:

                - **Wide-form tensor**: row ``i`` corresponds to column ``i`` of
                  the response matrix.
                - **LongFormData**: row ``i`` must correspond to the ``i``-th item
                  in ``sorted(data.responses["item_id"].unique())``. Call
                  ``data.to_fit_tensors()["item_ids"]`` to retrieve this ordering
                  before constructing ``item_features``.
        mask : torch.Tensor | None
            Boolean mask of shape ``(n_subjects, n_items)`` selecting which cells
            to use. Inferred from NaN/``-1`` when ``None``. Ignored for long-form
            input (absent rows are absent observations).
        max_epochs : int
            Maximum number of Adam optimisation epochs.
        lr : float
            Adam learning rate.
        weight_decay : float
            Adam L2 regularisation coefficient.
        convergence_tol : float
            Stop early when ``|loss_prev - loss_cur| < convergence_tol``.
        verbose : bool
            Show a tqdm progress bar during fitting.

        Returns
        -------
        dict
            Training history with key ``"losses"`` (per-epoch Bernoulli NLL).

        Raises
        ------
        ValueError
            If ``item_features`` is not a 2-D tensor or its second dimension does
            not match :attr:`item_feature_dim`.
        ValueError
            If no observed responses remain after applying ``mask`` / NaN filtering.
        ValueError
            If ``item_features`` has fewer rows than the maximum item index
            referenced in ``data``.
        TypeError
            If ``data`` is neither :class:`~torch_measure.datasets.LongFormData`
            nor :class:`torch.Tensor`.
        """
        if item_features.ndim != 2 or item_features.shape[1] != self.item_feature_dim:
            raise ValueError(
                f"item_features must be a 2-D tensor of shape "
                f"(n_items, {self.item_feature_dim}); "
                f"got shape {tuple(item_features.shape)}"
            )

        subject_idx, item_idx, response = _to_long_form(data, mask, self._device)

        if len(response) == 0:
            raise ValueError(
                "No observed responses to fit on. Check that data contains "
                "valid entries and that the mask selects at least one cell."
            )

        n_items_needed = int(item_idx.max().item()) + 1
        if item_features.shape[0] < n_items_needed:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows but "
                f"data references item index {n_items_needed - 1}. "
                f"Provide at least {n_items_needed} rows."
            )

        # Item features are fixed inputs, not model parameters. Index once before
        # the loop rather than re-slicing each epoch.
        obs_features = item_features.detach().to(self._device)[item_idx]  # (n_obs, item_feature_dim)
        training_query = {"subject_idx": subject_idx, "item_features": obs_features}

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        history = {"losses": []}

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Fitting DemandAssessor")
            except ImportError:
                pass

        prev_loss = float("inf")

        for _epoch in iterator:
            optimizer.zero_grad()
            probs = self.predict(training_query).clamp(1e-7, 1 - 1e-7)
            loss = bernoulli_nll(probs, response)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            history["losses"].append(loss_val)
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{loss_val:.6f}"})
            if abs(prev_loss - loss_val) < convergence_tol:
                break
            prev_loss = loss_val

        self.n_items_buf.fill_(item_features.shape[0])
        return history
