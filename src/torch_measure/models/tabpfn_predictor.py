# Copyright (c) 2026 AIMS Foundations. MIT License.

"""TabPFN-based predictor for cold-item performance prediction.

Per-cell tabular regression that consumes ``[item_features, subject_id]``
rows and predicts P(correct) directly via TabPFN's in-context learning.
Sibling to :class:`AmortizedIRT` but non-factorized: no latent ability,
no IRT factorization, no gradient training.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from tabpfn import TabPFNClassifier

from torch_measure.models._predictor import Predictor


class TabPFNPredictor(Predictor):
    """Per-cell TabPFN predictor for cold-item performance prediction.

    Each (subject, item) cell becomes a training row with features
    ``[item_features, subject_id]`` and label = response. ``subject_id``
    is appended as a categorical column at fit time. TabPFN does
    in-context learning over observed cells and predicts held cells.

    Unlike :class:`AmortizedIRT`, this model does not factorize into
    ability and difficulty -- subject identity is just one categorical
    feature alongside the item-side features. It inherits
    :class:`Predictor` directly (not :class:`IRTModel`) because it has
    no latent factor parameters. Use this when item features carry
    per-task signal beyond what subject identity already encodes; on
    homogeneous benchmarks (where they don't) a row-mean baseline can
    be hard to beat.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    n_features : int
        Number of item features (item_features.shape[1]).
    max_train : int, default 10000
        Maximum training rows passed to TabPFN. Larger contexts
        materialize an N x N attention tensor that exceeds GPU memory
        well below TabPFN's pretraining limits (e.g. 47K rows OOMs an
        H100 80 GB inside scaled_dot_product_attention). When the
        observed-cell count exceeds ``max_train``, a stratified random
        subsample is taken and a UserWarning is raised. TabPFN's design
        point is <=10K samples regardless.
    n_estimators : int, default 2
        Number of TabPFN ensemble members. Higher = better calibration
        at proportional cost.
    categorical_feature_indices : list[int] | None
        Indices into ``item_features`` of columns that should be
        treated as categorical. The internally-appended ``subject_id``
        column is always added to this list automatically.
    random_state : int, default 0
        Seed for the stratified subsample and TabPFN's internal RNG.
    device : str, default "cpu"
        Device for TabPFN inference. Use "cuda" or "cuda:0" for GPU.

    Notes
    -----
    The response matrix is treated as binary -- non-{0, 1} entries are
    cast to ``int`` after masking out NaN and -1, matching the
    convention of the other IRT models in this package.

    Examples
    --------
    >>> import torch
    >>> from torch_measure.models import TabPFNPredictor, predict_dense
    >>> response = (torch.rand(20, 30) > 0.5).float()
    >>> features = torch.randn(30, 8)
    >>> model = TabPFNPredictor(n_subjects=20, n_items=30, n_features=8)
    >>> _ = model.fit(response, features)
    >>> probs = predict_dense(model)
    >>> probs.shape
    torch.Size([20, 30])
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        n_features: int,
        max_train: int = 10000,
        n_estimators: int = 2,
        categorical_feature_indices: list[int] | None = None,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.n_features = n_features
        self.max_train = max_train
        self.n_estimators = n_estimators
        self.categorical_feature_indices = list(categorical_feature_indices or [])
        self.random_state = random_state

        self._classifier: Any | None = None
        self._item_features: torch.Tensor | None = None
        self._one_class_index: int | None = None

    def _build_xy(
        self,
        response_matrix: np.ndarray,
        item_features: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (response, mask, features) into per-cell (X, y)."""
        n_obs = int(mask.sum())
        n_feat = item_features.shape[1] + 1  # +1 for subject_id
        X = np.empty((n_obs, n_feat), dtype=float)
        y = np.empty(n_obs, dtype=int)
        k = 0
        for i in range(self.n_subjects):
            for j in range(self.n_items):
                if not mask[i, j]:
                    continue
                X[k, : item_features.shape[1]] = item_features[j]
                X[k, -1] = float(i)
                y[k] = int(response_matrix[i, j])
                k += 1
        return X, y

    def _maybe_subsample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(y) <= self.max_train:
            return X, y
        warnings.warn(
            f"TabPFNPredictor: training set has {len(y)} cells, exceeding "
            f"max_train={self.max_train}. Taking a stratified random "
            f"subsample. Increase max_train at your own risk -- large "
            f"contexts can OOM on GPU.",
            UserWarning,
            stacklevel=3,
        )
        from sklearn.model_selection import train_test_split

        try:
            X_keep, _, y_keep, _ = train_test_split(
                X,
                y,
                train_size=self.max_train,
                stratify=y,
                random_state=self.random_state,
            )
        except ValueError:
            # Stratification failed (e.g., one class too rare). Fall back
            # to a non-stratified random subsample.
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(y), size=self.max_train, replace=False)
            X_keep = X[idx]
            y_keep = y[idx]
        return X_keep, y_keep

    def fit(
        self,
        response_matrix: torch.Tensor,
        item_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Fit TabPFN on observed cells.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Binary response matrix of shape (n_subjects, n_items). Use
            NaN or -1 for missing entries.
        item_features : torch.Tensor
            Item features of shape (n_items, n_features).
        mask : torch.Tensor | None
            Boolean mask of cells to use for fitting. If None, uses all
            non-NaN, non-(-1) entries.

        Returns
        -------
        dict
            Training summary with keys ``n_train`` (rows used after the
            cap) and ``n_observed`` (rows before the cap).

        Raises
        ------
        ValueError
            If ``item_features`` has the wrong shape.
        RuntimeError
            If the training set has only one class.
        """
        if item_features.shape != (self.n_items, self.n_features):
            raise ValueError(
                f"item_features must have shape ({self.n_items}, {self.n_features}); got {tuple(item_features.shape)}"
            )

        Y = response_matrix.detach().cpu().numpy()
        X_feat = item_features.detach().cpu().numpy()

        mask_np = (~np.isnan(Y)) & (Y != -1) if mask is None else mask.detach().cpu().numpy().astype(bool)

        X, y = self._build_xy(Y, X_feat, mask_np)
        if len(np.unique(y)) < 2:
            raise RuntimeError(
                "TabPFNPredictor: training set has only one class. Need both 0s and 1s in observed cells."
            )

        n_observed = len(y)
        X, y = self._maybe_subsample(X, y)

        cat_indices = list(self.categorical_feature_indices) + [self.n_features]

        self._classifier = TabPFNClassifier(
            n_estimators=self.n_estimators,
            categorical_features_indices=cat_indices,
            device=str(self._device),
            ignore_pretraining_limits=True,
            random_state=self.random_state,
        )
        self._classifier.fit(X, y)

        self._item_features = item_features.detach().cpu()
        classes = list(self._classifier.classes_)
        self._one_class_index = classes.index(1)

        return {"n_train": int(len(y)), "n_observed": int(n_observed)}

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict P(correct) at the (subject, item) cells in ``query``.

        Parameters
        ----------
        query : dict[str, torch.Tensor]
            Must contain 1-D ``subject_idx`` and ``item_idx`` tensors of
            equal length ``N``.

        Returns
        -------
        torch.Tensor
            Probabilities, shape ``(N,)``, on the model's device.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._classifier is None or self._item_features is None:
            raise RuntimeError("Call fit() before predict()")

        s_idx = query["subject_idx"].cpu().numpy()
        i_idx = query["item_idx"].cpu().numpy()
        feats = self._item_features.numpy()
        # Per-row design matrix: item features for the queried items, plus
        # the subject id appended as the trailing categorical column.
        X = np.empty((len(s_idx), feats.shape[1] + 1), dtype=float)
        X[:, : feats.shape[1]] = feats[i_idx]
        X[:, -1] = s_idx.astype(float)

        proba = self._classifier.predict_proba(X)
        p1 = proba[:, self._one_class_index]
        return torch.from_numpy(p1).float().to(self._device)
