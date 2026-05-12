# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Many-Facet Rasch Model.

Consolidated from safety-irt/model/irt.py.

This model extends the Rasch model with additional facets (e.g., language,
rater) to separate construct-relevant from construct-irrelevant variance.

P(correct) = sigmoid(theta_n - (beta_j + gamma_l + tau_jl + delta_nl))

where:
- theta_n: subject ability
- beta_j: base item difficulty
- gamma_l: global facet shift (e.g., language effect)
- tau_jl: item-facet interaction (e.g., translation difficulty)
- delta_nl: subject-facet competence (e.g., model's language ability)
"""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class MultiFacetRasch(IRTModel):
    """Many-Facet Rasch Model.

    Extends the standard Rasch model with additional facets to model
    systematic sources of variation beyond ability and difficulty.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    n_facet_levels : int
        Number of levels in the additional facet (e.g., number of languages).
    device : str
        Device to place parameters on.
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        n_facet_levels: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.n_facet_levels = n_facet_levels

        # Core parameters
        self.ability = nn.Parameter(torch.randn(n_subjects, device=self._device))
        self.difficulty = nn.Parameter(torch.randn(n_items, device=self._device))

        # Facet parameters
        self.gamma = nn.Parameter(torch.zeros(n_facet_levels, device=self._device))  # facet shift
        self.tau = nn.Parameter(torch.zeros(n_items, n_facet_levels, device=self._device))  # item-facet interaction
        self.delta = nn.Parameter(torch.zeros(n_subjects, n_facet_levels, device=self._device))  # subject-facet

        # Anchor masks: set reference level (e.g., English) to zero
        self.register_buffer("gamma_mask", torch.ones(n_facet_levels, device=self._device))
        self.register_buffer("tau_mask", torch.ones(n_items, n_facet_levels, device=self._device))

    def set_reference_level(self, level_idx: int) -> None:
        """Set a facet level as the reference (anchor to zero).

        Parameters
        ----------
        level_idx : int
            Index of the reference level (e.g., 0 for English).
        """
        self.gamma_mask[level_idx] = 0.0
        self.tau_mask[:, level_idx] = 0.0

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute P(correct) at query rows for the given facet level(s).

        Query must contain ``subject_idx`` and ``item_idx`` (1-D, length N).
        Optionally ``facet_idx`` (1-D, length N or scalar). When omitted,
        defaults to facet level 0 — matches the prior behavior where
        fitting did not surface facet information.
        """
        s = query["subject_idx"]
        i = query["item_idx"]
        f = query.get("facet_idx")
        if f is None:
            f = torch.zeros_like(s)
        elif f.ndim == 0 or f.numel() == 1:
            f = torch.full_like(s, int(f.item()))

        gamma = self.gamma * self.gamma_mask
        tau = self.tau * self.tau_mask

        total_difficulty = self.difficulty[i] + gamma[f] + tau[i, f]
        subject_offset = self.delta[s, f]
        logit = (self.ability[s] - subject_offset) - total_difficulty
        return torch.sigmoid(logit)

    def fit(self, response_matrix, mask=None, method="mle", **kwargs):
        """Fit the model.

        Supports all fitting methods: 'mle', 'em', 'jml', 'svi'.
        """
        return super().fit(response_matrix, mask, method=method, **kwargs)
