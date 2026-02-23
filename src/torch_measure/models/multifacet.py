# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Many-Facet Rasch Model (requires pyro-ppl).

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

    Notes
    -----
    For Bayesian estimation via Pyro SVI, install pyro-ppl:
        pip install torch-measure[bayesian]
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

    def predict(self, facet_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Compute response probabilities for a specific facet level.

        Parameters
        ----------
        facet_indices : torch.Tensor | None
            Facet level index for each observation. If None, uses level 0.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        if facet_indices is None:
            facet_indices = torch.zeros(1, dtype=torch.long, device=self._device)

        gamma = self.gamma * self.gamma_mask
        tau = self.tau * self.tau_mask

        # For a single facet level
        if facet_indices.numel() == 1:
            fl = facet_indices.item()
            total_difficulty = self.difficulty + gamma[fl] + tau[:, fl]
            subject_offset = self.delta[:, fl]
            logit = (self.ability - subject_offset).unsqueeze(1) - total_difficulty.unsqueeze(0)
            return torch.sigmoid(logit)

        raise NotImplementedError("Batch facet indices not yet supported. Pass a single facet level.")

    def fit(self, response_matrix, mask=None, method="mle", **kwargs):
        """Fit the model.

        Supports all fitting methods: 'mle', 'em', 'jml', 'svi'.
        For Bayesian SVI, install pyro-ppl: ``pip install torch-measure[bayesian]``
        """
        return super().fit(response_matrix, mask, method=method, **kwargs)
