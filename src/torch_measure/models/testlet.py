# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Testlet IRT models (Bradlow, Wainer & Wang, 1999).

Accounts for local item dependence (LID) within testlets by adding
a random effect per (subject, testlet) pair:

    logit P(X_{ij} = 1) = theta_j - b_i + gamma_{j, t(i)}

where gamma_{j, t} ~ N(0, sigma^2_t) are testlet-specific random effects.
"""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


def build_testlet_map(
    item_ids: list[str],
    separator: str = ":",
) -> tuple[torch.Tensor, list[str]]:
    """Build a testlet mapping from hierarchical item identifiers.

    Parameters
    ----------
    item_ids : list[str]
        Item identifiers with testlet structure, e.g.
        ``["task_1:0", "task_1:1", "task_2:0", ...]``.
        The prefix before *separator* identifies the testlet.
    separator : str
        Delimiter between testlet name and sub-item index.

    Returns
    -------
    testlet_map : torch.Tensor
        Integer tensor of shape ``(n_items,)`` mapping each item to
        its testlet index.
    testlet_names : list[str]
        Ordered list of unique testlet names (first-seen order).
    """
    prefixes = [item_id.rsplit(separator, 1)[0] for item_id in item_ids]
    unique_testlets = list(dict.fromkeys(prefixes))
    name_to_idx = {name: idx for idx, name in enumerate(unique_testlets)}
    testlet_map = torch.tensor([name_to_idx[p] for p in prefixes], dtype=torch.long)
    return testlet_map, unique_testlets


class TestletRasch(IRTModel):
    __test__ = False  # prevent pytest collection

    """Testlet Rasch (1PL) IRT model.

    Extends the Rasch model with testlet random effects to account for
    local item dependence within testlets (groups of items sharing context).

    .. math::

        \\text{logit}\\, P(X_{ij}=1) = \\theta_j - b_i + \\gamma_{j,\\,t(i)}

    where :math:`\\gamma_{j,t} \\sim N(0, \\sigma_t^2)` are testlet-specific
    random effects.

    For MLE/JML fitting, testlet effects are treated as fixed effects
    (optimized directly). For SVI (Bayesian) fitting, a hierarchical prior
    ``testlet_scale ~ LogNormal`` provides shrinkage — recommended for
    best results.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (rows).
    n_items : int
        Number of items (columns).
    testlet_map : torch.Tensor
        Integer tensor of shape ``(n_items,)`` mapping each item to its
        testlet index in ``[0, n_testlets)``.
    device : str
        Device for parameters.

    Attributes
    ----------
    ability : nn.Parameter
        Subject ability, shape ``(n_subjects,)``.
    difficulty : nn.Parameter
        Item difficulty, shape ``(n_items,)``.
    testlet_effect : nn.Parameter
        Random effects, shape ``(n_subjects, n_testlets)``.
    n_testlets : int
        Number of unique testlets.
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        testlet_map: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)

        testlet_map = testlet_map.to(dtype=torch.long, device=self._device)
        if testlet_map.shape != (n_items,):
            raise ValueError(f"testlet_map must have shape ({n_items},), got {testlet_map.shape}")
        self.register_buffer("testlet_map", testlet_map)
        self.n_testlets = int(testlet_map.max().item()) + 1

        self.ability = nn.Parameter(torch.randn(n_subjects, device=self._device))
        self.difficulty = nn.Parameter(torch.randn(n_items, device=self._device))
        self.testlet_effect = nn.Parameter(
            torch.zeros(n_subjects, self.n_testlets, device=self._device)
        )

    @property
    def testlet_scale(self) -> torch.Tensor:
        """Empirical standard deviation of testlet effects per testlet.

        Returns
        -------
        torch.Tensor
            Shape ``(n_testlets,)``.
        """
        return self.testlet_effect.std(dim=0)

    def predict(self) -> torch.Tensor:
        """Compute response probabilities including testlet effects.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape ``(n_subjects, n_items)``.
        """
        logit = self.ability.unsqueeze(1) - self.difficulty.unsqueeze(0)
        logit = logit + self.testlet_effect[:, self.testlet_map]
        return torch.sigmoid(logit)
