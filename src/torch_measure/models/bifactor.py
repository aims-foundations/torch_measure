# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Bifactor model with general and group-specific factors."""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class Bifactor(IRTModel):
    """Bifactor Model.

    A constrained factor model with one general factor and multiple
    group-specific factors. The general factor loads on all items,
    while group factors load only on items in their cluster.

    P(correct) = sigmoid(g_n * lambda_g_j + sum_k(s_nk * lambda_sk_j) + z_j)

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    n_groups : int
        Number of group-specific factors.
    item_groups : torch.Tensor
        Group assignment for each item, shape (n_items,). Values in [0, n_groups).
    device : str
        Device.
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        n_groups: int,
        item_groups: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.n_groups = n_groups
        self.register_buffer("item_groups", item_groups.to(self._device))

        # General factor
        self.general_ability = nn.Parameter(torch.randn(n_subjects, device=self._device))
        self.general_loading = nn.Parameter(torch.randn(n_items, device=self._device))

        # Group-specific factors
        self.group_ability = nn.Parameter(torch.randn(n_subjects, n_groups, device=self._device))
        self.group_loading = nn.Parameter(torch.randn(n_items, device=self._device))

        # Intercept
        self.Z = nn.Parameter(torch.randn(n_items, device=self._device))

    @property
    def ability(self) -> torch.Tensor:
        return self.general_ability.detach()

    @property
    def difficulty(self) -> torch.Tensor:
        return -self.Z.detach()

    def predict(self) -> torch.Tensor:
        # General component: (N,) @ (M,) -> (N, M)
        general = self.general_ability.unsqueeze(1) * self.general_loading.unsqueeze(0)

        # Group-specific component
        group = torch.zeros(self.n_subjects, self.n_items, device=self._device)
        for k in range(self.n_groups):
            items_in_group = self.item_groups == k
            if items_in_group.any():
                # (N,) * (sum_items,) -> (N, sum_items)
                group[:, items_in_group] = (
                    self.group_ability[:, k].unsqueeze(1) * self.group_loading[items_in_group].unsqueeze(0)
                )

        logit = general + group + self.Z.unsqueeze(0)
        return torch.sigmoid(logit)
