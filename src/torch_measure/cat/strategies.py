# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Item selection strategies for Computerized Adaptive Testing.

Consolidated from irsl/utils.py _select_next_1pl, _select_next_2pl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from torch_measure.cat.fisher import fisher_information


class SelectionStrategy(ABC):
    """Abstract base for item selection strategies."""

    @abstractmethod
    def select(
        self,
        ability_estimate: torch.Tensor,
        difficulty: torch.Tensor,
        discrimination: torch.Tensor | None,
        administered: torch.Tensor,
    ) -> int:
        """Select the next item to administer.

        Parameters
        ----------
        ability_estimate : torch.Tensor
            Current ability estimate (scalar).
        difficulty : torch.Tensor
            Item difficulties (M,).
        discrimination : torch.Tensor | None
            Item discriminations (M,).
        administered : torch.Tensor
            Boolean mask of already-administered items (M,).

        Returns
        -------
        int
            Index of the selected item.
        """
        ...


class MaxInfoStrategy(SelectionStrategy):
    """Select the item with maximum Fisher information at current ability estimate."""

    def select(self, ability_estimate, difficulty, discrimination, administered):
        info = fisher_information(ability_estimate, difficulty, discrimination)
        if info.ndim > 1:
            info = info.squeeze(0)

        # Mask out administered items
        info[administered] = -float("inf")
        return info.argmax().item()


class SpanningStrategy(SelectionStrategy):
    """Select items spanning the difficulty range before switching to max-info.

    First selects items evenly across the difficulty range to get a rough
    ability estimate, then switches to maximum information selection.

    Parameters
    ----------
    n_spanning : int
        Number of items to select in the spanning phase.
    """

    def __init__(self, n_spanning: int = 10) -> None:
        self.n_spanning = n_spanning
        self._spanning_count = 0
        self._max_info = MaxInfoStrategy()

    def select(self, ability_estimate, difficulty, discrimination, administered):
        if self._spanning_count < self.n_spanning:
            self._spanning_count += 1
            return self._select_spanning(difficulty, administered)
        return self._max_info.select(ability_estimate, difficulty, discrimination, administered)

    def _select_spanning(self, difficulty, administered):
        available = ~administered
        if not available.any():
            return 0

        available_diffs = difficulty[available]
        available_indices = available.nonzero(as_tuple=True)[0]

        # Select items at evenly-spaced quantiles of difficulty
        n_available = len(available_diffs)
        target_quantile = self._spanning_count / max(self.n_spanning, 1)
        sorted_diffs, sorted_idx = available_diffs.sort()
        target_idx = min(int(target_quantile * n_available), n_available - 1)

        return available_indices[sorted_idx[target_idx]].item()

    def reset(self):
        """Reset spanning count for a new test session."""
        self._spanning_count = 0


class RandomStrategy(SelectionStrategy):
    """Select items randomly from the unadministered pool."""

    def select(self, ability_estimate, difficulty, discrimination, administered):
        available = (~administered).nonzero(as_tuple=True)[0]
        if len(available) == 0:
            return 0
        idx = torch.randint(len(available), (1,)).item()
        return available[idx].item()
