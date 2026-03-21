# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Computerized Adaptive Testing runner.

Consolidated from irsl/utils.py _cat_core.
"""

from __future__ import annotations

import torch

from torch_measure.cat.strategies import MaxInfoStrategy, SelectionStrategy, SpanningStrategy
from torch_measure.fitting._losses import bernoulli_nll


class AdaptiveTester:
    """Run a Computerized Adaptive Test to efficiently estimate ability.

    Given a calibrated item bank (known difficulties/discriminations),
    adaptively selects items and updates the ability estimate after each
    response.

    Parameters
    ----------
    model : IRTModel
        A fitted IRT model with item parameters.
    strategy : str or SelectionStrategy
        Item selection strategy: "fisher" (default), "spanning", "random",
        or a custom SelectionStrategy instance.
    n_spanning : int
        Number of spanning items (only for "spanning" strategy).
    """

    def __init__(
        self,
        model,
        strategy: str | SelectionStrategy = "fisher",
        n_spanning: int = 10,
    ) -> None:
        self.model = model
        self.difficulty = model.difficulty.detach().clone()
        self.discrimination = getattr(model, "discrimination", None)
        if self.discrimination is not None:
            self.discrimination = self.discrimination.detach().clone()

        # Detect Beta IRT models and use appropriate loss function
        if hasattr(model, "phi"):
            from functools import partial

            from torch_measure.fitting._losses import beta_nll

            self._loss_fn = partial(beta_nll, phi=model.phi)
        else:
            self._loss_fn = bernoulli_nll

        if isinstance(strategy, str):
            if strategy == "fisher":
                self.strategy = MaxInfoStrategy()
            elif strategy == "spanning":
                self.strategy = SpanningStrategy(n_spanning)
            elif strategy == "random":
                from torch_measure.cat.strategies import RandomStrategy

                self.strategy = RandomStrategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")
        else:
            self.strategy = strategy

    def run(
        self,
        responses: torch.Tensor,
        budget: int | None = None,
        lr: float = 0.1,
        n_steps: int = 50,
    ) -> dict:
        """Run adaptive testing on a single subject.

        Parameters
        ----------
        responses : torch.Tensor
            Full response vector for the subject (n_items,).
            Only items selected by the algorithm will be "seen".
        budget : int | None
            Maximum number of items to administer. Defaults to all items.
        lr : float
            Learning rate for ability estimation.
        n_steps : int
            Number of optimization steps per ability update.

        Returns
        -------
        dict
            Results with:
            - 'ability': Final ability estimate (scalar)
            - 'administered': List of item indices in administration order
            - 'responses': List of responses to administered items
            - 'ability_trajectory': Ability estimate after each item
        """
        n_items = len(self.difficulty)
        if budget is None:
            budget = n_items
        budget = min(budget, n_items)

        administered = torch.zeros(n_items, dtype=torch.bool)
        administered[torch.isnan(responses)] = True  # skip missing responses
        n_available = int((~administered).sum().item())
        budget = min(budget, n_available)
        ability = torch.tensor(0.0, requires_grad=True)

        admin_order = []
        resp_list = []
        ability_trajectory = []

        if isinstance(self.strategy, SpanningStrategy):
            self.strategy.reset()

        for _ in range(budget):
            # Select next item
            with torch.no_grad():
                item_idx = self.strategy.select(
                    ability.detach(),
                    self.difficulty,
                    self.discrimination,
                    administered,
                )

            administered[item_idx] = True
            response = responses[item_idx].float()
            admin_order.append(item_idx)
            resp_list.append(response.item())

            # Update ability estimate via MLE
            ability = self._update_ability(
                ability.detach().clone().requires_grad_(True),
                torch.tensor(admin_order),
                torch.tensor(resp_list),
                lr=lr,
                n_steps=n_steps,
            )
            ability_trajectory.append(ability.item())

        return {
            "ability": ability.item(),
            "administered": admin_order,
            "responses": resp_list,
            "ability_trajectory": ability_trajectory,
        }

    def _update_ability(
        self,
        ability: torch.Tensor,
        item_indices: torch.Tensor,
        responses: torch.Tensor,
        lr: float,
        n_steps: int,
    ) -> torch.Tensor:
        """Update ability estimate given administered items and responses."""
        optimizer = torch.optim.Adam([ability], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            logit = ability - self.difficulty[item_indices]
            if self.discrimination is not None:
                logit = self.discrimination[item_indices] * logit
            prob = torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7)
            loss = self._loss_fn(prob, responses)
            # N(0,1) prior on ability
            loss = loss + 0.01 * ability**2
            loss.backward()
            optimizer.step()

        return ability
