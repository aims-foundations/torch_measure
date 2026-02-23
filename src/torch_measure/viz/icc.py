# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Item Characteristic Curve (ICC) visualization."""

from __future__ import annotations

import torch


def plot_icc(
    difficulty: torch.Tensor,
    discrimination: torch.Tensor | None = None,
    guessing: torch.Tensor | None = None,
    item_indices: list[int] | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 200,
    ax=None,
    title: str = "Item Characteristic Curves",
    **kwargs,
):
    """Plot Item Characteristic Curves for selected items.

    Parameters
    ----------
    difficulty : torch.Tensor
        Item difficulties (M,).
    discrimination : torch.Tensor | None
        Item discriminations (M,). Defaults to 1.
    guessing : torch.Tensor | None
        Item guessing parameters (M,). Defaults to 0.
    item_indices : list[int] | None
        Which items to plot. Defaults to first 10.
    theta_range : tuple
        Range of ability values to plot.
    n_points : int
        Number of points on the x-axis.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if item_indices is None:
        item_indices = list(range(min(10, len(difficulty))))

    theta = torch.linspace(theta_range[0], theta_range[1], n_points)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for idx in item_indices:
        b = difficulty[idx].item()
        logit = theta - b

        if discrimination is not None:
            a = discrimination[idx].item()
            logit = a * logit
        else:
            a = 1.0

        prob = torch.sigmoid(logit)

        if guessing is not None:
            c = guessing[idx].item()
            prob = c + (1 - c) * prob
        else:
            c = 0.0

        label = f"Item {idx} (b={b:.2f}"
        if discrimination is not None:
            label += f", a={a:.2f}"
        if guessing is not None and c > 0.01:
            label += f", c={c:.2f}"
        label += ")"

        ax.plot(theta.numpy(), prob.numpy(), label=label, **kwargs)

    ax.set_xlabel(r"Ability ($\theta$)")
    ax.set_ylabel(r"$P(\mathrm{correct})$")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    return ax
