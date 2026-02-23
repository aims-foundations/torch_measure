# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Information function visualization."""

from __future__ import annotations

import torch

from torch_measure.cat.fisher import fisher_information


def plot_information(
    difficulty: torch.Tensor,
    discrimination: torch.Tensor | None = None,
    item_indices: list[int] | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 200,
    plot_total: bool = True,
    ax=None,
    title: str = "Item/Test Information Function",
    **kwargs,
):
    """Plot item and/or test information functions.

    Parameters
    ----------
    difficulty : torch.Tensor
        Item difficulties (M,).
    discrimination : torch.Tensor | None
        Item discriminations (M,).
    item_indices : list[int] | None
        Items to plot individually. None = skip individual items.
    theta_range : tuple
        Range of ability values.
    n_points : int
        Number of theta points.
    plot_total : bool
        Whether to plot the total test information function.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    theta = torch.linspace(theta_range[0], theta_range[1], n_points)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    info = fisher_information(theta, difficulty, discrimination)  # (n_points, n_items)

    if item_indices is not None:
        for idx in item_indices:
            ax.plot(theta.numpy(), info[:, idx].numpy(), alpha=0.5, label=f"Item {idx}", **kwargs)

    if plot_total:
        total_info = info.sum(dim=1)
        ax.plot(theta.numpy(), total_info.numpy(), color="black", linewidth=2, label="Total", **kwargs)

    ax.set_xlabel(r"Ability ($\theta$)")
    ax.set_ylabel("Information")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    return ax
