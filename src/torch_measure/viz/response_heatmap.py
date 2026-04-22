# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Response matrix heatmap visualization."""

from __future__ import annotations

import torch


def plot_response_heatmap(
    data: torch.Tensor,
    sort_by_ability: bool = True,
    sort_by_difficulty: bool = True,
    ax=None,
    cmap: str = "RdYlGn",
    title: str = "Response Matrix",
    xlabel: str = "Items",
    ylabel: str = "Subjects",
    **kwargs,
):
    """Plot a response matrix as a heatmap.

    Parameters
    ----------
    data : torch.Tensor
        Response matrix (n_subjects, n_items).
    sort_by_ability : bool
        Sort rows by total score (highest ability at top).
    sort_by_difficulty : bool
        Sort columns by item facility (easiest items left).
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates new figure if None.
    cmap : str
        Colormap name.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    data_np = data.detach().cpu().clone()
    mask = ~torch.isnan(data_np)
    data_np[~mask] = 0.0

    if sort_by_ability:
        row_means = (data_np * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        row_order = row_means.argsort(descending=True)
        data_np = data_np[row_order]

    if sort_by_difficulty:
        col_means = (data_np * mask.float()).sum(dim=0) / mask.float().sum(dim=0).clamp(min=1)
        col_order = col_means.argsort(descending=True)
        data_np = data_np[:, col_order]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data_np.numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="P(correct)")

    return ax
