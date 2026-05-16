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
    missing_color: str = "#bdbdbd",
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
    xlabel : str
        X axis label.
    ylabel : str
        Y axis label.
    missing_color : str
        Face color used for NaN / non-finite cells so they stay distinct from
        real zeros.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    data_t = data.detach().cpu().clone().float()
    valid = torch.isfinite(data_t)
    filled = torch.where(valid, data_t, torch.zeros_like(data_t))

    if sort_by_ability:
        counts = valid.float().sum(dim=1).clamp(min=1)
        row_means = filled.sum(dim=1) / counts
        row_order = row_means.argsort(descending=True)
        data_t = data_t[row_order]
        valid = valid[row_order]

    if sort_by_difficulty:
        counts = valid.float().sum(dim=0).clamp(min=1)
        col_means = filled.sum(dim=0) / counts
        col_order = col_means.argsort(descending=True)
        data_t = data_t[:, col_order]
        valid = valid[:, col_order]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    arr = np.ma.masked_where(~valid.numpy(), data_t.numpy())
    plot_cmap = mpl.colormaps[cmap].copy()
    plot_cmap.set_bad(color=missing_color)

    im = ax.imshow(arr, aspect="auto", cmap=plot_cmap, vmin=0, vmax=1, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="P(correct)")

    return ax
