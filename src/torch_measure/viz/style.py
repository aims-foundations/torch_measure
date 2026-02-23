# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Academic plotting style presets.

Consolidated from the tueplots + seaborn setup used across all AIMS repos.
"""

from __future__ import annotations


def set_academic_style(venue: str = "icml", usetex: bool = False) -> dict:
    """Set matplotlib to academic publication style.

    Parameters
    ----------
    venue : str
        Conference/journal preset: "icml", "neurips", "aistats", "jmlr".
    usetex : bool
        Whether to use LaTeX for text rendering.

    Returns
    -------
    dict
        The rcParams that were applied.
    """
    import matplotlib.pyplot as plt

    try:
        from tueplots import bundles

        venue_map = {
            "icml": bundles.icml2022,
            "neurips": bundles.neurips2024,
            "aistats": bundles.aistats2023,
            "jmlr": bundles.jmlr2001,
        }

        params = venue_map[venue](usetex=usetex) if venue in venue_map else bundles.icml2022(usetex=usetex)

        plt.rcParams.update(params)
        return params
    except ImportError:
        # Fallback: basic academic style without tueplots
        params = {
            "figure.figsize": (3.25, 2.0),
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
        plt.rcParams.update(params)
        return params
