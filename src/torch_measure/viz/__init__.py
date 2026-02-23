# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Visualization utilities for measurement analysis (requires matplotlib)."""

try:
    from torch_measure.viz.icc import plot_icc
    from torch_measure.viz.info import plot_information
    from torch_measure.viz.response_heatmap import plot_response_heatmap
    from torch_measure.viz.style import set_academic_style
except ImportError as err:
    raise ImportError(
        "Visualization requires matplotlib. Install with: pip install torch-measure[viz]"
    ) from err

__all__ = [
    "plot_response_heatmap",
    "plot_icc",
    "plot_information",
    "set_academic_style",
]
