# Copyright (c) 2026 AIMS Foundations. MIT License.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from torch_measure.viz.response_heatmap import plot_response_heatmap


class TestPlotResponseHeatmap:
    def test_masks_nan_cells(self):
        # Two NaNs must not disappear as numeric zeros once sorting is disabled.
        d = torch.tensor(
            [[1.0, float("nan"), 0.0], [float("nan"), 0.5, 1.0]],
            dtype=torch.float32,
        )
        _, ax = plt.subplots()
        plot_response_heatmap(d, sort_by_ability=False, sort_by_difficulty=False, ax=ax)
        plotted = ax.images[0].get_array()
        assert isinstance(plotted, np.ma.MaskedArray)
        m = plotted.mask
        assert m[0, 1]
        assert m[1, 0]
        assert not m[0, 2]
        assert not m[1, 1]

    def test_masks_non_finite(self):
        d = torch.tensor([[0.25, float("inf")]], dtype=torch.float32)
        _, ax = plt.subplots()
        plot_response_heatmap(d, sort_by_ability=False, sort_by_difficulty=False, ax=ax)
        plotted = ax.images[0].get_array()
        assert plotted.mask[0, 1]

    def test_zeros_not_masked_when_adjacent_nan(self):
        d = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        _, ax = plt.subplots()
        plot_response_heatmap(d, sort_by_ability=False, sort_by_difficulty=False, ax=ax)
        plotted = ax.images[0].get_array()
        assert plotted[0, 0] == pytest.approx(0.0)
        assert not plotted.mask[0, 0]
