# Copyright (c) 2026 AIMS Foundations. MIT License.

import numpy as np
import torch

from torch_measure.data import ResponseMatrix


class TestResponseMatrix:
    def test_init(self):
        data = torch.randn(10, 20)
        rm = ResponseMatrix(data)
        assert rm.n_subjects == 10
        assert rm.n_items == 20
        assert rm.n_rows == 10
        assert rm.n_cols == 20

    def test_shape(self):
        data = torch.randn(10, 20)
        rm = ResponseMatrix(data)
        assert rm.shape == (10, 20)

    def test_rejects_1d(self):
        try:
            ResponseMatrix(torch.randn(10))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_observed_mask(self):
        data = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        rm = ResponseMatrix(data)
        expected = torch.tensor([[True, False], [True, True]])
        assert torch.equal(rm.observed_mask, expected)

    def test_density(self):
        data = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        rm = ResponseMatrix(data)
        assert abs(rm.density - 0.75) < 1e-5

    def test_density_full(self):
        data = torch.randn(5, 5)
        rm = ResponseMatrix(data)
        assert abs(rm.density - 1.0) < 1e-5

    def test_subject_means(self):
        data = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        rm = ResponseMatrix(data)
        means = rm.subject_means
        assert abs(means[0].item() - 0.5) < 1e-5
        assert abs(means[1].item() - 0.5) < 1e-5

    def test_item_means(self):
        data = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        rm = ResponseMatrix(data)
        means = rm.item_means
        assert abs(means[0].item() - 1.0) < 1e-5
        assert abs(means[1].item() - 0.0) < 1e-5

    def test_binarize(self):
        data = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        rm = ResponseMatrix(data)
        binary = rm.binarize(threshold=0.5)
        assert binary.data[0, 0] == 0.0
        assert binary.data[0, 1] == 1.0
        assert binary.data[1, 0] == 1.0
        assert binary.data[1, 1] == 0.0

    def test_from_numpy(self):
        arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        rm = ResponseMatrix.from_numpy(arr)
        assert rm.shape == (2, 2)
        assert rm.data.dtype == torch.float32

    def test_repr(self):
        rm = ResponseMatrix(torch.randn(10, 20))
        r = repr(rm)
        assert "ResponseMatrix" in r
        assert "10" in r
        assert "20" in r

    def test_with_ids(self):
        data = torch.randn(3, 2)
        rm = ResponseMatrix(data, subject_ids=["a", "b", "c"], item_ids=["x", "y"])
        assert rm.subject_ids == ["a", "b", "c"]
        assert rm.item_ids == ["x", "y"]
