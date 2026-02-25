# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for the dataset loader (mocked HF downloads)."""

from unittest.mock import patch

import pytest
import torch

from torch_measure.data import ResponseMatrix
from torch_measure.datasets import load


class TestLoadValidation:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load("nonexistent/xyz")


class TestLoadWithMock:
    """Test load() by mocking ``hf_hub_download`` to return a local .pt file."""

    def _make_pt_file(self, tmp_path, payload):
        pt_path = tmp_path / "test.pt"
        torch.save(payload, pt_path)
        return str(pt_path)

    def test_load_dict_payload(self, tmp_path):
        data = torch.bernoulli(torch.full((5, 10), 0.5))
        subject_ids = [f"model_{i}" for i in range(5)]
        item_ids = [f"q_{i}" for i in range(10)]
        payload = {"data": data, "subject_ids": subject_ids, "item_ids": item_ids}
        pt_path = self._make_pt_file(tmp_path, payload)

        with patch("huggingface_hub.hf_hub_download", return_value=pt_path):
            rm = load("helm/mmlu")

        assert isinstance(rm, ResponseMatrix)
        assert rm.shape == (5, 10)
        assert rm.subject_ids == subject_ids
        assert rm.item_ids == item_ids

    def test_load_bare_tensor_payload(self, tmp_path):
        data = torch.bernoulli(torch.full((8, 15), 0.5))
        pt_path = self._make_pt_file(tmp_path, data)

        with patch("huggingface_hub.hf_hub_download", return_value=pt_path):
            rm = load("helm/mmlu")

        assert isinstance(rm, ResponseMatrix)
        assert rm.shape == (8, 15)
        assert rm.subject_ids is None
        assert rm.item_ids is None

    def test_load_preserves_nan(self, tmp_path):
        data = torch.rand(4, 6)
        data[0, 0] = float("nan")
        data[2, 3] = float("nan")
        payload = {"data": data}
        pt_path = self._make_pt_file(tmp_path, payload)

        with patch("huggingface_hub.hf_hub_download", return_value=pt_path):
            rm = load("helm/mmlu")

        assert torch.isnan(rm.data[0, 0])
        assert torch.isnan(rm.data[2, 3])

    def test_force_download_passed_through(self, tmp_path):
        data = torch.rand(3, 5)
        pt_path = self._make_pt_file(tmp_path, {"data": data})

        with patch("huggingface_hub.hf_hub_download", return_value=pt_path) as mock_dl:
            load("helm/mmlu", force_download=True)

        mock_dl.assert_called_once()
        _, kwargs = mock_dl.call_args
        assert kwargs["force_download"] is True

    def test_repo_type_is_dataset(self, tmp_path):
        data = torch.rand(3, 5)
        pt_path = self._make_pt_file(tmp_path, {"data": data})

        with patch("huggingface_hub.hf_hub_download", return_value=pt_path) as mock_dl:
            load("helm/mmlu")

        _, kwargs = mock_dl.call_args
        assert kwargs["repo_type"] == "dataset"
