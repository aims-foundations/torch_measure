# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the long-form dataset loader (local-dir mode — no network)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from torch_measure.data import ResponseMatrix  # noqa: E402
from torch_measure.datasets import LongFormData, load  # noqa: E402


def _write_registry(local_dir, bench="mtbench"):
    """Write a tiny self-consistent long-form dataset into ``local_dir``."""
    responses = pd.DataFrame({
        "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "item_id": ["i1", "i2", "i1", "i2", "i1", "i2"],
        "benchmark_id": [bench] * 6,
        "trial": [1, 1, 1, 1, 1, 1],
        "test_condition": [None] * 6,
        "response": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "correct_answer": [None] * 6,
        "trace": [None] * 6,
    })
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "other_item"],
        "benchmark_id": [bench, bench, "other_bench"],
        "raw_item_id": ["q1", "q2", "qX"],
        "content": ["first question", "second question", "irrelevant"],
        "correct_answer": [None, None, None],
        "content_hash": ["h1", "h2", "hX"],
    })
    subjects = pd.DataFrame({
        "subject_id": ["s1", "s2", "s3", "s_unused"],
        "display_name": ["Model A", "Model B", "Model C", "Unused"],
        "provider": ["p1", "p2", "p3", "pX"],
        "hub_repo": [None, None, None, None],
        "revision": [None, None, None, None],
        "params": [None, None, None, None],
        "release_date": [None, None, None, None],
        "raw_labels_seen": [["a"], ["b"], ["c"], ["x"]],
        "notes": [None, None, None, None],
    })
    benchmarks = pd.DataFrame({
        "benchmark_id": [bench],
        "name": ["MT-Bench"],
        "license": ["CC-BY-4.0"],
        "source_url": ["https://example.com/mtbench"],
        "description": ["A small test bench"],
        "response_type": ["binary"],
        "response_scale": ["{0, 1}"],
        "categorical": [True],
        "modality": [["text"]],
        "domain": [["preference"]],
        "paper_url": ["https://arxiv.org/abs/2306.05685"],
        "release_date": ["2023-06"],
    })

    responses.to_parquet(local_dir / f"{bench}.parquet")
    items.to_parquet(local_dir / "items.parquet")
    subjects.to_parquet(local_dir / "subjects.parquet")
    benchmarks.to_parquet(local_dir / "benchmarks.parquet")


class TestLoadLocalDir:
    def test_returns_long_form_data(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert isinstance(data, LongFormData)

    def test_name_matches(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert data.name == "mtbench"

    def test_responses_preserved(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert len(data.responses) == 6
        assert set(data.responses.columns) >= {
            "subject_id", "item_id", "benchmark_id", "trial",
            "test_condition", "response",
        }

    def test_items_filtered_to_benchmark(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert set(data.items["item_id"]) == {"i1", "i2"}
        assert (data.items["benchmark_id"] == "mtbench").all()

    def test_subjects_filtered_to_observed(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert set(data.subjects["subject_id"]) == {"s1", "s2", "s3"}
        assert "s_unused" not in set(data.subjects["subject_id"])

    def test_info_populated_from_benchmarks_parquet(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert data.info["benchmark_id"] == "mtbench"
        assert data.info["license"] == "CC-BY-4.0"
        assert data.info["response_type"] == "binary"

    def test_traces_absent_when_no_traces_parquet(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        assert data.traces is None

    def test_traces_present_when_traces_parquet(self, tmp_path):
        _write_registry(tmp_path)
        traces = pd.DataFrame({
            "subject_id": ["s1", "s1"],
            "item_id": ["i1", "i2"],
            "trial": [1, 1],
            "test_condition": [None, None],
            "trace": ["hello", "world"],
        })
        traces.to_parquet(tmp_path / "mtbench_traces.parquet")

        data = load("mtbench", local_dir=tmp_path)
        assert data.traces is not None
        assert len(data.traces) == 2

    def test_missing_dataset_parquet_raises(self, tmp_path):
        _write_registry(tmp_path)
        with pytest.raises(FileNotFoundError):
            load("nonexistent", local_dir=tmp_path)


class TestToResponseMatrix:
    def test_pivot_shape(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        rm = data.to_response_matrix()
        assert isinstance(rm, ResponseMatrix)
        assert rm.shape == (3, 2)

    def test_subject_ids_use_display_name(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        rm = data.to_response_matrix()
        assert set(rm.subject_ids) == {"Model A", "Model B", "Model C"}

    def test_item_contents_populated(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        rm = data.to_response_matrix()
        contents = set(rm.item_contents)
        assert "first question" in contents
        assert "second question" in contents

    def test_info_carries_benchmark_id(self, tmp_path):
        _write_registry(tmp_path)
        data = load("mtbench", local_dir=tmp_path)
        rm = data.to_response_matrix()
        assert rm.info is not None
        assert rm.info.get("benchmark_id") == "mtbench"

    def test_aggregates_over_trials(self, tmp_path):
        """Multiple trials per cell are averaged."""
        _write_registry(tmp_path)
        # Overwrite responses to introduce two trials for (s1, i1)
        responses = pd.DataFrame({
            "subject_id": ["s1", "s1", "s1", "s2", "s3"],
            "item_id": ["i1", "i1", "i2", "i1", "i1"],
            "benchmark_id": ["mtbench"] * 5,
            "trial": [1, 2, 1, 1, 1],
            "test_condition": [None] * 5,
            "response": [1.0, 0.0, 1.0, 1.0, 0.0],
            "correct_answer": [None] * 5,
            "trace": [None] * 5,
        })
        responses.to_parquet(tmp_path / "mtbench.parquet")

        data = load("mtbench", local_dir=tmp_path)
        rm = data.to_response_matrix()
        # (s1, i1) should average 1.0 and 0.0 → 0.5
        assert rm.shape == (3, 2)
        idx = rm.subject_ids.index("Model A")
        item_idx = rm.item_ids.index("i1")
        assert abs(float(rm.data[idx, item_idx]) - 0.5) < 1e-6
