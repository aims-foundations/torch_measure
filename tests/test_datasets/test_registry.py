# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the manifest-backed dataset registry (offline — no network)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from torch_measure.datasets import DatasetInfo, info, list_datasets


@pytest.fixture(autouse=True)
def _fake_manifest():
    """Substitute a deterministic manifest so tests don't hit HuggingFace."""
    fake = {
        "mtbench": {
            "benchmark_id": "mtbench",
            "name": "MT-Bench",
            "license": "CC-BY-4.0",
            "source_url": "https://arxiv.org/abs/2306.05685",
            "description": "Open-ended multi-turn chat bench",
            "response_type": "likert_10",
            "response_scale": "{1, 2, ..., 10}",
            "categorical": True,
            "modality": ["text"],
            "domain": ["preference"],
            "paper_url": "https://arxiv.org/abs/2306.05685",
            "release_date": "2023-06",
        },
        "swebench": {
            "benchmark_id": "swebench",
            "name": "SWE-bench Verified",
            "license": "MIT",
            "source_url": "https://github.com/SWE-bench/experiments",
            "description": "Software-engineering agent bench",
            "response_type": "binary",
            "response_scale": "{0, 1}",
            "categorical": True,
            "modality": ["text"],
            "domain": ["software_engineering"],
            "paper_url": "https://arxiv.org/abs/2310.06770",
            "release_date": "2023-10",
        },
        "arena_hard": {
            "benchmark_id": "arena_hard",
            "name": "Arena-Hard-Auto",
            "license": "Apache-2.0",
            "source_url": "https://arxiv.org/abs/2406.11939",
            "description": "Preference-ranked prompts",
            "response_type": "win_rate",
            "response_scale": "{0, 0.125, ..., 1}",
            "categorical": True,
            "modality": ["text"],
            "domain": ["preference"],
            "paper_url": "https://arxiv.org/abs/2406.11939",
            "release_date": "2024-06",
        },
    }
    # Reset the module-level cache before and after, and patch load_manifest.
    from torch_measure.datasets import _manifest as _m

    prev = _m._manifest_cache
    _m._manifest_cache = None

    def _fake_load_manifest(*, force_download: bool = False):  # noqa: ARG001
        return fake

    with patch.object(_m, "load_manifest", _fake_load_manifest):
        yield

    _m._manifest_cache = prev


class TestListDatasets:
    def test_returns_list(self):
        result = list_datasets()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted(self):
        result = list_datasets()
        assert result == sorted(result)

    def test_all_strings(self):
        for name in list_datasets():
            assert isinstance(name, str)

    def test_expected_datasets_present(self):
        names = list_datasets()
        for expected in ("mtbench", "swebench", "arena_hard"):
            assert expected in names

    def test_family_filter_by_domain(self):
        result = list_datasets(family="preference")
        assert set(result) == {"mtbench", "arena_hard"}

    def test_family_filter_by_modality(self):
        result = list_datasets(family="text")
        assert set(result) == {"mtbench", "swebench", "arena_hard"}

    def test_unknown_family_returns_empty(self):
        assert list_datasets(family="nonexistent_family_xyz") == []


class TestInfo:
    def test_returns_dataset_info(self):
        assert isinstance(info("mtbench"), DatasetInfo)

    def test_name_matches(self):
        assert info("mtbench").name == "mtbench"

    def test_basic_fields(self):
        entry = info("mtbench")
        assert entry.license == "CC-BY-4.0"
        assert entry.response_type == "likert_10"
        assert entry.description
        assert "text" in entry.modality
        assert "preference" in entry.domain
        assert entry.paper_url.startswith("https://")
        assert entry.release_date == "2023-06"

    def test_family_defaults_to_first_domain(self):
        assert info("mtbench").family == "preference"
        assert info("swebench").family == "software_engineering"

    def test_url_alias_matches_source_url(self):
        entry = info("swebench")
        assert entry.url == entry.source_url
        assert entry.source_url.startswith("https://")

    def test_tags_union_modality_and_domain(self):
        entry = info("mtbench")
        assert "text" in entry.tags
        assert "preference" in entry.tags

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            info("nonexistent_dataset")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="mtbench"):
            info("nonexistent_dataset")

    def test_all_entries_consistent(self):
        for name in list_datasets():
            entry = info(name)
            assert entry.name == name


class TestDatasetInfo:
    def test_frozen(self):
        di = DatasetInfo(name="test")
        with pytest.raises(AttributeError):
            di.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        di = DatasetInfo(name="test")
        assert di.subject_entity == "LLM"
        assert di.item_entity == "question"
        assert di.repo_id == "aims-foundations/measurement-db"
        assert di.tags == []
        assert di.modality == []
        assert di.domain == []
        assert di.citation == ""
        assert di.url == ""
        assert di.license == ""
        assert di.filename == ""
        assert di.categorical is True
        assert di.n_subjects == 0
        assert di.n_items == 0
