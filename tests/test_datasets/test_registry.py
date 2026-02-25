# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for the dataset registry (offline — no network required)."""

import pytest

from torch_measure.datasets import DatasetInfo, info, list_datasets


class TestListDatasets:
    def test_returns_list(self):
        result = list_datasets()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted(self):
        result = list_datasets()
        assert result == sorted(result)

    def test_all_strings(self):
        for name in list_datasets():
            assert isinstance(name, str)

    def test_family_filter(self):
        result = list_datasets(family="helm")
        assert len(result) >= 23
        assert all(name.startswith("helm/") for name in result)

    def test_unknown_family_returns_empty(self):
        result = list_datasets(family="nonexistent_family_xyz")
        assert result == []

    def test_helm_datasets_present(self):
        names = list_datasets()
        expected = [
            "helm/mmlu",
            "helm/gsm8k",
            "helm/math",
            "helm/truthfulqa",
            "helm/commonsense",
            "helm/boolq",
            "helm/imdb",
            "helm/civil_comments",
            "helm/lsat_qa",
            "helm/legalbench",
            "helm/legal_support",
            "helm/med_qa",
            "helm/bbq",
            "helm/air_bench_2024",
            "helm/babi_qa",
            "helm/wikifact",
            "helm/entity_matching",
            "helm/entity_data_imputation",
            "helm/raft",
            "helm/synthetic_reasoning",
            "helm/dyck_language",
            "helm/thai_exam",
            "helm/all",
        ]
        for name in expected:
            assert name in names, f"{name} missing from registry"


class TestInfo:
    def test_returns_dataset_info(self):
        result = info("helm/mmlu")
        assert isinstance(result, DatasetInfo)

    def test_name_matches(self):
        result = info("helm/mmlu")
        assert result.name == "helm/mmlu"

    def test_family_matches(self):
        result = info("helm/mmlu")
        assert result.family == "helm"

    def test_has_required_fields(self):
        result = info("helm/mmlu")
        assert result.description
        assert result.response_type in ("binary", "continuous")
        assert result.n_subjects > 0
        assert result.n_items > 0
        assert result.repo_id
        assert result.filename

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            info("nonexistent/dataset")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="helm/mmlu"):
            info("nonexistent/dataset")

    def test_all_entries_consistent(self):
        """Every dataset in the registry should have a name matching its key."""
        for name in list_datasets():
            entry = info(name)
            assert entry.name == name
            assert "/" in name
            assert entry.family == name.split("/")[0]


class TestDatasetInfo:
    def test_frozen(self):
        di = DatasetInfo(
            name="test/example",
            family="test",
            description="test dataset",
            response_type="binary",
            n_subjects=10,
            n_items=20,
        )
        with pytest.raises(AttributeError):
            di.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        di = DatasetInfo(
            name="test/example",
            family="test",
            description="test",
            response_type="binary",
            n_subjects=10,
            n_items=20,
        )
        assert di.subject_entity == "LLM"
        assert di.item_entity == "question"
        assert di.repo_id == "sangttruong/torch-measure-data"
        assert di.tags == []
        assert di.citation == ""
        assert di.url == ""
        assert di.license == ""
        assert di.filename == ""
