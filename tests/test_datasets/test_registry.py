# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the dataset registry (offline — no network required)."""

from unittest.mock import patch

import pytest

from torch_measure.datasets import DatasetInfo, info, list_datasets


def _manifest_entry(*, family: str, description: str = "manifest dataset") -> dict[str, object]:
    return {
        "response_type": "binary",
        "n_subjects": 1,
        "n_items": 2,
        "filename": "test.pt",
        "info": {
            "family": family,
            "tags": [family],
            "description": description,
        },
    }


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

    def test_openllm_family_filter(self):
        result = list_datasets(family="openllm")
        assert len(result) >= 3
        assert all(name.startswith("openllm/") for name in result)

    def test_arena_family_filter(self):
        result = list_datasets(family="arena")
        assert len(result) >= 1
        assert all(name.startswith("arena/") for name in result)

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

    def test_openllm_datasets_present(self):
        names = list_datasets()
        expected = [
            "openllm/bbh",
            "openllm/mmlu_pro",
            "openllm/all",
        ]
        for name in expected:
            assert name in names, f"{name} missing from registry"

    def test_arena_datasets_present(self):
        names = list_datasets()
        expected = ["arena/chatbot_arena"]
        for name in expected:
            assert name in names, f"{name} missing from registry"

    def test_metr_family_filter(self):
        result = list_datasets(family="metr")
        assert len(result) >= 8
        assert all(name.startswith("metr/") for name in result)

    def test_metr_datasets_present(self):
        names = list_datasets()
        expected = [
            "metr/all",
            "metr/all_score",
            "metr/hcast",
            "metr/rebench",
            "metr/swaa",
            "metr/hcast_score",
            "metr/rebench_score",
            "metr/swaa_score",
        ]
        for name in expected:
            assert name in names, f"{name} missing from registry"

    def test_agentic_family_filter(self):
        result = list_datasets(family="agentic")
        assert len(result) >= 28
        assert all(name.startswith("agentic/") for name in result)

    def test_agentic_datasets_present(self):
        names = list_datasets()
        expected = [
            "agentic/swebench",
            "agentic/assistantbench",
            "agentic/colbench",
            "agentic/corebench_hard",
            "agentic/gaia",
            "agentic/mind2web",
            "agentic/scicode",
            "agentic/scienceagentbench",
            "agentic/taubench_airline",
            "agentic/usaco",
            "agentic/colbench_raw_score",
            "agentic/corebench_hard_vision_score",
            "agentic/corebench_hard_written_score",
            "agentic/scienceagentbench_codebert_score",
            "agentic/scienceagentbench_success_rate",
            "agentic/scienceagentbench_valid_program",
            "agentic/colbench_rerun",
            "agentic/colbench_rerun_raw_score",
            "agentic/corebench_hard_rerun",
            "agentic/corebench_hard_rerun_vision_score",
            "agentic/corebench_hard_rerun_written_score",
            "agentic/scicode_rerun",
            "agentic/scicode_rerun_subtask_score",
            "agentic/scienceagentbench_rerun",
            "agentic/scienceagentbench_rerun_codebert_score",
            "agentic/scienceagentbench_rerun_success_rate",
            "agentic/scienceagentbench_rerun_valid_program",
            "agentic/all",
        ]
        for name in expected:
            assert name in names, f"{name} missing from registry"

    @pytest.mark.parametrize("manifest_key", ["foo", "newfam/foo"])
    def test_family_filter_includes_manifest_only_datasets(self, manifest_key):
        manifest = {
            "datasets": {
                manifest_key: _manifest_entry(family="newfam"),
            }
        }

        with patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest):
            assert "newfam/foo" in list_datasets()
            assert list_datasets(family="newfam") == ["newfam/foo"]

    def test_family_filter_uses_canonical_name_for_slash_keyed_manifest_entries(self):
        manifest = {
            "datasets": {
                "weird/foo": _manifest_entry(family="other"),
            }
        }

        with patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest):
            assert list_datasets(family="weird") == ["weird/foo"]
            assert list_datasets(family="other") == []


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
        assert result.response_type in ("binary", "continuous", "pairwise")
        assert result.n_subjects > 0
        assert result.n_items > 0
        assert result.repo_id
        assert result.filename

    def test_pairwise_response_type(self):
        result = info("arena/chatbot_arena")
        assert result.response_type == "pairwise"
        assert result.family == "arena"

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            info("nonexistent/dataset")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="helm/mmlu"):
            info("nonexistent/dataset")

    @pytest.mark.parametrize("manifest_key", ["foo", "newfam/foo"])
    def test_manifest_bare_lookup_is_key_shape_independent(self, manifest_key):
        manifest = {
            "datasets": {
                manifest_key: _manifest_entry(family="newfam"),
            }
        }

        with patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest):
            result = info("foo")

        assert result.name == "newfam/foo"
        assert result.family == "newfam"

    def test_manifest_slash_key_family_matches_canonical_name(self):
        manifest = {
            "datasets": {
                "weird/foo": _manifest_entry(family="other"),
            }
        }

        with patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest):
            result = info("weird/foo")

        assert result.name == "weird/foo"
        assert result.family == "weird"

    def test_manifest_collision_makes_registry_alias_ambiguous(self):
        manifest = {
            "datasets": {
                "mmlu": _manifest_entry(family="newfam"),
            }
        }

        with (
            patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest),
            pytest.raises(ValueError, match="Ambiguous dataset name") as exc_info,
        ):
            info("mmlu")

        message = str(exc_info.value)
        assert "helm/mmlu" in message
        assert "newfam/mmlu" in message

    def test_manifest_cannot_bypass_ambiguous_registry_bare_name(self):
        manifest = {
            "datasets": {
                "swebench": _manifest_entry(family="bench"),
            }
        }

        with (
            patch("torch_measure.datasets._manifest.load_manifest", return_value=manifest),
            pytest.raises(ValueError, match="Ambiguous dataset name") as exc_info,
        ):
            info("swebench")

        message = str(exc_info.value)
        assert "agentic/swebench" in message
        assert "bench/swebench" in message

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
        assert di.repo_id == "aims-foundations/measurement-db"
        assert di.tags == []
        assert di.citation == ""
        assert di.url == ""
        assert di.license == ""
        assert di.filename == ""
