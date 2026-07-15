# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Integration tests for DemandAnnotator with a stub Gemini client.

No real API calls are made. Tests verify:
- Correct number and order of API calls
- Prompt construction (rubric content, dimension names)
- Feature vector alignment with DIMENSION_ORDER
- Cache hit/miss behavior
- UGAnnotator pipeline
"""

import math
from pathlib import Path

import pytest

from torch_measure.annotation._annotator import DemandAnnotator
from torch_measure.annotation._cache import AnnotationCache
from torch_measure.annotation._rubrics import RubricsCatalog
from torch_measure.annotation._types import (
    DEMAND_DIMENSIONS,
    DIMENSION_ORDER,
    AnnotationJob,
    ItemAnnotation,
)
from torch_measure.annotation._ug import UGAnnotator


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------

_DEMAND_RESPONSE = (
    "Step 1: The task involves basic recall.\n\n"
    "Step 2: No complex operations needed.\n\n"
    "Thus, the level of *TestDim* demanded by the given TASK INSTANCE is: 2"
)
_UG_RESPONSE = "4"  # 4-choice MCQ → (1 - 1/4) * 100 = 75.0


class StubClient:
    """Records every generate() call; returns configurable stub responses."""

    def __init__(
        self,
        demand_response: str = _DEMAND_RESPONSE,
        ug_response: str = _UG_RESPONSE,
    ) -> None:
        self.model = "stub-model-001"
        self.calls: list[str] = []
        self._demand_response = demand_response
        self._ug_response = ug_response

    def generate(self, prompt: str) -> tuple[str, str]:
        self.calls.append(prompt)
        if "Reference answer:" in prompt:
            return self._ug_response, "stop"
        return self._demand_response, "stop"

    @property
    def demand_calls(self) -> list[str]:
        return [c for c in self.calls if "Reference answer:" not in c]

    @property
    def ug_calls(self) -> list[str]:
        return [c for c in self.calls if "Reference answer:" in c]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def catalog():
    return RubricsCatalog()


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


@pytest.fixture
def job() -> AnnotationJob:
    return AnnotationJob(
        item_id="item_test_001",
        content="What is the capital of France?",
        reference_answer="Paris",
    )


@pytest.fixture
def annotator(stub, catalog) -> DemandAnnotator:
    return DemandAnnotator(client=stub, rubrics=catalog, cache=None)


# ---------------------------------------------------------------------------
# API call count and ordering
# ---------------------------------------------------------------------------

class TestAPICallCount:

    def test_exactly_19_calls_per_item(self, stub, annotator, job):
        annotator.annotate(job)
        assert len(stub.calls) == 19

    def test_exactly_18_demand_calls(self, stub, annotator, job):
        annotator.annotate(job)
        assert len(stub.demand_calls) == 18

    def test_exactly_1_ug_call(self, stub, annotator, job):
        annotator.annotate(job)
        assert len(stub.ug_calls) == 1

    def test_demand_calls_before_ug_call(self, stub, annotator, job):
        """All 18 demand calls happen before the UG call."""
        annotator.annotate(job)
        first_ug_idx = next(
            i for i, c in enumerate(stub.calls) if "Reference answer:" in c
        )
        assert first_ug_idx == 18  # demand calls are indices 0-17


class TestPromptContent:

    def test_demand_prompts_contain_dimension_names(self, stub, catalog, annotator, job):
        annotator.annotate(job)
        rubrics = catalog.all_demand_rubrics()
        for i, rubric in enumerate(rubrics):
            assert rubric.dimension_name in stub.demand_calls[i], (
                f"Prompt {i} for rubric {rubric.acronym} missing "
                f"dimension name '{rubric.dimension_name}'"
            )

    def test_demand_prompts_contain_item_text(self, stub, annotator, job):
        annotator.annotate(job)
        for prompt in stub.demand_calls:
            assert job.content in prompt

    def test_demand_prompts_contain_task_instance_label(self, stub, annotator, job):
        annotator.annotate(job)
        for prompt in stub.demand_calls:
            assert "TASK INSTANCE:" in prompt

    def test_demand_prompts_contain_chain_of_thoughts_header(self, stub, annotator, job):
        annotator.annotate(job)
        for prompt in stub.demand_calls:
            assert "CHAIN-OF-THOUGHTS REASONING STEPS" in prompt

    def test_ug_prompt_contains_reference_answer_label(self, stub, annotator, job):
        annotator.annotate(job)
        ug_prompt = stub.ug_calls[0]
        assert "Reference answer:" in ug_prompt

    def test_ug_prompt_contains_item_content(self, stub, annotator, job):
        annotator.annotate(job)
        ug_prompt = stub.ug_calls[0]
        assert job.content in ug_prompt

    def test_ug_prompt_contains_reference_answer_value(self, stub, annotator, job):
        annotator.annotate(job)
        ug_prompt = stub.ug_calls[0]
        assert job.reference_answer in ug_prompt

    def test_demand_prompts_ordered_by_dimension_order(self, stub, catalog, annotator, job):
        """The i-th demand call must be for DEMAND_DIMENSIONS[i]."""
        annotator.annotate(job)
        rubrics = catalog.all_demand_rubrics()
        for i, rubric in enumerate(rubrics):
            assert rubric.acronym == DEMAND_DIMENSIONS[i]


# ---------------------------------------------------------------------------
# ItemAnnotation structure
# ---------------------------------------------------------------------------

class TestItemAnnotationStructure:

    def test_returns_item_annotation_type(self, annotator, job):
        result = annotator.annotate(job)
        assert isinstance(result, ItemAnnotation)

    def test_item_id_preserved(self, annotator, job):
        result = annotator.annotate(job)
        assert result.item_id == job.item_id

    def test_demands_dict_has_all_18_keys(self, annotator, job):
        result = annotator.annotate(job)
        assert set(result.demands.keys()) == set(DEMAND_DIMENSIONS)

    def test_demand_levels_match_stub_response(self, annotator, job):
        """Stub returns '...is: 2' → extract_demand_level returns 2.0."""
        result = annotator.annotate(job)
        for dim in DEMAND_DIMENSIONS:
            assert result.demands[dim].level == 2.0

    def test_demand_item_ids_match_job(self, annotator, job):
        result = annotator.annotate(job)
        for dim in DEMAND_DIMENSIONS:
            assert result.demands[dim].item_id == job.item_id

    def test_ug_annotation_present(self, annotator, job):
        result = annotator.annotate(job)
        assert result.ug is not None

    def test_ug_score_matches_stub_response(self, annotator, job):
        """Stub returns '4' → (1 - 1/4) * 100 = 75.0."""
        result = annotator.annotate(job)
        assert abs(result.ug.ug_score - 75.0) < 1e-6

    def test_ug_item_id_matches_job(self, annotator, job):
        result = annotator.annotate(job)
        assert result.ug.item_id == job.item_id

    def test_model_response_stored(self, annotator, job):
        result = annotator.annotate(job)
        for dim in DEMAND_DIMENSIONS:
            assert len(result.demands[dim].model_response) > 0
        assert len(result.ug.model_response) > 0

    def test_finish_reason_stored(self, annotator, job):
        result = annotator.annotate(job)
        for dim in DEMAND_DIMENSIONS:
            assert result.demands[dim].finish_reason == "stop"
        assert result.ug.finish_reason == "stop"

    def test_nan_level_when_parse_fails(self, catalog, job):
        """Unparseable model response → math.nan stored in demand level."""
        bad_stub = StubClient(demand_response="I cannot determine the level.")
        annotator = DemandAnnotator(client=bad_stub, rubrics=catalog, cache=None)
        result = annotator.annotate(job)
        for dim in DEMAND_DIMENSIONS:
            assert math.isnan(result.demands[dim].level)


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class TestFeatureVector:

    def test_length_is_19(self, annotator, job):
        result = annotator.annotate(job)
        assert len(result.to_feature_vector()) == 19

    def test_first_18_positions_are_demand_scores(self, annotator, job):
        result = annotator.annotate(job)
        vec = result.to_feature_vector()
        for i in range(18):
            assert vec[i] == 2.0, f"Position {i} should be 2.0, got {vec[i]}"

    def test_last_position_is_ug_score(self, annotator, job):
        result = annotator.annotate(job)
        vec = result.to_feature_vector()
        assert abs(vec[18] - 75.0) < 1e-6

    def test_ordering_matches_dimension_order(self, annotator, job):
        """vec[i] == demands[DEMAND_DIMENSIONS[i]].level for i in 0..17."""
        result = annotator.annotate(job)
        vec = result.to_feature_vector()
        for i, dim in enumerate(DEMAND_DIMENSIONS):
            assert vec[i] == result.demands[dim].level, (
                f"Position {i} ({dim}): expected {result.demands[dim].level}, got {vec[i]}"
            )

    def test_missing_dimension_fills_with_nan(self, annotator, job):
        result = annotator.annotate(job)
        del result.demands["AS"]
        vec = result.to_feature_vector()
        as_idx = list(DEMAND_DIMENSIONS).index("AS")
        assert math.isnan(vec[as_idx])
        for i in range(18):
            if i != as_idx:
                assert not math.isnan(vec[i])


# ---------------------------------------------------------------------------
# annotate_dataset
# ---------------------------------------------------------------------------

class TestAnnotateDataset:

    def test_tensor_shape(self, stub, catalog):
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=None)
        jobs = [
            AnnotationJob(f"item_{i}", f"content {i}", f"answer {i}")
            for i in range(3)
        ]
        dv = annotator.annotate_dataset(jobs)
        assert dv.tensor.shape == (3, 19)

    def test_item_ids_in_input_order(self, stub, catalog):
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=None)
        jobs = [
            AnnotationJob(f"item_{i}", f"content {i}", f"answer {i}")
            for i in range(4)
        ]
        dv = annotator.annotate_dataset(jobs)
        assert dv.item_ids == [f"item_{i}" for i in range(4)]

    def test_row_order_matches_job_order(self, stub, catalog):
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=None)
        jobs = [
            AnnotationJob(f"item_{i}", f"content {i}", f"answer {i}")
            for i in range(3)
        ]
        dv = annotator.annotate_dataset(jobs)
        for i in range(len(jobs)):
            assert dv.item_ids[i] == jobs[i].item_id

    def test_total_api_calls_is_19_per_item(self, catalog):
        stub = StubClient()
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=None)
        n_items = 3
        jobs = [
            AnnotationJob(f"item_{i}", f"content {i}", f"answer {i}")
            for i in range(n_items)
        ]
        annotator.annotate_dataset(jobs)
        assert len(stub.calls) == n_items * 19


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

class TestCacheIntegration:

    def test_cache_hit_prevents_api_call(self, catalog, tmp_path):
        stub = StubClient()
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=cache)
        job = AnnotationJob("i1", "What is 2+2?", "4")

        annotator.annotate(job)
        assert len(stub.calls) == 19

        # Second annotation of same item: all served from cache
        annotator.annotate(job)
        assert len(stub.calls) == 19  # no new calls

    def test_cache_persists_across_annotator_instances(self, catalog, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        job = AnnotationJob("i1", "Test content", "Test answer")

        stub1 = StubClient()
        DemandAnnotator(client=stub1, rubrics=catalog, cache=AnnotationCache(cache_path)).annotate(job)
        assert len(stub1.calls) == 19

        stub2 = StubClient()
        DemandAnnotator(client=stub2, rubrics=catalog, cache=AnnotationCache(cache_path)).annotate(job)
        assert len(stub2.calls) == 0  # all from cache

    def test_different_items_both_hit_api(self, catalog, tmp_path):
        stub = StubClient()
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=cache)

        job1 = AnnotationJob("i1", "Content A", "Answer A")
        job2 = AnnotationJob("i2", "Content B", "Answer B")
        annotator.annotate(job1)
        annotator.annotate(job2)
        assert len(stub.calls) == 38  # 19 per unique item

    def test_item_id_is_correct_on_cache_hit(self, catalog, tmp_path):
        """Cache key excludes item_id; cache hit with different item_id
        must still return annotation with the CURRENT item_id."""
        stub = StubClient()
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=cache)

        same_content = "What is the capital of France?"
        job1 = AnnotationJob("item_alpha", same_content, "Paris")
        job2 = AnnotationJob("item_beta", same_content, "Paris")  # same content, different ID

        annotator.annotate(job1)
        result2 = annotator.annotate(job2)

        # Cache hit (same content → same key) but item_id must be job2's
        assert result2.item_id == "item_beta"

    def test_no_cache_uses_api_every_time(self, catalog):
        stub = StubClient()
        annotator = DemandAnnotator(client=stub, rubrics=catalog, cache=None)
        job = AnnotationJob("i1", "Content", "Answer")

        annotator.annotate(job)
        annotator.annotate(job)
        assert len(stub.calls) == 38  # called twice, 19 each time
