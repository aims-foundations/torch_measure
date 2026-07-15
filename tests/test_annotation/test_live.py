# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Live end-to-end tests — require a real API key.

These tests make real API calls and consume quota. Run only when you want
to verify actual model behavior.

Usage (Gemini):
    $env:GEMINI_API_KEY = "<key>"
    pytest tests/test_annotation/test_live.py -v -s -m "network and slow"

Usage (Claude):
    $env:ANNOTATOR_CLIENT = "claude"
    $env:ANTHROPIC_API_KEY = "<key>"
    pytest tests/test_annotation/test_live.py -v -s -m "network and slow"

Usage (OpenAI):
    $env:ANNOTATOR_CLIENT = "openai"
    $env:OPENAI_API_KEY = "<key>"
    pytest tests/test_annotation/test_live.py -v -s -m "network and slow"

Skip automatically if the required API key is not set.
"""

import math
import os

import pytest

from torch_measure.annotation import (
    AnnotationCache,
    AnnotationJob,
    ClaudeClient,
    DemandAnnotator,
    GeminiClient,
    OpenAIClient,
    RubricsCatalog,
)
from torch_measure.annotation._types import DEMAND_DIMENSIONS

pytestmark = [pytest.mark.network, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client_type():
    return os.environ.get("ANNOTATOR_CLIENT", "gemini").strip().lower()


@pytest.fixture(scope="module")
def api_key(client_type):
    if client_type == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    elif client_type == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
    else:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            pytest.skip("GEMINI_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def model_id(client_type):
    if client_type == "claude":
        return os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
    if client_type == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


@pytest.fixture(scope="module")
def live_annotator(api_key, model_id, client_type, tmp_path_factory):
    if client_type == "claude":
        client = ClaudeClient(api_key=api_key, model=model_id)
    elif client_type == "openai":
        client = OpenAIClient(api_key=api_key, model=model_id)
    else:
        client = GeminiClient(api_key=api_key, model=model_id)
    rubrics = RubricsCatalog()
    cache_dir = tmp_path_factory.mktemp("live_annotation_cache")
    cache = AnnotationCache(cache_dir / "cache.jsonl")
    print(f"\nClient: {client_type}  Model: {model_id}")
    return DemandAnnotator(client=client, rubrics=rubrics, cache=cache)


# Test items chosen to have predictable characteristics.
OPEN_ENDED_ITEM = AnnotationJob(
    item_id="live_open_001",
    content="What is the capital of France?",
    reference_answer="Paris",
)

MCQ_ITEM = AnnotationJob(
    item_id="live_mcq_001",
    content=(
        "Which of the following is a planet in our solar system?\n"
        "A) Sun\nB) Moon\nC) Mars\nD) Comet"
    ),
    reference_answer="C) Mars",
)

COMPLEX_ITEM = AnnotationJob(
    item_id="live_complex_001",
    content=(
        "Prove that for all positive integers n, the sum 1 + 2 + ... + n = n(n+1)/2 "
        "using mathematical induction."
    ),
    reference_answer="Base case: n=1, sum=1=1(2)/2. Inductive step: assume true for k, show k+1.",
)


# ---------------------------------------------------------------------------
# Structural correctness
# ---------------------------------------------------------------------------

class TestLiveStructure:

    def test_returns_19_dimensional_vector(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert len(result.to_feature_vector()) == 19

    def test_all_18_demand_dimensions_present(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert set(result.demands.keys()) == set(DEMAND_DIMENSIONS)

    def test_ug_annotation_present(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert result.ug is not None

    def test_model_responses_stored_for_all_dimensions(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        for dim in DEMAND_DIMENSIONS:
            assert len(result.demands[dim].model_response) > 0, (
                f"{dim} has empty model_response"
            )
        assert len(result.ug.model_response) > 0


# ---------------------------------------------------------------------------
# Score ranges
# ---------------------------------------------------------------------------

class TestLiveScoreRanges:

    def test_demand_scores_in_valid_range(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        for dim in DEMAND_DIMENSIONS:
            level = result.demands[dim].level
            assert math.isnan(level) or 0.0 <= level <= 5.0, (
                f"{dim} level {level} outside [0, 5]"
            )

    def test_ug_score_in_valid_range(self, live_annotator):
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        ug = result.ug.ug_score
        assert math.isnan(ug) or 0.0 <= ug <= 100.0, f"UG score {ug} outside [0, 100]"

    def test_no_demand_parse_failures_on_simple_item(self, live_annotator):
        """A simple factual question should produce parseable scores for all 18 dimensions."""
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        failures = [dim for dim in DEMAND_DIMENSIONS if math.isnan(result.demands[dim].level)]
        assert len(failures) == 0, (
            f"Parse failures on simple item for dimensions: {failures}"
        )


# ---------------------------------------------------------------------------
# UG classification
# ---------------------------------------------------------------------------

class TestLiveUG:

    def test_open_ended_item_classified_as_open(self, live_annotator):
        """'What is the capital of France?' should be open-ended → ug_score=100."""
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert result.ug.ug_score == 100.0, (
            f"Expected open-ended classification (100.0), got {result.ug.ug_score}. "
            f"Raw output: {repr(result.ug.raw_output)}"
        )

    def test_mcq_item_classified_as_mcq(self, live_annotator):
        """Explicit 4-choice MCQ should be classified with N=4 → ug_score=75.0."""
        result = live_annotator.annotate(MCQ_ITEM)
        assert result.ug.ug_score < 100.0, (
            f"MCQ item incorrectly classified as open-ended. "
            f"Raw output: {repr(result.ug.raw_output)}"
        )


# ---------------------------------------------------------------------------
# Finish reason
# ---------------------------------------------------------------------------

class TestLiveFinishReason:

    def test_all_finish_reasons_normalized(self, live_annotator):
        """All finish reasons must be 'stop', 'length', or 'other'."""
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        valid = {"stop", "length", "other"}
        for dim in DEMAND_DIMENSIONS:
            fr = result.demands[dim].finish_reason
            assert fr in valid, f"{dim} finish_reason {repr(fr)} not in {valid}"
        assert result.ug.finish_reason in valid

    def test_simple_item_finishes_with_stop(self, live_annotator):
        """A simple item should not trigger token limit."""
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        length_finishes = [
            dim for dim in DEMAND_DIMENSIONS
            if result.demands[dim].finish_reason == "length"
        ]
        assert len(length_finishes) == 0, (
            f"Unexpected 'length' finish on simple item for: {length_finishes}"
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestLiveCache:

    def test_second_call_returns_identical_vector(self, live_annotator):
        """Cache hit returns bit-identical results to original API call."""
        result1 = live_annotator.annotate(OPEN_ENDED_ITEM)
        result2 = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert result1.to_feature_vector() == result2.to_feature_vector()

    def test_second_call_has_same_ug_score(self, live_annotator):
        result1 = live_annotator.annotate(OPEN_ENDED_ITEM)
        result2 = live_annotator.annotate(OPEN_ENDED_ITEM)
        assert result1.ug.ug_score == result2.ug.ug_score


# ---------------------------------------------------------------------------
# Semantic sanity checks (not strict pass/fail — advisory)
# ---------------------------------------------------------------------------

class TestLiveSemantics:

    def test_complex_item_has_higher_qll_than_simple_item(self, live_annotator):
        """Mathematical induction proof should score higher on QLl (logical reasoning)
        than a simple factual recall question. Not a strict guarantee but a strong prior.
        """
        simple = live_annotator.annotate(OPEN_ENDED_ITEM)
        complex_ = live_annotator.annotate(COMPLEX_ITEM)
        simple_qll = simple.demands["QLl"].level
        complex_qll = complex_.demands["QLl"].level
        if math.isnan(simple_qll) or math.isnan(complex_qll):
            pytest.skip("Parse failure on QLl — cannot compare")
        # This is an advisory check, not a hard requirement
        if simple_qll >= complex_qll:
            pytest.xfail(
                f"Expected complex item (QLl={complex_qll}) > simple item (QLl={simple_qll}), "
                "but model disagreed. This is a semantic sanity check, not a hard pass/fail."
            )


# ---------------------------------------------------------------------------
# Output inspection — prints all 19 scores for manual comparison to paper
# ---------------------------------------------------------------------------

class TestLiveOutputInspection:

    def test_print_full_annotation_output(self, live_annotator):
        """Prints all 19 scores for OPEN_ENDED_ITEM. Uses cached result — 0 API calls.
        Run with -s to see output. Compare to paper's expected annotation for this item.
        """
        result = live_annotator.annotate(OPEN_ENDED_ITEM)
        print(f"\n{'='*60}")
        print(f"Item: {OPEN_ENDED_ITEM.content!r}")
        print(f"{'='*60}")
        print(f"{'Dim':<8} {'Score':>6}  {'Finish':<8}  Response excerpt")
        print(f"{'-'*60}")
        for dim in DEMAND_DIMENSIONS:
            ann = result.demands[dim]
            score = f"{ann.level:.1f}" if not math.isnan(ann.level) else "NaN"
            excerpt = ann.model_response[:60].replace("\n", " ")
            print(f"{dim:<8} {score:>6}  {ann.finish_reason:<8}  {excerpt}")
        ug = result.ug
        ug_score = f"{ug.ug_score:.1f}" if not math.isnan(ug.ug_score) else "NaN"
        print(f"{'UG':<8} {ug_score:>6}  {ug.finish_reason:<8}  {ug.model_response[:60].replace(chr(10), ' ')}")
        print(f"{'='*60}")
        print(f"Feature vector: {[round(v, 2) for v in result.to_feature_vector()]}")
        print(f"{'='*60}\n")
