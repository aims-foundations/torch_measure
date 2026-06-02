# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for :class:`LLMJudgeIRT` and its difficulty prompt builder.

The judge function itself (a frozen LLM call) is mocked throughout these
tests; we are not testing the LLM, we are testing the IRT combination,
parameter fitting, and graceful-failure semantics.
"""

from __future__ import annotations

import math

import pytest

from torch_measure.experimental import LLMJudgeIRT, build_difficulty_prompt
from torch_measure.models import ColdStartLookupPredictor

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def minimal_lookup() -> ColdStartLookupPredictor:
    """A tiny lookup that always falls through to a level-3 IRT blend."""
    return ColdStartLookupPredictor(
        sbc={},
        sb={},
        subj={"gpt-4": 0.75, "claude-3": 0.70, "llama-2-7b": 0.35},
        bench={"mmlupro": 0.50, "cybench": 0.25, "ai2d_test": 0.70},
        global_mean=0.645,
        name_aliases={},
        name_lc={"gpt-4": "gpt-4", "claude-3": "claude-3"},
    )


# ---- Prompt construction ---------------------------------------------------


class TestDifficultyPrompt:
    def test_prompt_contains_benchmark_name(self) -> None:
        prompt = build_difficulty_prompt("What is 2+2?", "mmlupro")
        assert "`mmlupro`" in prompt

    def test_prompt_truncates_item_to_max_chars(self) -> None:
        long_item = "x" * 5000
        prompt = build_difficulty_prompt(long_item, "mmlupro", max_item_chars=100)
        # The prompt should still contain "Question:" but the item portion is bounded.
        assert "Question:" in prompt
        question_start = prompt.index("Question:") + len("Question:\n")
        question_end = prompt.index("\n\nIs this question difficult?")
        item_in_prompt = prompt[question_start:question_end]
        assert len(item_in_prompt) <= 100

    def test_prompt_ends_with_answer_marker(self) -> None:
        """The judge reads next-token logprobs at the position right after `Answer:`."""
        prompt = build_difficulty_prompt("anything", "mmlupro")
        assert prompt.rstrip().endswith("Answer:")

    def test_prompt_does_not_mention_specific_subject(self) -> None:
        """By design the difficulty prompt is subject-independent."""
        prompt = build_difficulty_prompt("Q", "mmlupro")
        # The prompt should NOT contain any explicit subject identifiers.
        for forbidden in ("gpt-4", "Name:", "claude"):
            assert forbidden.lower() not in prompt.lower()


# ---- Predictor behaviour ---------------------------------------------------


class TestPredict:
    def test_alpha_zero_recovers_lookup_logit(self, minimal_lookup: ColdStartLookupPredictor) -> None:
        """alpha=0, beta=0 -> predict matches sigmoid(logit(lookup)). Should equal lookup."""

        def constant_judge(item: str, bench: str) -> float:
            return 7.5  # any value -- multiplied by zero anyway

        model = LLMJudgeIRT(
            lookup=minimal_lookup,
            judge_fn=constant_judge,
            alpha=0.0,
            beta=0.0,
        )
        record = {
            "benchmark": "mmlupro",
            "condition": "none",
            "subject_content": "Name: gpt-4",
            "item_content": "any item",
        }
        p_irt = model.predict(record)
        p_lookup = minimal_lookup.predict(record)
        assert abs(p_irt - p_lookup) < 1e-6

    def test_positive_alpha_with_hard_judge_lowers_prediction(self, minimal_lookup: ColdStartLookupPredictor) -> None:
        """alpha>0 and judge says yes (hard) -> delta>0 -> P(correct) decreases."""

        def hard_judge(item: str, bench: str) -> float:
            return 1.0  # yes > no

        model = LLMJudgeIRT(
            lookup=minimal_lookup,
            judge_fn=hard_judge,
            alpha=0.5,
            beta=0.0,
        )
        record = {
            "benchmark": "mmlupro",
            "condition": "none",
            "subject_content": "Name: gpt-4",
            "item_content": "hard q",
        }
        p_irt = model.predict(record)
        p_lookup = minimal_lookup.predict(record)
        assert p_irt < p_lookup

    def test_positive_alpha_with_easy_judge_raises_prediction(self, minimal_lookup: ColdStartLookupPredictor) -> None:
        def easy_judge(item: str, bench: str) -> float:
            return -1.0  # no > yes -> easy

        model = LLMJudgeIRT(
            lookup=minimal_lookup,
            judge_fn=easy_judge,
            alpha=0.5,
            beta=0.0,
        )
        record = {
            "benchmark": "mmlupro",
            "condition": "none",
            "subject_content": "Name: gpt-4",
            "item_content": "easy q",
        }
        p_irt = model.predict(record)
        p_lookup = minimal_lookup.predict(record)
        assert p_irt > p_lookup

    def test_judge_exception_handled_gracefully(self, minimal_lookup: ColdStartLookupPredictor) -> None:
        """A raised exception from the judge should degrade to the lookup value."""

        def broken_judge(item: str, bench: str) -> float:
            raise RuntimeError("simulated judge failure")

        model = LLMJudgeIRT(
            lookup=minimal_lookup,
            judge_fn=broken_judge,
            alpha=0.5,
            beta=0.0,
        )
        record = {
            "benchmark": "mmlupro",
            "condition": "none",
            "subject_content": "Name: gpt-4",
            "item_content": "q",
        }
        p_irt = model.predict(record)
        p_lookup = minimal_lookup.predict(record)
        # judge_logit collapses to 0 -> delta=0 -> recover lookup.
        assert abs(p_irt - p_lookup) < 1e-6

    def test_logit_cap_bounds_extreme_outputs(self, minimal_lookup: ColdStartLookupPredictor) -> None:
        """Even with extreme judge logits, output stays inside output_clip."""

        def extreme_judge(item: str, bench: str) -> float:
            return 100.0

        model = LLMJudgeIRT(
            lookup=minimal_lookup,
            judge_fn=extreme_judge,
            alpha=10.0,  # large slope
            beta=0.0,
            logit_cap=3.0,
        )
        record = {
            "benchmark": "mmlupro",
            "condition": "none",
            "subject_content": "Name: gpt-4",
            "item_content": "q",
        }
        p = model.predict(record)
        lo, hi = minimal_lookup.output_clip
        assert lo <= p <= hi
        assert math.isfinite(p)


# ---- Parameter fitting ----------------------------------------------------


class TestFitAlphaBeta:
    def test_zero_signal_yields_near_zero_alpha(self) -> None:
        """If judge_logits have no relationship with labels, alpha shrinks toward 0."""
        pytest.importorskip("scipy")
        import numpy as np  # noqa: F401

        rng_thetas = [0.0] * 200
        rng_jl = [(i % 7) * 0.1 - 0.3 for i in range(200)]
        # Labels uncorrelated with jl.
        rng_labels = [i % 2 for i in range(200)]
        alpha, beta, _ = LLMJudgeIRT.fit_alpha_beta(
            rng_thetas,
            rng_jl,
            rng_labels,
            l2_penalty=2.0,
        )
        assert abs(alpha) < 0.2, f"alpha should be near 0 for noise, got {alpha}"

    def test_strong_signal_recovers_alpha_sign(self) -> None:
        """With a clean positive signal, the fitted alpha should be > 0."""
        pytest.importorskip("scipy")

        # Construct a dataset where higher judge_logit -> lower label probability,
        # which under the IRT formula P = sigmoid(theta - alpha*jl - beta) implies
        # alpha > 0 is needed to drive P down when jl is high.
        n = 400
        thetas = [0.5] * n
        judge_logits = []
        labels = []
        for i in range(n):
            jl = (i / n - 0.5) * 4  # range [-2, 2]
            # True alpha=1.5, beta=0: P = sigmoid(0.5 - 1.5*jl)
            p_true = 1.0 / (1.0 + math.exp(-(0.5 - 1.5 * jl)))
            judge_logits.append(jl)
            labels.append(1 if p_true > 0.5 else 0)
        alpha, beta, _ = LLMJudgeIRT.fit_alpha_beta(
            thetas,
            judge_logits,
            labels,
            l2_penalty=0.0,
        )
        assert alpha > 0.5, f"alpha should clearly be positive, got {alpha}"

    def test_l2_penalty_shrinks_alpha(self) -> None:
        """Increasing L2 penalty should shrink the absolute value of alpha."""
        pytest.importorskip("scipy")

        n = 100
        thetas = [0.0] * n
        judge_logits = [(-1.0 + 2 * i / n) for i in range(n)]
        labels = [1 if jl < 0 else 0 for jl in judge_logits]

        alpha_no_l2, _, _ = LLMJudgeIRT.fit_alpha_beta(
            thetas,
            judge_logits,
            labels,
            l2_penalty=0.0,
        )
        alpha_strong_l2, _, _ = LLMJudgeIRT.fit_alpha_beta(
            thetas,
            judge_logits,
            labels,
            l2_penalty=100.0,
        )
        assert abs(alpha_strong_l2) < abs(alpha_no_l2)
