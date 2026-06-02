# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for :class:`ColdStartLookupPredictor`.

Coverage targets:
    - All 6 fallback levels of the lookup hierarchy
    - Name resolution: provider prefixes, case-insensitive matches, aliases
    - IRT-blend math correctness against a closed-form expectation
    - Platt calibration: intercept-only, shift cap, no-global-contamination
    - Round-trip JSON persistence
    - Numerical safety: extreme probabilities, missing keys
"""

from __future__ import annotations

import json
import math

import pytest

from torch_measure.models import ColdStartLookupPredictor
from torch_measure.models.cold_start_lookup import (
    _logit,
    _sigmoid,
    parse_subject_name,
    resolve_subject_name,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def lookup_tables() -> dict:
    """Hand-crafted lookup mirroring the structure built by m3_build_colab.py.

    Three subjects, three benchmarks, a deliberately-sparse triple table so
    each fallback level can be exercised by at least one test.
    """
    return {
        "sbc": {
            "gpt-4||mmlupro||zero-shot": 0.78,
            "gpt-4||mmlupro||cot": 0.85,
            "claude-3||ai2d_test||none": 0.72,
        },
        "sb": {
            "gpt-4||mmlupro": 0.80,
            "gpt-4||cybench": 0.40,
            "claude-3||ai2d_test": 0.72,
            "llama-2-7b||cybench": 0.18,
        },
        "subj": {
            "gpt-4": 0.75,
            "claude-3": 0.70,
            "llama-2-7b": 0.35,
        },
        "bench": {
            "mmlupro": 0.50,
            "cybench": 0.25,
            "ai2d_test": 0.70,
        },
        "global": 0.645,
        "name_aliases": {"Llama-2-7b-chat": "llama-2-7b"},
        "name_lc": {
            "gpt-4": "gpt-4",
            "claude-3": "claude-3",
            "llama-2-7b": "llama-2-7b",
        },
    }


@pytest.fixture
def predictor(lookup_tables: dict) -> ColdStartLookupPredictor:
    return ColdStartLookupPredictor(
        sbc=lookup_tables["sbc"],
        sb=lookup_tables["sb"],
        subj=lookup_tables["subj"],
        bench=lookup_tables["bench"],
        global_mean=lookup_tables["global"],
        name_aliases=lookup_tables["name_aliases"],
        name_lc=lookup_tables["name_lc"],
    )


# ---- Helpers ---------------------------------------------------------------


def _make_record(benchmark: str, condition: str, subject_name: str) -> dict:
    return {
        "benchmark": benchmark,
        "condition": condition,
        "subject_content": f"Name: {subject_name}",
        "item_content": "irrelevant for lookup",
    }


# ---- Level-by-level lookup tests --------------------------------------------


class TestLookupLevels:
    """Each test exercises one fallback level and asserts the expected source."""

    def test_level1_triple_match(self, predictor: ColdStartLookupPredictor) -> None:
        p = predictor.predict(_make_record("mmlupro", "zero-shot", "gpt-4"))
        assert abs(p - 0.78) < 1e-6

    def test_level1_falls_to_none_within_level(self, predictor: ColdStartLookupPredictor) -> None:
        """Unknown condition: tries `none` condition before falling out to L2."""
        # claude-3 has sbc[claude-3||ai2d_test||none] = 0.72.
        # An unseen condition `xyz` should still hit level 1 via the `none` fallback.
        p = predictor.predict(_make_record("ai2d_test", "xyz", "claude-3"))
        assert abs(p - 0.72) < 1e-6

    def test_level2_pair_match(self, predictor: ColdStartLookupPredictor) -> None:
        """Unseen condition with no level-1 match -> level 2 (subject, benchmark)."""
        p = predictor.predict(_make_record("mmlupro", "few-shot", "gpt-4"))
        assert abs(p - 0.80) < 1e-6

    def test_level3_irt_blend(self, predictor: ColdStartLookupPredictor) -> None:
        """Both single priors known, no pair: closed-form 1-PL Rasch blend."""
        p = predictor.predict(_make_record("ai2d_test", "none", "gpt-4"))
        expected = _sigmoid(_logit(0.75) + _logit(0.70) - _logit(0.645))
        assert abs(p - expected) < 1e-6

    def test_level4_bench_only(self, predictor: ColdStartLookupPredictor) -> None:
        """Unknown subject, known benchmark -> benchmark prior."""
        p = predictor.predict(_make_record("mmlupro", "none", "SomeUnknownModel"))
        assert abs(p - 0.50) < 1e-6

    def test_level5_subj_only(self, predictor: ColdStartLookupPredictor) -> None:
        """Known subject, unknown benchmark -> subject prior."""
        p = predictor.predict(_make_record("newbench_x", "none", "gpt-4"))
        assert abs(p - 0.75) < 1e-6

    def test_level6_global_fallback(self, predictor: ColdStartLookupPredictor) -> None:
        """Neither known -> global mean."""
        p = predictor.predict(_make_record("newbench_x", "none", "AlsoUnknown"))
        assert abs(p - 0.645) < 1e-6


# ---- Name resolution -------------------------------------------------------


class TestNameResolution:
    def test_provider_prefix_stripped(self, predictor: ColdStartLookupPredictor) -> None:
        """meta-llama/Llama-2-7b-chat -> alias -> llama-2-7b -> sb hit."""
        p = predictor.predict(_make_record("cybench", "none", "meta-llama/Llama-2-7b-chat"))
        assert abs(p - 0.18) < 1e-6

    def test_case_insensitive_subject(self, predictor: ColdStartLookupPredictor) -> None:
        """GPT-4 (different case) resolves to gpt-4 via the lower-case map."""
        p = predictor.predict(_make_record("mmlupro", "zero-shot", "GPT-4"))
        assert abs(p - 0.78) < 1e-6

    def test_parse_subject_name_from_multiline(self) -> None:
        sc = "Name: gpt-4\nOrganization: OpenAI\nParameters: unknown"
        assert parse_subject_name(sc) == "gpt-4"

    def test_parse_subject_name_no_label_falls_to_first_line(self) -> None:
        # If `Name:` is missing the first line is taken as a heuristic.
        assert parse_subject_name("gpt-4\nfoo\nbar") == "gpt-4"

    def test_parse_subject_name_empty(self) -> None:
        assert parse_subject_name("") == ""
        assert parse_subject_name(None) == ""  # type: ignore[arg-type]

    def test_resolve_unknown_name_returns_raw(self, lookup_tables: dict) -> None:
        """A name with no provider prefix and no alias falls through untouched."""
        out = resolve_subject_name(
            "TotallyNewModel-V99",
            subj_lookup=lookup_tables["subj"],
            sb_lookup=lookup_tables["sb"],
            sbc_lookup=lookup_tables["sbc"],
            aliases=lookup_tables["name_aliases"],
            name_lc=lookup_tables["name_lc"],
        )
        assert out == "TotallyNewModel-V99"


# ---- Adaptive calibration --------------------------------------------------


class TestCalibration:
    def test_calibration_shifts_target_benchmark(self, predictor: ColdStartLookupPredictor) -> None:
        """Labels that look hard pull predictions DOWN for that benchmark."""
        labeled = [
            {"benchmark": "mmlupro", "condition": "zero-shot", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "condition": "zero-shot", "subject_content": "Name: claude-3", "label": 0},
            {"benchmark": "mmlupro", "condition": "zero-shot", "subject_content": "Name: llama-2-7b", "label": 0},
            {"benchmark": "mmlupro", "condition": "zero-shot", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "condition": "zero-shot", "subject_content": "Name: claude-3", "label": 1},
        ]
        raw = predictor.predict(_make_record("mmlupro", "zero-shot", "gpt-4"))
        cal = predictor.predict(_make_record("mmlupro", "zero-shot", "gpt-4"), labeled)
        assert cal < raw - 0.01, f"Expected calibration to lower the prediction; got raw={raw:.4f}, cal={cal:.4f}"

    def test_calibration_does_not_contaminate_other_benchmarks(self, predictor: ColdStartLookupPredictor) -> None:
        """A mmlupro-labeled list must not change cybench predictions."""
        labeled = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: llama-2-7b", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 1},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 0},
        ]
        raw = predictor.predict(_make_record("cybench", "none", "gpt-4"))
        cal = predictor.predict(_make_record("cybench", "none", "gpt-4"), labeled)
        assert abs(cal - raw) < 1e-6, f"cybench prediction changed by mmlupro calibration: raw={raw:.4f}, cal={cal:.4f}"

    def test_calibration_skips_singletons(self, predictor: ColdStartLookupPredictor) -> None:
        """A single labeled example for a benchmark should not produce a calibrator."""
        labeled = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
        ]
        predictor.calibrate(labeled)
        assert "mmlupro" not in predictor._platt

    def test_calibration_skips_malformed_labels(self, predictor: ColdStartLookupPredictor) -> None:
        labeled = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": "bad"},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 1},
            {"benchmark": "mmlupro", "subject_content": "Name: llama-2-7b", "label": 0},
        ]
        predictor.calibrate(labeled)
        assert "mmlupro" in predictor._platt

    def test_calibration_shift_is_capped(self, predictor: ColdStartLookupPredictor) -> None:
        """All-zeros labels would imply an infinite negative shift; we cap it."""
        labeled = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
        ]
        # mean_y = 0 -> calibrator skipped (logit undefined). No entry, no crash.
        predictor.calibrate(labeled)
        assert "mmlupro" not in predictor._platt

    def test_calibration_is_deterministic_for_identical_input(self, predictor: ColdStartLookupPredictor) -> None:
        """Repeated calls with the SAME labeled list produce identical Platt state.

        Earlier this method was a `len(labeled)`-keyed cache short-circuit. The
        cache was removed in the Lane B / PR #2 v2 round because library callers
        may pass distinct labeled lists of equal length within one process, and
        a length-key cache would silently return stale shifts for the second
        call. The behavior we now pin is determinism on identical input.
        """
        labeled = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 1},
        ]
        predictor.calibrate(labeled)
        first_state = dict(predictor._platt)
        predictor.calibrate(labeled)
        assert predictor._platt == first_state

    def test_calibration_refits_on_same_length_different_content(self, predictor: ColdStartLookupPredictor) -> None:
        """A second calibrate() with same length but different LABELS must
        produce different Platt shifts. Regression guard against the
        previously-cached ``_platt_fit_key = len(labeled)`` short-circuit
        that would have returned the FIRST fit's shifts unchanged.
        """
        labeled_all_zeros = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 0},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 1},
        ]
        labeled_mostly_ones = [
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 1},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 1},
            {"benchmark": "mmlupro", "subject_content": "Name: gpt-4", "label": 1},
            {"benchmark": "mmlupro", "subject_content": "Name: claude-3", "label": 0},
        ]
        assert len(labeled_all_zeros) == len(labeled_mostly_ones)

        predictor.calibrate(labeled_all_zeros)
        shift_low = predictor._platt["mmlupro"][1]

        predictor.calibrate(labeled_mostly_ones)
        shift_high = predictor._platt["mmlupro"][1]

        # mostly_ones should push the shift UP (positive) relative to all_zeros.
        assert shift_high > shift_low + 0.5, (
            f"calibrate did not refit on same-length-different-content; "
            f"shift_low={shift_low:.4f}, shift_high={shift_high:.4f}. "
            "Did a len(labeled)-key cache get reintroduced?"
        )


# ---- IRT-blend math (level 3) ----------------------------------------------


class TestIRTBlend:
    def test_blend_with_unit_global(self) -> None:
        """If global mean equals subj prior, the blend reduces to bench prior."""
        predictor = ColdStartLookupPredictor(
            sbc={},
            sb={},
            subj={"X": 0.6},
            bench={"B": 0.4},
            global_mean=0.6,
            name_aliases={},
            name_lc={"x": "X"},
        )
        p = predictor.predict({"benchmark": "B", "condition": "none", "subject_content": "Name: X", "item_content": ""})
        assert abs(p - 0.4) < 1e-6

    def test_blend_above_global_pushes_up(self) -> None:
        """High subj + high bench relative to global -> blend > both single priors."""
        predictor = ColdStartLookupPredictor(
            sbc={},
            sb={},
            subj={"X": 0.8},
            bench={"B": 0.7},
            global_mean=0.5,
            name_aliases={},
            name_lc={"x": "X"},
        )
        p = predictor.predict({"benchmark": "B", "condition": "none", "subject_content": "Name: X", "item_content": ""})
        assert p > 0.8 and p < 0.95


# ---- Persistence -----------------------------------------------------------


class TestPersistence:
    def test_round_trip_via_json(self, predictor: ColdStartLookupPredictor, tmp_path) -> None:
        path = tmp_path / "lookup.json"
        predictor.to_lookup_json(path)
        loaded = ColdStartLookupPredictor.from_lookup_json(path)

        sample = _make_record("mmlupro", "zero-shot", "gpt-4")
        assert abs(predictor.predict(sample) - loaded.predict(sample)) < 1e-12

    def test_from_lookup_json_schema_match(self, tmp_path) -> None:
        payload = {
            "sbc": {"a||b||c": 0.5},
            "sb": {"a||b": 0.5},
            "subj": {"a": 0.5},
            "bench": {"b": 0.5},
            "global": 0.5,
        }
        path = tmp_path / "lookup.json"
        path.write_text(json.dumps(payload))
        predictor = ColdStartLookupPredictor.from_lookup_json(path)
        assert predictor.global_mean == 0.5
        assert predictor.name_aliases == {}
        assert predictor.name_lc == {}


# ---- Numerical safety ------------------------------------------------------


class TestNumericalSafety:
    def test_extreme_probabilities_clipped(self) -> None:
        """A prior of 0.999 in the lookup should still produce a finite logit."""
        predictor = ColdStartLookupPredictor(
            sbc={},
            sb={},
            subj={"X": 0.999},
            bench={"B": 0.001},
            global_mean=0.5,
            name_aliases={},
            name_lc={"x": "X"},
        )
        # Level 3 IRT blend: logit(0.999) + logit(0.001) - logit(0.5).
        # The two extremes nearly cancel and the result is close to 0.5.
        p = predictor.predict({"benchmark": "B", "condition": "none", "subject_content": "Name: X", "item_content": ""})
        assert math.isfinite(p)
        assert 0.0 < p < 1.0

    def test_empty_input_returns_global(self) -> None:
        predictor = ColdStartLookupPredictor(
            sbc={},
            sb={},
            subj={},
            bench={},
            global_mean=0.42,
            name_aliases={},
            name_lc={},
        )
        p = predictor.predict({"benchmark": "", "condition": "", "subject_content": "", "item_content": ""})
        assert abs(p - 0.42) < 1e-6

    def test_output_clipping_respected(self) -> None:
        """Output clip should bound predictions even when lookup returns 1.0."""
        predictor = ColdStartLookupPredictor(
            sbc={"x||b||none": 1.0},
            sb={},
            subj={},
            bench={},
            global_mean=0.5,
            name_aliases={},
            name_lc={"x": "x"},
            output_clip=(0.05, 0.95),
        )
        p = predictor.predict({"benchmark": "b", "condition": "none", "subject_content": "Name: x", "item_content": ""})
        assert p == 0.95
