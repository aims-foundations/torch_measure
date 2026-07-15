# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for _parsers.py — zero API calls, zero file I/O.

These tests cover the verbatim ADeLe paper parsing logic and must pass
entirely offline. They are the highest-confidence signal that the paper's
methodology is faithfully reproduced.
"""

import math

import pytest

from torch_measure.annotation._parsers import extract_demand_level, extract_ug_score


# ---------------------------------------------------------------------------
# extract_demand_level
# ---------------------------------------------------------------------------

class TestExtractDemandLevel:

    # --- canonical success path ---

    def test_standard_cot_conclusion(self):
        """Normal CoT response ending with the paper's conclusion sentence."""
        response = (
            "Step 1: The task is a simple factual question.\n\n"
            "Step 2: No computation is required.\n\n"
            "Thus, the level of *Attention and Search* demanded by the given "
            "TASK INSTANCE is: 3"
        )
        assert extract_demand_level(response) == 3.0

    def test_score_zero(self):
        assert extract_demand_level("Analysis.\n\nThus, the level is: 0") == 0.0

    def test_score_five(self):
        assert extract_demand_level("Complex.\n\nThe level is 5.") == 5.0

    def test_returns_float_not_int(self):
        result = extract_demand_level("Thus, the level is 2")
        assert isinstance(result, float)

    # --- split on \n\n ---

    def test_uses_last_double_newline_segment(self):
        """Parser splits on \\n\\n and takes the LAST segment."""
        response = "Segment one has digit 1.\n\nSegment two has digit 2.\n\nFinal: 4"
        assert extract_demand_level(response) == 4.0

    def test_no_double_newline_uses_whole_response(self):
        assert extract_demand_level("The level is 2") == 2.0

    def test_trailing_double_newline_last_segment_is_empty(self):
        """Response ending in \\n\\n: last segment is empty → no digits → nan."""
        response = "The level is 3.\n\n"
        assert math.isnan(extract_demand_level(response))

    # --- last integer rule ---

    def test_last_integer_in_conclusion_is_taken(self):
        """When multiple integers present, the LAST one determines the score."""
        response = "Level 2 is close.\n\nFinal verdict: score 4, not 2. Answer: 3"
        # digits in conclusion = ['4', '2', '3'], last = 3
        assert extract_demand_level(response) == 3.0

    def test_large_number_then_valid_score(self):
        """Large out-of-range number followed by valid score → valid score returned."""
        response = "Analysis.\n\nConsidering 2023 data, the score is 2"
        # digits = ['2023', '2'], last = 2, valid
        assert extract_demand_level(response) == 2.0

    # --- range validation ---

    def test_score_6_returns_nan(self):
        assert math.isnan(extract_demand_level("The level is 6"))

    def test_score_9_returns_nan(self):
        assert math.isnan(extract_demand_level("Answer: 9"))

    def test_only_large_number_returns_nan(self):
        assert math.isnan(extract_demand_level("The year 2023 is relevant"))

    # --- failure paths ---

    def test_empty_response_returns_nan(self):
        assert math.isnan(extract_demand_level(""))

    def test_no_digits_returns_nan(self):
        assert math.isnan(extract_demand_level("The level cannot be determined."))

    def test_whitespace_only_returns_nan(self):
        assert math.isnan(extract_demand_level("   \n\n   "))

    # --- section-number rejection ---

    def test_rejects_section_number_at_line_start(self):
        """'4. Conclusion: ...' — the only digit is a section header → nan."""
        response = "Analysis.\n\n4. Conclusion: I cannot determine the score."
        assert math.isnan(extract_demand_level(response))

    def test_rejects_numbered_summary_section(self):
        response = "Reasoning.\n\n3. Summary: No definitive answer available."
        assert math.isnan(extract_demand_level(response))

    def test_does_not_reject_score_in_sentence_body(self):
        """Score digit appearing mid-sentence is NOT at line start → not rejected."""
        response = "Analysis.\n\nThus the level is 3."
        assert extract_demand_level(response) == 3.0

    def test_does_not_reject_when_multiple_digits_present(self):
        """Section rejection only applies when len(digits) == 1."""
        response = "Analysis.\n\n4. Summary: The final score is 3"
        # digits = ['4', '3'], len = 2 → section check skipped → last = 3
        assert extract_demand_level(response) == 3.0

    def test_valid_score_with_period_not_at_line_start(self):
        """'The level is 3. It reflects...' — 3. is not at line start → accepted."""
        response = "Analysis.\n\nThe level is 3. It reflects moderate demand."
        # digits = ['3'], ^3\. would match "3." only if at start of line
        # "The level is 3." — '3' is NOT at start → returned
        assert extract_demand_level(response) == 3.0


# ---------------------------------------------------------------------------
# extract_ug_score
# ---------------------------------------------------------------------------

class TestExtractUGScore:

    # --- "open" variants ---

    def test_open_lowercase(self):
        raw, score = extract_ug_score("open")
        assert raw == "open"
        assert score == 100.0

    def test_open_uppercase(self):
        _, score = extract_ug_score("OPEN")
        assert score == 100.0

    def test_open_mixed_case(self):
        _, score = extract_ug_score("Open")
        assert score == 100.0

    def test_open_with_surrounding_whitespace(self):
        _, score = extract_ug_score("  open  ")
        assert score == 100.0

    def test_open_ignores_lines_after_first(self):
        raw, score = extract_ug_score("open\nextra line")
        assert score == 100.0

    # --- MCQ integer paths ---

    def test_four_choices(self):
        _, score = extract_ug_score("4")
        assert abs(score - 75.0) < 1e-6  # (1 - 1/4) * 100

    def test_two_choices_yes_no(self):
        _, score = extract_ug_score("2")
        assert abs(score - 50.0) < 1e-6  # (1 - 1/2) * 100

    def test_seven_choices_days_of_week(self):
        _, score = extract_ug_score("7")
        expected = (1 - 1 / 7) * 100
        assert abs(score - expected) < 1e-5

    def test_one_choice_gives_zero(self):
        """n=1 → (1-1/1)*100 = 0.0. Degenerate case; formula still applies."""
        _, score = extract_ug_score("1")
        assert score == 0.0

    def test_large_n_approaches_100(self):
        _, score = extract_ug_score("1000")
        assert abs(score - 99.9) < 0.01

    def test_raw_output_preserved(self):
        raw, _ = extract_ug_score("4")
        assert raw == "4"

    def test_only_first_line_used_for_integer(self):
        raw, score = extract_ug_score("4\nignored")
        assert raw == "4"
        assert abs(score - 75.0) < 1e-6

    # --- failure paths ---

    def test_zero_returns_nan(self):
        _, score = extract_ug_score("0")
        assert math.isnan(score)

    def test_negative_integer_returns_nan(self):
        # "-1" → int parses to -1, n < 1 → nan
        _, score = extract_ug_score("-1")
        assert math.isnan(score)

    def test_float_string_returns_nan(self):
        """'3.5' raises ValueError in int() → nan."""
        _, score = extract_ug_score("3.5")
        assert math.isnan(score)

    def test_alphabetic_returns_nan(self):
        _, score = extract_ug_score("abc")
        assert math.isnan(score)

    def test_empty_string_returns_nan(self):
        _, score = extract_ug_score("")
        assert math.isnan(score)

    def test_whitespace_only_returns_nan(self):
        _, score = extract_ug_score("   ")
        assert math.isnan(score)

    def test_returns_tuple(self):
        result = extract_ug_score("4")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nan_raw_output_still_returned(self):
        """On parse failure raw_output is still the stripped first line."""
        raw, score = extract_ug_score("bad_value")
        assert raw == "bad_value"
        assert math.isnan(score)
