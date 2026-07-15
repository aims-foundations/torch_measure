# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for _prompts.py — zero API calls, zero file I/O.

Verifies that get_full_instruction() reproduces the verbatim ADeLe paper
prompt template with exact spacing, blank lines, and wording.
"""

import pytest

from torch_measure.annotation._prompts import get_full_instruction, get_ug_instruction


class TestGetFullInstruction:

    # --- structure ---

    def test_starts_with_rubric_header(self):
        result = get_full_instruction("MyDim", "Level 0: None.", "ITEM")
        assert result.startswith(
            "The following rubric describes six distinct levels of *MyDim*"
            " required by different tasks:\n"
        )

    def test_ends_with_cot_prompt(self):
        result = get_full_instruction("X", "content", "item")
        assert result.endswith(
            "CHAIN-OF-THOUGHTS REASONING STEPS to score the level of *X*"
            " demanded by the given TASK INSTANCE above:\n"
        )

    def test_task_instance_label_present(self):
        result = get_full_instruction("X", "content", "My item text")
        assert "TASK INSTANCE: My item text" in result

    def test_instruction_label_present(self):
        result = get_full_instruction("X", "content", "item")
        assert "INSTRUCTION: Score the level of *X*" in result

    # --- blank-line spacing ---

    def test_exactly_one_blank_line_before_task_instance(self):
        """rubric_content (no trailing \\n) + \\n + \\n → exactly one blank line."""
        content = "Level 0: None."  # no trailing newline
        result = get_full_instruction("D", content, "ITEM")
        idx = result.index("TASK INSTANCE:")
        assert result[idx - 2 : idx] == "\n\n", (
            f"Expected '\\n\\n' before TASK INSTANCE, "
            f"got {repr(result[idx - 4 : idx + 4])}"
        )

    def test_exactly_one_blank_line_before_instruction(self):
        content = "Level 0: None."
        result = get_full_instruction("D", content, "ITEM")
        idx = result.index("INSTRUCTION:")
        assert result[idx - 2 : idx] == "\n\n"

    def test_exactly_one_blank_line_before_cot(self):
        content = "Level 0: None."
        result = get_full_instruction("D", content, "ITEM")
        # rindex: the phrase appears twice — once inside the instruction sentence
        # and once as the section header at the end. We want the header.
        idx = result.rindex("CHAIN-OF-THOUGHTS REASONING STEPS")
        assert result[idx - 2 : idx] == "\n\n"

    def test_no_extra_blank_line_when_content_has_no_trailing_newline(self):
        """Content without trailing \\n must produce exactly one blank line."""
        content = "Level 0: None.\nLevel 5: Very high."
        result = get_full_instruction("D", content, "ITEM")
        content_end = result.index(content) + len(content)
        # Must be \n\nT (TASK), not \n\n\nT
        assert result[content_end : content_end + 3] == "\n\nT", (
            f"Got {repr(result[content_end : content_end + 5])}"
        )

    # --- verbatim wording ---

    def test_conclusion_statement_verbatim(self):
        result = get_full_instruction("X", "content", "item")
        expected = (
            '"Thus, the level of *X* demanded by the given TASK INSTANCE is: SCORE",'
            " where SCORE is an integer score you have determined."
        )
        assert expected in result

    def test_instruction_text_verbatim(self):
        result = get_full_instruction("X", "content", "item")
        assert (
            "Score the level of *X* demanded by the given TASK INSTANCE "
            "using a discrete value from 0 to 5. "
            "Use CHAIN-OF-THOUGHTS REASONING to reason step by step before assigning the score. "
            "After the CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the "
            'statement: "Thus, the level of *X* demanded by the given TASK INSTANCE is: SCORE"'
        ) in result

    def test_dimension_appears_in_all_four_positions(self):
        """dimension appears in: header, INSTRUCTION (×2), CHAIN-OF-THOUGHTS."""
        result = get_full_instruction("TargetDim", "content", "item")
        assert result.count("TargetDim") == 4

    def test_item_text_appears_verbatim(self):
        item = "What is 2 + 2?"
        result = get_full_instruction("X", "content", item)
        assert f"TASK INSTANCE: {item}" in result

    def test_rubric_content_appears_verbatim(self):
        content = "Level 0: None.\nLevel 1: Low.\nLevel 5: Very high."
        result = get_full_instruction("X", content, "item")
        assert content in result

    # --- no extra instructions ---

    def test_no_gemini_specific_instructions(self):
        """Prompt must not contain Gemini-specific tokens or instructions."""
        result = get_full_instruction("X", "content", "item")
        forbidden = ["gemini", "bard", "google", "think step by step", "let's think"]
        for phrase in forbidden:
            assert phrase.lower() not in result.lower(), (
                f"Found forbidden phrase '{phrase}' in prompt"
            )

    def test_prompt_has_no_system_instruction_marker(self):
        result = get_full_instruction("X", "content", "item")
        assert "system:" not in result.lower()
        assert "<system>" not in result.lower()


class TestGetUGInstruction:

    def test_exact_structure(self):
        result = get_ug_instruction("Question", "Answer", "Rubric")
        assert result == "Question\n\nReference answer: Answer\n\nRubric"

    def test_two_blank_lines_separate_each_section(self):
        result = get_ug_instruction("Q", "A", "R")
        parts = result.split("\n\n")
        assert len(parts) == 3
        assert parts[0] == "Q"
        assert parts[1] == "Reference answer: A"
        assert parts[2] == "R"

    def test_ug_rubric_appended_without_modification(self):
        rubric = "You are tasked...\nOutput:"
        result = get_ug_instruction("Q", "A", rubric)
        assert result.endswith(rubric)

    def test_reference_answer_label_verbatim(self):
        result = get_ug_instruction("Q", "Paris", "R")
        assert "Reference answer: Paris" in result
