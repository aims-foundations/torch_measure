# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for RubricsCatalog — reads bundled rubric files, no API calls.

These tests verify that rubric files are loaded correctly, that the
dimension order matches the paper specification, and that validation
works correctly.
"""

import shutil

import pytest

from torch_measure.annotation._rubrics import RubricsCatalog
from torch_measure.annotation._types import DEMAND_DIMENSIONS, DIMENSION_ORDER


@pytest.fixture(scope="module")
def catalog():
    return RubricsCatalog()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestRubricsCatalogLoading:

    def test_loads_exactly_18_demand_rubrics(self, catalog):
        assert len(catalog.all_demand_rubrics()) == 18

    def test_all_demand_dimensions_present(self, catalog):
        for acronym in DEMAND_DIMENSIONS:
            rubric = catalog.get(acronym)
            assert rubric is not None
            assert rubric.acronym == acronym

    def test_ug_content_loaded(self, catalog):
        assert len(catalog.ug_content) > 0

    def test_ug_hash_nonempty(self, catalog):
        assert len(catalog.ug_hash) == 16
        assert all(c in "0123456789abcdef" for c in catalog.ug_hash)

    def test_ug_content_ends_with_output_prompt(self, catalog):
        """UG_choice_num.txt ends with 'Output:' — the model's response anchor."""
        assert catalog.ug_content.rstrip().endswith("Output:")

    def test_ug_content_contains_classification_instructions(self, catalog):
        assert "multiple-choice" in catalog.ug_content.lower()
        assert "open-ended" in catalog.ug_content.lower()

    def test_ug_not_in_demand_rubrics(self, catalog):
        acronyms = [r.acronym for r in catalog.all_demand_rubrics()]
        assert "UG" not in acronyms
        assert "UG_choice_num" not in acronyms


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------

class TestDimensionOrder:

    def test_all_demand_rubrics_in_canonical_order(self, catalog):
        rubrics = catalog.all_demand_rubrics()
        acronyms = [r.acronym for r in rubrics]
        assert acronyms == list(DEMAND_DIMENSIONS)

    def test_dimension_order_has_19_elements(self):
        assert len(DIMENSION_ORDER) == 19

    def test_demand_dimensions_is_first_18(self):
        assert DEMAND_DIMENSIONS == DIMENSION_ORDER[:18]

    def test_ug_is_last_in_dimension_order(self):
        assert DIMENSION_ORDER[-1] == "UG"


# ---------------------------------------------------------------------------
# Content format
# ---------------------------------------------------------------------------

class TestRubricContent:

    def test_content_has_no_trailing_newline(self, catalog):
        """strip('\\n') must have been applied; trailing \\n causes extra blank lines in prompts."""
        for rubric in catalog.all_demand_rubrics():
            assert not rubric.content.endswith("\n"), (
                f"{rubric.acronym}.content ends with \\n — "
                "this produces extra blank lines in the generated prompt"
            )

    def test_content_has_no_leading_newline(self, catalog):
        for rubric in catalog.all_demand_rubrics():
            assert not rubric.content.startswith("\n"), (
                f"{rubric.acronym}.content starts with \\n"
            )

    def test_content_is_nonempty(self, catalog):
        for rubric in catalog.all_demand_rubrics():
            assert len(rubric.content) > 100, (
                f"{rubric.acronym} content suspiciously short: {len(rubric.content)} chars"
            )

    def test_content_contains_level_definitions(self, catalog):
        """Every rubric must contain Level 0 through Level 5."""
        for rubric in catalog.all_demand_rubrics():
            for level in range(6):
                assert f"Level {level}:" in rubric.content, (
                    f"{rubric.acronym} missing 'Level {level}:'"
                )

    def test_dimension_name_is_nonempty(self, catalog):
        for rubric in catalog.all_demand_rubrics():
            assert len(rubric.dimension_name) > 0

    def test_rubric_hash_is_16_hex_chars(self, catalog):
        for rubric in catalog.all_demand_rubrics():
            assert len(rubric.rubric_hash) == 16
            assert all(c in "0123456789abcdef" for c in rubric.rubric_hash), (
                f"{rubric.acronym} has non-hex rubric_hash: {rubric.rubric_hash}"
            )

    def test_all_hashes_distinct(self, catalog):
        """Each rubric file has distinct content → distinct hashes."""
        hashes = [r.rubric_hash for r in catalog.all_demand_rubrics()]
        assert len(hashes) == len(set(hashes)), (
            "Two rubrics have identical content hashes — check for duplicate files"
        )


# ---------------------------------------------------------------------------
# Paper-specific rubric format checks
# ---------------------------------------------------------------------------

class TestPaperSpecificFormat:

    def test_at_txt_starts_with_level_0_no_description(self, catalog):
        """AT.txt must use the delean-batch-manager version (no opening description ¶)."""
        at = catalog.get("AT")
        first_line = at.content.split("\n")[0]
        assert first_line.startswith("Level 0"), (
            f"AT.txt should start with 'Level 0:' (no description paragraph). "
            f"First line: {repr(first_line)}"
        )

    def test_cl_txt_starts_with_level_0_no_description(self, catalog):
        """CL.txt must use the delean-batch-manager version (no opening description ¶)."""
        cl = catalog.get("CL")
        first_line = cl.content.split("\n")[0]
        assert first_line.startswith("Level 0"), (
            f"CL.txt should start with 'Level 0:' (no description paragraph). "
            f"First line: {repr(first_line)}"
        )

    def test_as_txt_has_description_paragraph(self, catalog):
        """AS.txt should have a description paragraph before Level 0."""
        as_rubric = catalog.get("AS")
        first_line = as_rubric.content.split("\n")[0]
        assert not first_line.startswith("Level"), (
            "AS.txt should have a description paragraph as its first content. "
            "Only AT.txt and CL.txt use the no-description version."
        )

    def test_ug_content_not_stripped(self, catalog):
        """UG_choice_num.txt has no # header → content = full file text (not stripped)."""
        # The UG file starts with "You are tasked..." not with a level definition
        assert "You are tasked" in catalog.ug_content or "classif" in catalog.ug_content.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestRubricsCatalogValidation:

    def test_missing_rubric_raises_runtime_error(self, tmp_path):
        """If any demand rubric file is absent, __init__ must raise RuntimeError."""
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src/torch_measure/annotation/rubrics"
        if not src_dir.exists():
            pytest.skip("Cannot locate bundled rubrics directory for this test")

        dest_dir = tmp_path / "rubrics"
        shutil.copytree(src_dir, dest_dir)
        (dest_dir / "AS.txt").unlink()

        with pytest.raises(RuntimeError, match="Missing rubric files"):
            RubricsCatalog(dest_dir)

    def test_error_message_names_missing_rubric(self, tmp_path):
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src/torch_measure/annotation/rubrics"
        if not src_dir.exists():
            pytest.skip("Cannot locate bundled rubrics directory for this test")

        dest_dir = tmp_path / "rubrics"
        shutil.copytree(src_dir, dest_dir)
        (dest_dir / "QLl.txt").unlink()

        with pytest.raises(RuntimeError, match="QLl"):
            RubricsCatalog(dest_dir)

    def test_multiple_missing_rubrics_named_in_error(self, tmp_path):
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src/torch_measure/annotation/rubrics"
        if not src_dir.exists():
            pytest.skip("Cannot locate bundled rubrics directory for this test")

        dest_dir = tmp_path / "rubrics"
        shutil.copytree(src_dir, dest_dir)
        (dest_dir / "AS.txt").unlink()
        (dest_dir / "CL.txt").unlink()

        with pytest.raises(RuntimeError) as exc_info:
            RubricsCatalog(dest_dir)
        msg = str(exc_info.value)
        assert "AS" in msg
        assert "CL" in msg

    def test_custom_rubrics_dir_accepted(self, tmp_path):
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src/torch_measure/annotation/rubrics"
        if not src_dir.exists():
            pytest.skip("Cannot locate bundled rubrics directory for this test")

        dest_dir = tmp_path / "rubrics"
        shutil.copytree(src_dir, dest_dir)
        catalog = RubricsCatalog(dest_dir)
        assert len(catalog.all_demand_rubrics()) == 18
