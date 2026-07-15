from __future__ import annotations

import hashlib
from pathlib import Path

from ._types import DEMAND_DIMENSIONS, Rubric


class RubricsCatalog:
    """Loads rubric .txt files from the bundled rubrics/ directory."""

    def __init__(self, rubrics_dir: Path | None = None) -> None:
        if rubrics_dir is None:
            rubrics_dir = Path(__file__).parent / "rubrics"
        self._rubrics: dict[str, Rubric] = {}
        self._ug_content: str = ""
        self._ug_hash: str = ""
        self._load(rubrics_dir)
        missing = [a for a in DEMAND_DIMENSIONS if a not in self._rubrics]
        if missing:
            raise RuntimeError(f"Missing rubric files: {missing}")

    def _load(self, rubrics_dir: Path) -> None:
        for path in rubrics_dir.glob("*.txt"):
            acronym = path.stem
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(keepends=True)

            if lines and lines[0].startswith("#"):
                dimension_name = lines[0].lstrip("#").strip()
                content = "".join(lines[1:]).strip("\n")
            else:
                dimension_name = acronym
                content = text

            rubric_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            if acronym == "UG_choice_num":
                self._ug_content = content
                self._ug_hash = rubric_hash
            else:
                self._rubrics[acronym] = Rubric(
                    acronym=acronym,
                    dimension_name=dimension_name,
                    content=content,
                    rubric_hash=rubric_hash,
                )

    def get(self, acronym: str) -> Rubric:
        return self._rubrics[acronym]

    @property
    def ug_content(self) -> str:
        return self._ug_content

    @property
    def ug_hash(self) -> str:
        return self._ug_hash

    def all_demand_rubrics(self) -> list[Rubric]:
        """Return all 18 demand rubrics in canonical DIMENSION_ORDER."""
        return [self._rubrics[a] for a in DEMAND_DIMENSIONS]
