# Copyright (c) 2026 AIMS Foundations. MIT License.

"""PRM800K (OpenAI) dataset definitions.

PRM800K is a process reward model training dataset containing ~800K
step-level human correctness labels for model-generated solutions to
problems from the MATH dataset.

Each response matrix has:

- **Rows (subjects)**: Individual solution attempts (model-generated
  solutions with human step-level labels).
- **Columns (items)**: Step positions within a solution (step 0, step 1, ...).
- **Values**: Step-level correctness ratings mapped to {0, 1, NaN}.
  Original labels: +1 (correct) -> 1, -1 (incorrect) -> 0,
  0 (neutral) -> NaN, None (unlabeled) -> NaN.  Steps beyond a
  solution's length are NaN.

Source data:
    - ``openai/prm800k`` on GitHub (phase 1 & phase 2 JSONL files).
    - MATH dataset problems with ground-truth solutions.

Two phases of data collection:
    - **Phase 1**: Labelers could write alternative steps; smaller scale.
    - **Phase 2**: Used active learning with PRM to select solutions; bulk of data.

Train/test split follows the nonstandard MATH split from the paper:
    - Train: 12,000 problems (7,500 MATH train + 4,500 MATH test).
    - Test: 500 held-out MATH test problems.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PRM800K_URL = "https://github.com/openai/prm800k"

_PRM800K_CITATION = (
    "@article{lightman2023lets,\n"
    "  title={Let's Verify Step by Step},\n"
    "  author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yuri "
    "and Edwards, Harri and Baker, Bowen and Lee, Teddy "
    "and Leike, Jan and Schulman, John and Sutskever, Ilya "
    "and Alignment, OpenAI},\n"
    "  journal={arXiv preprint arXiv:2305.20050},\n"
    "  year={2023},\n"
    "  url={https://arxiv.org/abs/2305.20050}\n"
    "}"
)


def _register_prm800k_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for PRM800K step-level response matrices.

    PRM800K provides step-level human labels for model-generated math
    solutions.  Each response matrix is:

    - **Rows**: Solution attempts (each a labeled solution to a MATH problem).
    - **Columns**: Step positions (step_0, step_1, ..., up to the max
      number of steps in that split).
    - **Values**: Binary {0, 1} from step correctness ratings, or NaN
      for neutral/unlabeled steps and padding beyond a solution's length.

    We provide train, test, and combined (all) matrices, as well as
    phase-specific splits.  Dimensions are determined by the data
    collection script and should be updated after building.
    """
    d: dict[str, DatasetInfo] = {}

    # ── Combined (all phases, all splits) ────────────────────────────
    d["all"] = DatasetInfo(
        name="all",
        family="prm800k",
        description=(
            "PRM800K all phases & splits — step-level correctness labels "
            "for math solutions (101599 solutions x 107 steps)"
        ),
        response_type="binary",
        n_subjects=101599,
        n_items=107,
        subject_entity="solution",
        item_entity="step",
        filename="all.pt",
        citation=_PRM800K_CITATION,
        url=_PRM800K_URL,
        license="MIT",
        tags=["process-reward", "math", "step-level", "binary"],
    )

    # ── Train split (phase 1 + phase 2 train) ───────────────────────
    d["train"] = DatasetInfo(
        name="train",
        family="prm800k",
        description=(
            "PRM800K train split — step-level correctness labels "
            "(98731 solutions x 107 steps)"
        ),
        response_type="binary",
        n_subjects=98731,
        n_items=107,
        subject_entity="solution",
        item_entity="step",
        filename="train.pt",
        citation=_PRM800K_CITATION,
        url=_PRM800K_URL,
        license="MIT",
        tags=["process-reward", "math", "step-level", "train"],
    )

    # ── Test split (phase 1 + phase 2 test) ─────────────────────────
    d["test"] = DatasetInfo(
        name="test",
        family="prm800k",
        description=(
            "PRM800K test split — step-level correctness labels "
            "(2868 solutions x 53 steps)"
        ),
        response_type="binary",
        n_subjects=2868,
        n_items=53,
        subject_entity="solution",
        item_entity="step",
        filename="test.pt",
        citation=_PRM800K_CITATION,
        url=_PRM800K_URL,
        license="MIT",
        tags=["process-reward", "math", "step-level", "test"],
    )

    # ── Phase 2 only (bulk of the data) ─────────────────────────────
    d["phase2"] = DatasetInfo(
        name="phase2",
        family="prm800k",
        description=(
            "PRM800K phase 2 only — active-learning-selected solutions "
            "(100544 solutions x 107 steps)"
        ),
        response_type="binary",
        n_subjects=100544,
        n_items=107,
        subject_entity="solution",
        item_entity="step",
        filename="phase2.pt",
        citation=_PRM800K_CITATION,
        url=_PRM800K_URL,
        license="MIT",
        tags=["process-reward", "math", "step-level", "phase2"],
    )

    return d
