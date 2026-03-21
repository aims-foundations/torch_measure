# Copyright (c) 2026 AIMS Foundation. MIT License.

"""PRISM (Preference, Reflection, and Ideological Signal Measurement) dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PRISM_URL = "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment"

_PRISM_CITATION = (
    "@misc{kirk2024prism,\n"
    "  title={The PRISM Alignment Dataset: What Participatory, Representative "
    "and Individualised Human Feedback Reveals About the Subjective and "
    "Multicultural Alignment of Large Language Models},\n"
    "  author={Kirk, Hannah Rose and Whitefield, Alexander and R{\\\"o}ttger, Paul and "
    "Bean, Andrew and Margatina, Katerina and Ciro, Juan and Mosquera, Rafael and "
    "Bartolo, Max and Williams, Adina and He, He and Vidgen, Bertie and Hale, Scott A.},\n"
    "  year={2024},\n"
    "  eprint={2404.16019},\n"
    "  archivePrefix={arXiv},\n"
    "  url={https://arxiv.org/abs/2404.16019}\n"
    "}"
)


def _register_prism_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for PRISM alignment response matrices.

    PRISM captures diverse human preferences for LLM outputs across
    demographics.  1,500+ participants from 75 countries each rate multiple
    LLM responses on a 1-100 cardinal scale.  The dataset is uniquely
    suited for studying preference heterogeneity and personalization.

    Each response matrix has:

    - **Rows (subjects)**: Human participants (raters).
    - **Columns (items)**: Utterances (model responses to user prompts).
    - **Values**: Continuous [0, 1] scores (from original 1-100 scale) or
      binary chosen/not-chosen indicators.

    Source data: ``HannahRoseKirk/prism-alignment`` on HuggingFace
    (``utterances`` configuration).

    Matrices produced:

    - ``prism/scores``: participants x utterances, continuous [0, 1]
      (normalized cardinal ratings).
    - ``prism/chosen``: participants x utterances, binary 1 = chosen /
      0 = not chosen.
    - ``prism/crossed_scores``: utterances x participants, transposed
      continuous matrix for G-theory facet analysis.
    - ``prism/crossed_chosen``: utterances x participants, transposed
      binary matrix for G-theory facet analysis.
    """
    datasets: dict[str, DatasetInfo] = {}

    # Placeholders for dimensions; updated after running the data collection
    # and migration scripts.  The utterances config has ~1,500 participants
    # and ~68,371 rated utterances, but the actual response matrix will be
    # sparse (each participant only rates a subset of utterances).
    _N_PARTICIPANTS = 1500
    _N_UTTERANCES = 68371

    # --- Continuous scores (1-100 normalized to 0-1) ---

    datasets["prism/scores"] = DatasetInfo(
        name="prism/scores",
        family="prism",
        description=(
            "PRISM -- human cardinal ratings of LLM responses, "
            f"normalized to [0,1] ({_N_PARTICIPANTS} participants x "
            f"{_N_UTTERANCES} utterances)"
        ),
        response_type="continuous",
        n_subjects=_N_PARTICIPANTS,
        n_items=_N_UTTERANCES,
        subject_entity="human",
        item_entity="utterance",
        filename="prism/scores.pt",
        citation=_PRISM_CITATION,
        url=_PRISM_URL,
        license="CC-BY-NC-4.0",
        tags=[
            "preference",
            "human-rated",
            "cardinal",
            "multicultural",
            "personalization",
            "heterogeneity",
        ],
    )

    # --- Binary chosen indicator ---

    datasets["prism/chosen"] = DatasetInfo(
        name="prism/chosen",
        family="prism",
        description=(
            "PRISM -- binary chosen indicator for LLM responses "
            f"({_N_PARTICIPANTS} participants x {_N_UTTERANCES} utterances)"
        ),
        response_type="binary",
        n_subjects=_N_PARTICIPANTS,
        n_items=_N_UTTERANCES,
        subject_entity="human",
        item_entity="utterance",
        filename="prism/chosen.pt",
        citation=_PRISM_CITATION,
        url=_PRISM_URL,
        license="CC-BY-NC-4.0",
        tags=[
            "preference",
            "human-rated",
            "binary",
            "multicultural",
            "personalization",
            "heterogeneity",
        ],
    )

    # --- Crossed: utterances x participants (continuous) ---

    datasets["prism/crossed_scores"] = DatasetInfo(
        name="prism/crossed_scores",
        family="prism",
        description=(
            "PRISM -- utterances x participants continuous scores, "
            "transposed for G-theory facet analysis"
        ),
        response_type="continuous",
        n_subjects=_N_UTTERANCES,
        n_items=_N_PARTICIPANTS,
        subject_entity="utterance",
        item_entity="human",
        filename="prism/crossed_scores.pt",
        citation=_PRISM_CITATION,
        url=_PRISM_URL,
        license="CC-BY-NC-4.0",
        tags=[
            "preference",
            "human-rated",
            "cardinal",
            "g-theory",
            "crossed",
            "multicultural",
        ],
    )

    # --- Crossed: utterances x participants (binary) ---

    datasets["prism/crossed_chosen"] = DatasetInfo(
        name="prism/crossed_chosen",
        family="prism",
        description=(
            "PRISM -- utterances x participants binary chosen, "
            "transposed for G-theory facet analysis"
        ),
        response_type="binary",
        n_subjects=_N_UTTERANCES,
        n_items=_N_PARTICIPANTS,
        subject_entity="utterance",
        item_entity="human",
        filename="prism/crossed_chosen.pt",
        citation=_PRISM_CITATION,
        url=_PRISM_URL,
        license="CC-BY-NC-4.0",
        tags=[
            "preference",
            "human-rated",
            "binary",
            "g-theory",
            "crossed",
            "multicultural",
        ],
    )

    return datasets
