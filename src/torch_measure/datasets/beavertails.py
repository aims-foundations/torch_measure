# Copyright (c) 2026 AIMS Foundation. MIT License.

"""BeaverTails (PKU-Alignment) safety classification dataset definitions.

BeaverTails is a safety-focused dataset containing 330K+ QA pairs with
human-annotated safety labels across 14 harm categories.  Each QA pair
has a binary overall ``is_safe`` label and per-category binary labels
indicating which (if any) of the 14 harm categories apply.

Each response matrix follows the standard torch_measure convention:

- **Rows (subjects)**: Harm categories (binary safety classifiers).
- **Columns (items)**: Individual QA pairs from the 330k_test split.
- **Values**: Binary {0, 1} where 1 = unsafe (flagged for that category),
  0 = safe.  For the ``overall`` dataset, 1 = unsafe (``is_safe=False``).

Data files live on HuggingFace Hub at ``sangttruong/torch-measure-data``
under the ``beavertails/`` prefix (e.g. ``beavertails/overall.pt``).

Source data: ``PKU-Alignment/BeaverTails`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_BEAVERTAILS_URL = "https://huggingface.co/datasets/PKU-Alignment/BeaverTails"

_BEAVERTAILS_CITATION = (
    "@article{ji2023beavertails,\n"
    "  title={BeaverTails: Towards Improved Safety Alignment of LLM via a "
    "Human-Preference Dataset},\n"
    "  author={Ji, Jiaming and Liu, Mickel and Dai, Juntao and Pan, Xuehai and "
    "Zhang, Chi and Bian, Ce and Chen, Boyuan and Sun, Ruiyang and Wang, Yizhou "
    "and Yang, Yaodong},\n"
    "  journal={Advances in Neural Information Processing Systems},\n"
    "  volume={36},\n"
    "  year={2023}\n"
    "}"
)

# The 14 harm categories as they appear in the dataset.
HARM_CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]

# Registry-friendly short names for per-category splits.
CATEGORY_NAME_MAP = {
    "animal_abuse": "animal_abuse",
    "child_abuse": "child_abuse",
    "controversial_topics,politics": "controversial_topics",
    "discrimination,stereotype,injustice": "discrimination",
    "drug_abuse,weapons,banned_substance": "drug_abuse",
    "financial_crime,property_crime,theft": "financial_crime",
    "hate_speech,offensive_language": "hate_speech",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "non_violent_unethical",
    "privacy_violation": "privacy_violation",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexually_explicit",
    "terrorism,organized_crime": "terrorism",
    "violence,aiding_and_abetting,incitement": "violence",
}

# Number of items in the 330k_test split.
_N_ITEMS = 33432

# Number of subjects: 14 harm categories + 1 overall = 15
_N_SUBJECTS_ALL = 15


def _register_beavertails_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for BeaverTails safety classification matrices.

    BeaverTails (PKU-Alignment) provides 330K+ QA pairs annotated with binary
    safety labels across 14 harm categories.  We use the ``330k_test`` split
    (33,432 QA pairs) and build response matrices where:

    - **Subjects**: Harm categories (binary classifiers).
    - **Items**: QA pairs.
    - **Values**: Binary {0, 1} (1 = unsafe / flagged).

    Datasets:
        - ``beavertails/all``: All 15 classifiers (14 categories + overall)
          x 33,432 QA pairs.
        - ``beavertails/overall``: Overall ``is_safe`` label only (1 x 33,432).
        - ``beavertails/<category>``: Per-category binary label (1 x 33,432).

    Source data: ``PKU-Alignment/BeaverTails`` (CC-BY-NC-4.0).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- All categories + overall combined ---
    datasets["beavertails/all"] = DatasetInfo(
        name="beavertails/all",
        family="beavertails",
        description=(
            "BeaverTails — all 14 harm categories + overall safety label "
            f"({_N_SUBJECTS_ALL} classifiers x {_N_ITEMS:,} QA pairs)"
        ),
        response_type="binary",
        n_subjects=_N_SUBJECTS_ALL,
        n_items=_N_ITEMS,
        subject_entity="harm_category",
        item_entity="qa_pair",
        filename="beavertails/all.pt",
        citation=_BEAVERTAILS_CITATION,
        url=_BEAVERTAILS_URL,
        license="CC-BY-NC-4.0",
        tags=["safety", "classification", "harm-categories", "binary"],
    )

    # --- Overall is_safe label ---
    datasets["beavertails/overall"] = DatasetInfo(
        name="beavertails/overall",
        family="beavertails",
        description=(
            f"BeaverTails — overall binary safety label (1 x {_N_ITEMS:,} QA pairs)"
        ),
        response_type="binary",
        n_subjects=1,
        n_items=_N_ITEMS,
        subject_entity="harm_category",
        item_entity="qa_pair",
        filename="beavertails/overall.pt",
        citation=_BEAVERTAILS_CITATION,
        url=_BEAVERTAILS_URL,
        license="CC-BY-NC-4.0",
        tags=["safety", "classification", "binary", "overall"],
    )

    # --- Per-category datasets ---
    for category in HARM_CATEGORIES:
        short = CATEGORY_NAME_MAP[category]
        name = f"beavertails/{short}"
        datasets[name] = DatasetInfo(
            name=name,
            family="beavertails",
            description=(
                f"BeaverTails — {category} binary safety label "
                f"(1 x {_N_ITEMS:,} QA pairs)"
            ),
            response_type="binary",
            n_subjects=1,
            n_items=_N_ITEMS,
            subject_entity="harm_category",
            item_entity="qa_pair",
            filename=f"beavertails/{short}.pt",
            citation=_BEAVERTAILS_CITATION,
            url=_BEAVERTAILS_URL,
            license="CC-BY-NC-4.0",
            tags=["safety", "classification", "binary", short],
        )

    return datasets
