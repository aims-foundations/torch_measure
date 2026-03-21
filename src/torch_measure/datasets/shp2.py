# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Stanford Human Preferences v2 (SHP-2) dataset definitions.

SHP-2 contains 4.3M naturally-occurring pairwise preferences across 124
subject areas from Reddit and StackExchange.  Preferences are inferred from
upvote differentials rather than explicit human annotation.

Two response matrices are provided:

- **domain_stats**: Per-domain summary statistics.
  Rows = domains (subreddits/sites), columns = normalized metrics.
- **sampled_pairs**: 100K reservoir-sampled preference pairs.
  Rows = response positions (A, B), columns = pair indices.
  Binary encoding: 1 if that response position was preferred, 0 otherwise.

Source data: ``stanfordnlp/SHP-2`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_SHP2_URL = "https://huggingface.co/datasets/stanfordnlp/SHP-2"

_SHP2_CITATION = (
    "@inproceedings{ethayarajh2022understanding,\n"
    "  title={Understanding Dataset Difficulty with $\\mathcal{V}$-Usable "
    "Information},\n"
    "  author={Ethayarajh, Kawin and Choi, Yejin and Swayamdipta, Swabha},\n"
    "  booktitle={International Conference on Machine Learning},\n"
    "  year={2022},\n"
    "  url={https://huggingface.co/datasets/stanfordnlp/SHP-2}\n"
    "}"
)


def _register_shp2_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for SHP-2 response matrices.

    Stanford Human Preferences v2 (SHP-2) provides 4.8M naturally-occurring
    pairwise preferences from Reddit and StackExchange, where preference is
    inferred from upvote differentials.

    Two matrices:

    - ``shp2/domain_stats``: Per-domain aggregated statistics as a response
      matrix (domains x 5 metrics).  Values are continuous [0, 1].
    - ``shp2/sampled_pairs``: 100K reservoir-sampled preference pairs
      (2 response positions x 100K pairs).  Binary preferred label.

    Source data: ``stanfordnlp/SHP-2`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    datasets["shp2/domain_stats"] = DatasetInfo(
        name="shp2/domain_stats",
        family="shp2",
        description=(
            "SHP-2 — per-domain preference statistics from 4.3M Reddit/SE pairs "
            "(124 domains x 5 metrics)"
        ),
        response_type="continuous",
        n_subjects=124,
        n_items=5,
        subject_entity="domain",
        item_entity="metric",
        filename="shp2/domain_stats.pt",
        citation=_SHP2_CITATION,
        url=_SHP2_URL,
        license="ODC-BY",
        tags=["preference", "pairwise", "reddit", "stackexchange", "human", "summary"],
    )

    datasets["shp2/sampled_pairs"] = DatasetInfo(
        name="shp2/sampled_pairs",
        family="shp2",
        description=(
            "SHP-2 — 100K sampled preference pairs, binary preferred label "
            "(2 response positions x 100,000 pairs)"
        ),
        response_type="binary",
        n_subjects=2,
        n_items=100000,
        subject_entity="response_position",
        item_entity="pair",
        filename="shp2/sampled_pairs.pt",
        citation=_SHP2_CITATION,
        url=_SHP2_URL,
        license="ODC-BY",
        n_comparisons=100000,
        tags=["preference", "pairwise", "reddit", "stackexchange", "human", "sampled"],
    )

    return datasets
