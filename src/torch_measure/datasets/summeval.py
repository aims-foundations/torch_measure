# Copyright (c) 2026 AIMS Foundations. MIT License.

"""SummEval dataset definitions.

SummEval (Fabbri et al., 2021) evaluates text summarization quality across
100 CNN/DailyMail source documents, each summarized by 16 models.  Every
(model, document) summary is annotated by 3 experts and 5 crowd workers on
4 quality dimensions: coherence, consistency, fluency, and relevance
(1-5 Likert scale).

Each 2D response matrix has:

- **Rows (subjects)**: Summarization models (16 models).
- **Columns (items)**: Source documents (100 documents).
- **Values**: Continuous mean ratings averaged across annotators (experts or
  crowd workers), on a 1-5 scale.

The ``summeval/expert_3d`` dataset provides a 3D tensor
(16 models x 100 documents x 3 experts) with per-expert mean-of-4-dimension
scores, suitable for Generalizability Theory (G-theory) studies.

Source data: ``mteb/summeval`` on HuggingFace Hub.
Original data: Yale-LILY/SummEval on GitHub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_SUMMEVAL_URL = "https://github.com/Yale-LILY/SummEval"

_SUMMEVAL_CITATION = (
    "@article{fabbri2021summeval,\n"
    "  title={SummEval: Re-evaluating Summarization Evaluation},\n"
    "  author={Fabbri, Alexander R and Kry{\\'s}ci{\\'n}ski, Wojciech and "
    "McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},\n"
    "  journal={Transactions of the Association for Computational Linguistics},\n"
    "  volume={9},\n"
    "  pages={568--600},\n"
    "  year={2021},\n"
    "  publisher={MIT Press},\n"
    "  url={https://arxiv.org/abs/2007.12626}\n"
    "}"
)

_N_MODELS = 16
_N_DOCS = 100
_N_EXPERTS = 3
_N_TURKERS = 5

_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def _register_summeval_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for SummEval response matrices.

    Registers 2D response matrices for expert and crowd annotations on each
    quality dimension plus overall (mean of 4 dimensions), and a 3D expert
    tensor for G-theory analyses.  Total: 11 datasets.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Expert 2D response matrices (16 models x 100 docs) ---

    for dim in _DIMENSIONS:
        name = f"expert_{dim}"
        datasets[name] = DatasetInfo(
            name=name,
            family="summeval",
            description=(f"SummEval expert {dim} ratings, mean across 3 experts ({_N_MODELS} models x {_N_DOCS} docs)"),
            response_type="continuous",
            n_subjects=_N_MODELS,
            n_items=_N_DOCS,
            subject_entity="model",
            item_entity="document",
            filename=f"expert_{dim}.pt",
            citation=_SUMMEVAL_CITATION,
            url=_SUMMEVAL_URL,
            license="MIT",
            tags=["nlp", "summarization", "expert-rated", dim, "likert"],
        )

    # Expert overall (mean of 4 dimensions)
    datasets["expert_overall"] = DatasetInfo(
        name="expert_overall",
        family="summeval",
        description=(
            "SummEval expert overall rating (mean of 4 dimensions), "
            f"mean across 3 experts ({_N_MODELS} models x {_N_DOCS} docs)"
        ),
        response_type="continuous",
        n_subjects=_N_MODELS,
        n_items=_N_DOCS,
        subject_entity="model",
        item_entity="document",
        filename="expert_overall.pt",
        citation=_SUMMEVAL_CITATION,
        url=_SUMMEVAL_URL,
        license="MIT",
        tags=["nlp", "summarization", "expert-rated", "overall", "likert"],
    )

    # --- Crowd 2D response matrices (16 models x 100 docs) ---

    for dim in _DIMENSIONS:
        name = f"crowd_{dim}"
        datasets[name] = DatasetInfo(
            name=name,
            family="summeval",
            description=(f"SummEval crowd {dim} ratings, mean across 5 turkers ({_N_MODELS} models x {_N_DOCS} docs)"),
            response_type="continuous",
            n_subjects=_N_MODELS,
            n_items=_N_DOCS,
            subject_entity="model",
            item_entity="document",
            filename=f"crowd_{dim}.pt",
            citation=_SUMMEVAL_CITATION,
            url=_SUMMEVAL_URL,
            license="MIT",
            tags=["nlp", "summarization", "crowd-rated", dim, "likert"],
        )

    # Crowd overall (mean of 4 dimensions)
    datasets["crowd_overall"] = DatasetInfo(
        name="crowd_overall",
        family="summeval",
        description=(
            "SummEval crowd overall rating (mean of 4 dimensions), "
            f"mean across 5 turkers ({_N_MODELS} models x {_N_DOCS} docs)"
        ),
        response_type="continuous",
        n_subjects=_N_MODELS,
        n_items=_N_DOCS,
        subject_entity="model",
        item_entity="document",
        filename="crowd_overall.pt",
        citation=_SUMMEVAL_CITATION,
        url=_SUMMEVAL_URL,
        license="MIT",
        tags=["nlp", "summarization", "crowd-rated", "overall", "likert"],
    )

    # --- 3D expert tensor (16 models x 100 docs x 3 experts) ---

    datasets["expert_3d"] = DatasetInfo(
        name="expert_3d",
        family="summeval",
        description=(
            "SummEval 3D expert tensor, per-expert mean-of-4-dimensions "
            f"({_N_MODELS} models x {_N_DOCS} docs x {_N_EXPERTS} experts)"
        ),
        response_type="continuous",
        n_subjects=_N_MODELS,
        n_items=_N_DOCS,
        subject_entity="model",
        item_entity="document",
        filename="expert_3d.pt",
        citation=_SUMMEVAL_CITATION,
        url=_SUMMEVAL_URL,
        license="MIT",
        tags=["nlp", "summarization", "expert-rated", "3d-tensor", "fully-crossed", "g-theory"],
    )

    return datasets
