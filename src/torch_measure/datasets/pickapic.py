# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Pick-a-Pic text-to-image human preference dataset definitions.

Pick-a-Pic is a large-scale dataset of human preferences over text-to-image
model outputs, collected via the pickapic.io web application.  Users submit
text prompts, receive two generated images from different models, and indicate
which image they prefer (or declare a tie).

The dataset includes comparisons across multiple text-to-image models:
Stable Diffusion 2.1, Dreamlike Photoreal 2.0, Stable Diffusion XL variants,
and others (14 unique models in v2).

Each pairwise comparison payload has:

- **subject_ids**: Sorted unique text-to-image model names.
- **Comparisons**: One per user preference judgment; outcome encodes which
  model's image was preferred (1.0 = model_a wins, 0.0 = model_b wins,
  0.5 = tie).
- **item_ids**: Unique prompt identifiers (ranking_id).
- **item_contents**: The text prompt (caption) for each comparison.
- **comparison_metadata**: Per-comparison metadata (user_id, model names, etc.).

Source data: ``yuvalkirstain/pickapic_v2`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PICKAPIC_URL = "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2"

_PICKAPIC_CITATION = (
    "@article{kirstain2023pickapic,\n"
    "  title={Pick-a-Pic: An Open Dataset of User Preferences for "
    "Text-to-Image Generation},\n"
    "  author={Kirstain, Yuval and Polyak, Adam and Singer, Uriel and "
    "Matiana, Shahbuland and Penna, Joe and Levy, Omer},\n"
    "  journal={Advances in Neural Information Processing Systems},\n"
    "  volume={36},\n"
    "  year={2023}\n"
    "}"
)


def _register_pickapic_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Pick-a-Pic pairwise preference data.

    Pick-a-Pic collects human preference judgments on text-to-image model
    outputs.  Users compare pairs of generated images for a given prompt and
    pick the preferred one (or declare a tie).

    The full v2 dataset has ~1M comparisons across 14 text-to-image models.
    We provide a 100K reservoir-sampled subset for tractable analysis.

    Each dataset has:

    - **Subjects**: Text-to-image model names (e.g., ``"SDXL-BETA"``,
      ``"SD2.1"``, ``"DP2.0"``).
    - **Comparisons**: One per user preference judgment.
    - **Items**: Text prompts submitted by users.

    Source data: ``yuvalkirstain/pickapic_v2`` on HuggingFace Hub (CC0 license).
    """
    datasets: dict[str, DatasetInfo] = {}

    datasets["pickapic/sampled_100k"] = DatasetInfo(
        name="pickapic/sampled_100k",
        family="pickapic",
        description=(
            "Pick-a-Pic v2 — 100K sampled human preference comparisons "
            "over text-to-image model outputs (14 models)"
        ),
        response_type="pairwise",
        n_subjects=14,
        n_items=0,
        n_comparisons=100000,
        subject_entity="text-to-image model",
        item_entity="prompt",
        filename="pickapic/sampled_100k.pt",
        citation=_PICKAPIC_CITATION,
        url=_PICKAPIC_URL,
        license="CC0-1.0",
        tags=["multimodal", "text-to-image", "pairwise", "preference",
              "human-evaluation", "sampled"],
    )

    return datasets
