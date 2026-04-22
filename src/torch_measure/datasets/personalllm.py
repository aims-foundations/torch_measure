# Copyright (c) 2026 AIMS Foundations. MIT License.

"""PersonalLLM dataset definitions.

PersonalLLM (Namkoong Lab, Columbia) studies personalized preference modeling
by evaluating LLM-generated responses through diverse reward models that
serve as proxies for users with heterogeneous preferences.

Each response matrix has:

- **Rows (subjects)**: 10 reward models acting as user proxies with distinct
  preference profiles.
- **Columns (items)**: (prompt, response) pairs — each of 10,402 prompts has
  8 responses from different LLMs, yielding up to 83,216 items total.
- **Values**: Continuous reward scores (unbounded, model-specific scale).

The dataset enables research on preference heterogeneity and personalized
alignment: different reward models exhibit systematically different
preferences, analogous to different human users.

Source data: ``namkoong-lab/PersonalLLM`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PERSONALLLM_URL = "https://huggingface.co/datasets/namkoong-lab/PersonalLLM"

_PERSONALLLM_CITATION = (
    "@article{2024personalllm,\n"
    "  title={PersonalLLM: Tailoring LLMs to Individual Preferences},\n"
    "  author={Zollo, Thomas P. and Siah, Andrew Wei Tung and Ye, Naimeng "
    "and Li, Ang and Namkoong, Hongseok},\n"
    "  journal={arXiv preprint arXiv:2409.20296},\n"
    "  year={2024}\n"
    "}"
)

# Reward models used as user proxies (subjects).
REWARD_MODELS = [
    "weqweasdas/RM-Gemma-2B",
    "weqweasdas/RM-Gemma-7B",
    "hendrydong/Mistral-RM-for-RAFT-GSHF-v0",
    "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback",
    "weqweasdas/RM-Mistral-7B",
    "sfairXC/FsfairX-LLaMA3-RM-v0.1",
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "PKU-Alignment/beaver-7b-v1.0-cost",
    "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
    "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
]

# LLMs that generated responses (8 per prompt).
RESPONSE_MODELS = [
    "cohere/command-r-plus",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "meta-llama/llama-3-70b-instruct:nitro",
    "google/gemini-pro-1.5",
    "mistralai/mixtral-8x22b-instruct",
]


def _register_personalllm_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for PersonalLLM response matrices.

    PersonalLLM uses 10 reward models as proxies for users with diverse
    preferences.  Each of 10,402 prompts has 8 LLM responses scored by all
    10 reward models, yielding a response matrix of reward models x
    (prompt, response) pairs with continuous reward scores.

    Variants:

    - ``personalllm/all``: All prompts (train + test), 10 x 83,216.
    - ``personalllm/train``: Train split only, 10 x 75,216.
    - ``personalllm/test``: Test split only, 10 x 8,000.
    """
    d: dict[str, DatasetInfo] = {}

    # --- All prompts (train + test), all 8 responses per prompt ---
    d["all"] = DatasetInfo(
        name="all",
        family="personalllm",
        description=(
            "PersonalLLM — all prompts, reward-model scores on LLM responses "
            "(10 reward models x 83,216 prompt-response pairs)"
        ),
        response_type="continuous",
        n_subjects=10,
        n_items=83216,
        subject_entity="reward_model",
        item_entity="prompt_response",
        filename="all.pt",
        citation=_PERSONALLLM_CITATION,
        url=_PERSONALLLM_URL,
        license="CC-BY-4.0",
        tags=["personalization", "preference", "reward-model", "heterogeneous"],
    )

    # --- Train split only ---
    d["train"] = DatasetInfo(
        name="train",
        family="personalllm",
        description=(
            "PersonalLLM train split — reward-model scores on LLM responses "
            "(10 reward models x 75,216 prompt-response pairs)"
        ),
        response_type="continuous",
        n_subjects=10,
        n_items=75216,
        subject_entity="reward_model",
        item_entity="prompt_response",
        filename="train.pt",
        citation=_PERSONALLLM_CITATION,
        url=_PERSONALLLM_URL,
        license="CC-BY-4.0",
        tags=["personalization", "preference", "reward-model", "train"],
    )

    # --- Test split only ---
    d["test"] = DatasetInfo(
        name="test",
        family="personalllm",
        description=(
            "PersonalLLM test split — reward-model scores on LLM responses "
            "(10 reward models x 8,000 prompt-response pairs)"
        ),
        response_type="continuous",
        n_subjects=10,
        n_items=8000,
        subject_entity="reward_model",
        item_entity="prompt_response",
        filename="test.pt",
        citation=_PERSONALLLM_CITATION,
        url=_PERSONALLLM_URL,
        license="CC-BY-4.0",
        tags=["personalization", "preference", "reward-model", "test"],
    )

    return d
