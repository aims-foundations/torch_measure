# Copyright (c) 2026 AIMS Foundation. MIT License.

"""PKU-SafeRLHF dataset definitions.

PKU-SafeRLHF is a large-scale safety preference dataset from Peking University
containing expert comparison pairs with dual annotations: helpfulness preference
and safety (harmlessness) preference.  Each sample presents two responses to a
prompt, annotated with:

- **better_response_id**: Which response is more helpful (0 or 1).
- **safer_response_id**: Which response is safer (0 or 1).
- **Safety metadata**: Per-response safety flags, severity levels (0--3), and
  harm category labels across 20 categories.

The dataset has four configurations on HuggingFace:

- ``default``: All three Alpaca model variants combined.
- ``alpaca-7b``: Responses from Alpaca-7B only.
- ``alpaca2-7b``: Responses from Alpaca2-7B only.
- ``alpaca3-8b``: Responses from Alpaca3-8B only.

We register two pairwise comparison datasets per configuration:

1. **Helpfulness**: Uses ``better_response_id`` as the preference signal.
2. **Safety**: Uses ``safer_response_id`` as the preference signal.

Each pairwise comparison payload has:

- **subject_ids**: ``["response_0", "response_1"]`` (2 subjects).
- **Comparisons**: One per sample; the preferred response wins (outcome=1.0).
- **item_contents**: The prompt text for each comparison.
- **comparison_metadata**: Contains both response texts and safety annotations.

Source data: ``PKU-Alignment/PKU-SafeRLHF`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PKU_SAFERLHF_URL = "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF"

_PKU_SAFERLHF_CITATION = (
    "@inproceedings{ji2024pku_saferlhf,\n"
    "  title={PKU-SafeRLHF: Towards Multi-Level Safety Alignment for "
    "LLMs with Human Preference},\n"
    "  author={Ji, Jiaming and Hong, Donghai and Zhang, Borong and "
    "Chen, Boyuan and Dai, Juntao and Zheng, Boren and Qiu, Tianyi "
    "and Zhou, Jiayi and Wang, Kaile and Li, Boxuan and Han, Sirui "
    "and Guo, Yike and Yang, Yaodong},\n"
    "  booktitle={ACL},\n"
    "  year={2025}\n"
    "}"
)

# Per-config sizes: (train, test, total)
_CONFIG_SIZES = {
    "default": (73907, 8211, 82118),
    "alpaca-7b": (27393, 3036, 30429),
    "alpaca2-7b": (25564, 2848, 28412),
    "alpaca3-8b": (20950, 2327, 23277),
}


def _register_pku_saferlhf_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for PKU-SafeRLHF pairwise preference data.

    PKU-SafeRLHF provides expert comparison pairs with dual annotations for
    helpfulness and safety.  Each dataset has:

    - **Subjects**: ``["response_0", "response_1"]`` (2).
    - **Comparisons**: One per sample (preferred response wins).
    - **Items**: Each comparison is a unique item identified by prompt.

    We create entries for both helpfulness and safety preferences, for each
    configuration (default, alpaca-7b, alpaca2-7b, alpaca3-8b) and for both
    train-only and combined (train+test) splits.

    Source data: ``PKU-Alignment/PKU-SafeRLHF`` on HuggingFace Hub
    (CC-BY-NC-4.0 license).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Default config (all Alpaca variants combined) ---

    # Helpfulness: combined train+test
    datasets["helpfulness"] = DatasetInfo(
        name="helpfulness",
        family="pku_saferlhf",
        description=(
            "PKU-SafeRLHF helpfulness preference pairs "
            "(default config, train+test, 82118 comparisons)"
        ),
        response_type="pairwise",
        n_subjects=2,
        n_items=82118,
        n_comparisons=82118,
        subject_entity="response",
        item_entity="prompt_pair",
        filename="helpfulness.pt",
        citation=_PKU_SAFERLHF_CITATION,
        url=_PKU_SAFERLHF_URL,
        license="CC-BY-NC-4.0",
        tags=["rlhf", "preference", "pairwise", "helpfulness", "human-evaluation", "safety"],
    )

    # Safety: combined train+test
    datasets["safety"] = DatasetInfo(
        name="safety",
        family="pku_saferlhf",
        description=(
            "PKU-SafeRLHF safety preference pairs "
            "(default config, train+test, 82118 comparisons)"
        ),
        response_type="pairwise",
        n_subjects=2,
        n_items=82118,
        n_comparisons=82118,
        subject_entity="response",
        item_entity="prompt_pair",
        filename="safety.pt",
        citation=_PKU_SAFERLHF_CITATION,
        url=_PKU_SAFERLHF_URL,
        license="CC-BY-NC-4.0",
        tags=["rlhf", "preference", "pairwise", "safety", "harmlessness", "human-evaluation"],
    )

    # --- Per-config splits ---

    _CONFIG_FRIENDLY = {
        "alpaca-7b": "alpaca7b",
        "alpaca2-7b": "alpaca2_7b",
        "alpaca3-8b": "alpaca3_8b",
    }

    for config, friendly_name in _CONFIG_FRIENDLY.items():
        _train, _test, total = _CONFIG_SIZES[config]

        # Helpfulness
        datasets[f"{friendly_name}_helpfulness"] = DatasetInfo(
            name=f"{friendly_name}_helpfulness",
            family="pku_saferlhf",
            description=(
                f"PKU-SafeRLHF helpfulness preference pairs "
                f"({config} config, train+test, {total} comparisons)"
            ),
            response_type="pairwise",
            n_subjects=2,
            n_items=total,
            n_comparisons=total,
            subject_entity="response",
            item_entity="prompt_pair",
            filename=f"{friendly_name}_helpfulness.pt",
            citation=_PKU_SAFERLHF_CITATION,
            url=_PKU_SAFERLHF_URL,
            license="CC-BY-NC-4.0",
            tags=["rlhf", "preference", "pairwise", "helpfulness", "human-evaluation", "safety"],
        )

        # Safety
        datasets[f"{friendly_name}_safety"] = DatasetInfo(
            name=f"{friendly_name}_safety",
            family="pku_saferlhf",
            description=(
                f"PKU-SafeRLHF safety preference pairs "
                f"({config} config, train+test, {total} comparisons)"
            ),
            response_type="pairwise",
            n_subjects=2,
            n_items=total,
            n_comparisons=total,
            subject_entity="response",
            item_entity="prompt_pair",
            filename=f"{friendly_name}_safety.pt",
            citation=_PKU_SAFERLHF_CITATION,
            url=_PKU_SAFERLHF_URL,
            license="CC-BY-NC-4.0",
            tags=["rlhf", "preference", "pairwise", "safety", "harmlessness", "human-evaluation"],
        )

    return datasets
