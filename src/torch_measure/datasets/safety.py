# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Safety, bias, and red teaming dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_REPO = "aims-foundation/torch-measure-data"

_BBQ_CITATION = (
    "@inproceedings{parrish2022bbq,\n"
    "  title={BBQ: A Hand-Built Bias Benchmark for Question Answering},\n"
    "  author={Parrish, Alicia and Chen, Angelica and Nangia, Nikita and others},\n"
    "  booktitle={ACL Findings},\n"
    "  year={2022}\n"
    "}"
)

_JBB_CITATION = (
    "@article{chao2024jailbreakbench,\n"
    "  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},\n"
    "  author={Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and others},\n"
    "  year={2024},\n"
    "  url={https://github.com/JailbreakBench/jailbreakbench}\n"
    "}"
)


def _register_safety_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for safety, bias, and red teaming datasets."""
    d: dict[str, DatasetInfo] = {}

    # ── BBQ (Bias Benchmark for QA) ────────────────────────────────────

    d["safety/bbq"] = DatasetInfo(
        name="safety/bbq",
        family="safety",
        description="BBQ: 7 models x 58,492 bias QA items across 11 demographic categories",
        response_type="binary",
        n_subjects=7,
        n_items=58492,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_REPO,
        filename="safety/bbq.pt",
        citation=_BBQ_CITATION,
        url="https://github.com/nyu-mll/BBQ",
        license="CC-BY-4.0",
        tags=["safety", "bias", "fairness", "qa", "demographic"],
    )

    # ── JailbreakBench ─────────────────────────────────────────────────

    for method in ["pair", "gcg", "dsn", "jbc", "prompt_with_random_search"]:
        n_models = {"pair": 4, "gcg": 4, "dsn": 2, "jbc": 4, "prompt_with_random_search": 4}[method]
        d[f"safety/jailbreakbench_{method}"] = DatasetInfo(
            name=f"safety/jailbreakbench_{method}",
            family="safety",
            description=f"JailbreakBench {method.upper()}: {n_models} models x 100 harmful behaviors, binary jailbroken",
            response_type="binary",
            n_subjects=n_models,
            n_items=100,
            subject_entity="LLM",
            item_entity="behavior",
            repo_id=_REPO,
            filename=f"safety/jailbreakbench_{method}.pt",
            citation=_JBB_CITATION,
            url="https://github.com/JailbreakBench/jailbreakbench",
            license="MIT",
            tags=["safety", "red-teaming", "jailbreak", method],
        )

    d["safety/jailbreakbench_all"] = DatasetInfo(
        name="safety/jailbreakbench_all",
        family="safety",
        description="JailbreakBench all methods: 18 (model x method) x 100 behaviors, binary jailbroken",
        response_type="binary",
        n_subjects=18,
        n_items=100,
        subject_entity="LLM",
        item_entity="behavior",
        repo_id=_REPO,
        filename="safety/jailbreakbench_all.pt",
        citation=_JBB_CITATION,
        url="https://github.com/JailbreakBench/jailbreakbench",
        license="MIT",
        tags=["safety", "red-teaming", "jailbreak", "combined"],
    )

    return d
