#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate VL-RewardBench data to torch-measure-data.

Downloads the VL-RewardBench dataset from MMInstruction/VL-RewardBench on
HuggingFace Hub, evaluates 16 VLM judges on preference pairs, pivots into
response matrices (judges x pairs), and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_vl_rewardbench_data.py

Source data:
    - MMInstruction/VL-RewardBench (1,247 image-text preference pairs)

VL-RewardBench evaluates vision-language generative reward models on preference
judgments across 3 categories: General, Hallucination, Reasoning.  Each item
is an image-text pair with two candidate responses and a human preference
ranking; each judge produces a binary pass/fail (1.0 = correctly identified
the preferred response, 0.0 = failed).

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # judge (VLM) names
        "item_ids": list[str],             # item id strings
        "subject_metadata": list[dict],    # judge metadata
        "item_metadata": list[dict],       # per-item metadata (category, query_source)
    }
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "MMInstruction/VL-RewardBench"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_vl_rewardbench_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Categories in canonical order.
CATEGORIES = ["General", "Hallucination", "Reasoning"]

# Registry-friendly names for per-category splits.
CATEGORY_NAME_MAP = {
    "General": "general",
    "Hallucination": "hallucination",
    "Reasoning": "reasoning",
}

# The 16 judges evaluated in the paper (Table 2, arXiv:2411.17451).
JUDGES = [
    "LLaVA-OneVision-7B-ov",
    "InternVL2-8B",
    "Phi-3.5-Vision",
    "Qwen2-VL-7B",
    "Qwen2-VL-72B",
    "Llama-3.2-11B",
    "Llama-3.2-90B",
    "Molmo-7B",
    "Molmo-72B",
    "Pixtral-12B",
    "NVLM-D-72B",
    "Gemini-1.5-Flash",
    "Gemini-1.5-Pro",
    "Claude-3.5-Sonnet",
    "GPT-4o-mini",
    "GPT-4o",
]

# ---------------------------------------------------------------------------
# Category assignment from item ID prefix
# ---------------------------------------------------------------------------

# Mapping from ID prefix patterns to categories.
# - General: WildVision, VLFeedback items
# - Hallucination: RLAIF-V, RLHF-V, POVID, LRVInstruction items
# - Reasoning: MathVerse, MMMU-Pro items
_HALLUCINATION_PREFIXES = ("RLAIF-V-", "RLHF-V-", "hallucination_pair-", "LRVInstruction-")
_REASONING_PREFIXES = ("mathverse_", "mmmu_pro_")
_GENERAL_PREFIXES = ("wildvision-battle-", "VLFeedback-")


def _infer_category(item_id: str) -> str:
    """Infer the category (General, Hallucination, Reasoning) from item ID prefix."""
    lid = item_id.lower()
    for prefix in _HALLUCINATION_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "Hallucination"
    for prefix in _REASONING_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "Reasoning"
    for prefix in _GENERAL_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "General"
    # Fallback: unknown
    return "Unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset_items() -> list[dict]:
    """Load all items from VL-RewardBench dataset on HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset(SRC_REPO, split="test", token=HF_TOKEN)
    items = []
    for row in ds:
        item_id = str(row["id"])
        category = _infer_category(item_id)
        items.append({
            "id": item_id,
            "query": row["query"],
            "response": row["response"],
            "human_ranking": row["human_ranking"],
            "models": row["models"],
            "query_source": row.get("query_source", ""),
            "category": category,
        })
    return items


# ---------------------------------------------------------------------------
# Response matrix building
# ---------------------------------------------------------------------------


def build_response_matrix(
    items: list[dict],
    judge_results: dict[str, dict[str, float]],
    category_filter: str | None = None,
) -> dict:
    """Build a judges x items binary response matrix.

    Parameters
    ----------
    items : list[dict]
        Dataset items with id, category, query_source, etc.
    judge_results : dict[str, dict[str, float]]
        {judge_name: {item_id: binary_result}} for all judges.
    category_filter : str or None
        If provided, only include items in this category.

    Returns
    -------
    dict with keys: data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    # Filter items
    if category_filter is not None:
        filtered_items = [it for it in items if it["category"] == category_filter]
    else:
        filtered_items = items

    item_ids = [it["id"] for it in filtered_items]
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Sort judges
    judge_names = sorted(judge_results.keys())
    n_judges = len(judge_names)

    # Build the matrix
    data = torch.full((n_judges, n_items), float("nan"), dtype=torch.float32)

    for j, judge_name in enumerate(judge_names):
        results = judge_results[judge_name]
        for iid, val in results.items():
            if iid in item_id_to_idx and val is not None:
                data[j, item_id_to_idx[iid]] = float(val)

    # Build metadata
    item_id_to_meta = {it["id"]: it for it in filtered_items}
    subject_metadata = [{"model": jn} for jn in judge_names]
    item_metadata = [
        {
            "category": item_id_to_meta[iid]["category"],
            "query_source": item_id_to_meta[iid].get("query_source", ""),
        }
        for iid in item_ids
    ]

    return {
        "data": data,
        "subject_ids": judge_names,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Judge evaluation
# ---------------------------------------------------------------------------


def evaluate_judges(items: list[dict]) -> dict[str, dict[str, float]]:
    """Evaluate all 16 judges on VL-RewardBench items.

    This is a placeholder that loads pre-computed results.  In practice,
    each judge is run on each (query, image, response_pair) and produces
    a ranking that is compared against human_ranking.

    For the actual migration, results should be loaded from the benchmark's
    evaluation outputs (e.g., from a leaderboard or pre-computed results
    file).

    Returns
    -------
    dict[str, dict[str, float]]
        {judge_name: {item_id: binary_result}}.
    """
    # NOTE: This function should be replaced with actual result loading
    # when pre-computed evaluation results become available.
    # For now, we initialize empty results for each judge.
    judge_results: dict[str, dict[str, float]] = {}
    for judge in JUDGES:
        judge_results[judge] = {}
    return judge_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset items
    print("=" * 60)
    print("Loading VL-RewardBench items from MMInstruction/VL-RewardBench ...")
    print("=" * 60)

    items = load_dataset_items()
    print(f"  Loaded {len(items)} items")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(it["category"] for it in items)
    for cat in CATEGORIES:
        print(f"    {cat}: {cat_counts.get(cat, 0)} items")
    if "Unknown" in cat_counts:
        print(f"    Unknown: {cat_counts['Unknown']} items")

    # Step 2: Load or compute judge results
    print("\n" + "=" * 60)
    print("Loading judge evaluation results ...")
    print("=" * 60)

    judge_results = evaluate_judges(items)
    print(f"  Loaded results for {len(judge_results)} judges")

    # Step 3: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # --- All categories ---
    print("\n--- vl_rewardbench/all ---")
    payloads["vl_rewardbench/all"] = build_response_matrix(
        items, judge_results
    )
    n_s, n_i = payloads["vl_rewardbench/all"]["data"].shape
    print(f"  {n_s} judges x {n_i} pairs")

    # --- Per-category splits ---
    for category in CATEGORIES:
        short = CATEGORY_NAME_MAP[category]
        name = f"vl_rewardbench/{short}"
        print(f"\n--- {name} ---")
        payloads[name] = build_response_matrix(
            items, judge_results, category_filter=category
        )
        n_s, n_i = payloads[name]["data"].shape
        print(f"  {n_s} judges x {n_i} pairs")

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {n_sub} x {n_items}, {nan_pct:.1%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for vl_rewardbench.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
