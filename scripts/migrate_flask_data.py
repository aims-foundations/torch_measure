#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate FLASK evaluation data to torch-measure-data.

Downloads GPT-4 review results from the kaistAI/FLASK GitHub repository,
pivots into response matrices (models x instructions) for the overall
mean and each of the 12 fine-grained skills, and uploads .pt files to
HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_flask_data.py

Source data:
    - kaistAI/FLASK on GitHub: gpt_review/outputs/<model>.jsonl
    - 15 models evaluated on 1,700 instructions
    - 12 fine-grained skills scored on 1-5 scale by GPT-4

FLASK defines 4 primary skill categories with 12 fine-grained skills:
    - Logical Thinking: logical_correctness, logical_robustness, logical_efficiency
    - Background Knowledge: factuality, commonsense_understanding
    - Problem Handling: comprehension, insightfulness, completeness, metacognition
    - User Alignment: conciseness, readability, harmlessness

Each instruction is evaluated on 2-3 relevant skills (not all 12), so per-skill
response matrices contain NaN where a skill does not apply to an item.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # instruction IDs (question_id as str)
        "subject_metadata": list[dict],    # per-model metadata
        "item_metadata": list[dict],       # per-item metadata (domain, difficulty, skills)
    }
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_BASE_URL = "https://raw.githubusercontent.com/kaistAI/FLASK/main"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_flask_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Model review files in gpt_review/outputs/ (filename -> display name).
MODEL_FILES: dict[str, str] = {
    "alpaca_13b.jsonl": "alpaca-13b",
    "bard_review.jsonl": "bard",
    "chatgpt_review.jsonl": "chatgpt",
    "claude_v1_review.jsonl": "claude-v1",
    "davinci_003_review.jsonl": "davinci-003",
    "gpt4_review.jsonl": "gpt-4",
    "llama2_chat_13b.jsonl": "llama-2-chat-13b",
    "llama2_chat_70b.jsonl": "llama-2-chat-70b",
    "tulu_7b_review.jsonl": "tulu-7b",
    "tulu_13b_review.jsonl": "tulu-13b",
    "tulu_30b_review.jsonl": "tulu-30b",
    "tulu_65b_review.jsonl": "tulu-65b",
    "vicuna_13b.jsonl": "vicuna-13b",
    "vicuna_33b.jsonl": "vicuna-33b",
    "wizardlm_13b.jsonl": "wizardlm-13b",
}

# Normalize variant score key names to canonical skill names.
SKILL_NORMALIZE: dict[str, str] = {
    "commonsense understanding": "commonsense_understanding",
    "commonsense": "commonsense_understanding",
    "commonsense reasoning": "commonsense_understanding",
    "completeness": "completeness",
    "comprehension": "comprehension",
    "comprehension score": "comprehension",
    "conciseness": "conciseness",
    "conciseness score": "conciseness",
    "factuality": "factuality",
    "harmlessness": "harmlessness",
    "insightfulness": "insightfulness",
    "logical correctness": "logical_correctness",
    "logical efficiency": "logical_efficiency",
    "logical robustness": "logical_robustness",
    "metacognition": "metacognition",
    "readability": "readability",
    "readability score": "readability",
}

# Canonical skill ordering.
SKILLS = [
    "logical_correctness",
    "logical_robustness",
    "logical_efficiency",
    "factuality",
    "commonsense_understanding",
    "comprehension",
    "insightfulness",
    "completeness",
    "metacognition",
    "conciseness",
    "readability",
    "harmlessness",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_review_file(filename: str) -> list[dict]:
    """Download a single model review JSONL from the FLASK GitHub repo."""
    url = f"{SRC_BASE_URL}/gpt_review/outputs/{filename}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    entries = []
    for line in text.strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def download_evaluation_set() -> list[dict]:
    """Download the evaluation set (instruction metadata) from FLASK GitHub."""
    url = f"{SRC_BASE_URL}/evaluation_set/flask_evaluation.jsonl"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    entries = []
    for line in text.strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def download_all_reviews() -> dict[str, list[dict]]:
    """Download review data for all models.

    Returns {display_name: [entry_dict, ...]}.
    """
    all_reviews: dict[str, list[dict]] = {}
    for filename, display_name in sorted(MODEL_FILES.items()):
        print(f"  Downloading {filename} ...")
        entries = download_review_file(filename)
        all_reviews[display_name] = entries
        print(f"    {len(entries)} entries")
    return all_reviews


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def parse_scores(entry: dict) -> dict[str, float]:
    """Extract normalized skill -> score mapping from a review entry.

    Returns {canonical_skill: float_score}, skipping N/A values.
    """
    scores: dict[str, float] = {}
    raw_scores = entry.get("score", {})
    for key, val in raw_scores.items():
        norm_skill = SKILL_NORMALIZE.get(key.lower())
        if norm_skill is None:
            continue
        if isinstance(val, (int, float)):
            scores[norm_skill] = float(val)
        # Skip "N/A" and other non-numeric values
    return scores


def build_response_matrices(
    all_reviews: dict[str, list[dict]],
    eval_set: list[dict],
) -> dict[str, dict]:
    """Build response matrices for overall and each skill.

    Parameters
    ----------
    all_reviews : dict
        {model_name: [entry_dict, ...]} from GPT-4 reviews.
    eval_set : list[dict]
        Evaluation set entries with instruction metadata.

    Returns
    -------
    dict[str, dict]
        {dataset_name: payload_dict} where payload_dict has keys:
        data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    # Determine item ordering from the first model's review data
    first_model = next(iter(all_reviews.values()))
    item_ids = [str(entry["question_id"]) for entry in first_model]
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Build evaluation set metadata lookup (idx -> metadata)
    eval_meta: dict[str, dict] = {}
    for entry in eval_set:
        idx_str = str(entry["idx"])
        eval_meta[idx_str] = {
            "domain": entry.get("domain", ""),
            "difficulty": entry.get("difficulty", ""),
            "skills": entry.get("skill", []),
            "task": entry.get("task", ""),
        }

    # Sort models alphabetically
    model_names = sorted(all_reviews.keys())
    n_models = len(model_names)

    # Build per-skill score matrices: skill -> (n_models, n_items) with NaN
    skill_matrices: dict[str, torch.Tensor] = {}
    for skill in SKILLS:
        skill_matrices[skill] = torch.full(
            (n_models, n_items), float("nan"), dtype=torch.float32
        )

    # Also build an overall mean matrix
    overall_sum = torch.zeros((n_models, n_items), dtype=torch.float32)
    overall_count = torch.zeros((n_models, n_items), dtype=torch.float32)

    for m, model_name in enumerate(model_names):
        entries = all_reviews[model_name]
        for entry in entries:
            qid = str(entry["question_id"])
            if qid not in item_id_to_idx:
                continue
            i = item_id_to_idx[qid]
            scores = parse_scores(entry)
            for skill, score in scores.items():
                if skill in skill_matrices:
                    skill_matrices[skill][m, i] = score
                    overall_sum[m, i] += score
                    overall_count[m, i] += 1.0

    # Compute overall mean (NaN where no skills were scored)
    overall_data = torch.where(
        overall_count > 0,
        overall_sum / overall_count,
        torch.tensor(float("nan")),
    )

    # Build item metadata
    item_metadata = []
    for iid in item_ids:
        meta = eval_meta.get(iid, {})
        item_metadata.append(meta)

    # Build subject metadata
    subject_metadata = [{"model": name} for name in model_names]

    # Assemble payloads
    payloads: dict[str, dict] = {}

    # Overall
    payloads["flask/overall"] = {
        "data": overall_data,
        "subject_ids": model_names,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }

    # Per-skill: filter to only items where that skill is applicable
    for skill in SKILLS:
        full_matrix = skill_matrices[skill]

        # Find items that have at least one non-NaN score for this skill
        has_score = ~torch.isnan(full_matrix)
        item_has_any = has_score.any(dim=0)  # (n_items,)
        valid_indices = torch.where(item_has_any)[0].tolist()

        if len(valid_indices) == 0:
            print(f"  Warning: no valid items for skill {skill}, skipping")
            continue

        # Slice to only valid items
        filtered_data = full_matrix[:, valid_indices]
        filtered_item_ids = [item_ids[i] for i in valid_indices]
        filtered_item_metadata = [item_metadata[i] for i in valid_indices]

        payloads[f"flask/{skill}"] = {
            "data": filtered_data,
            "subject_ids": model_names,
            "item_ids": filtered_item_ids,
            "subject_metadata": subject_metadata,
            "item_metadata": filtered_item_metadata,
        }

    return payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download evaluation set metadata
    print("=" * 60)
    print("Downloading FLASK evaluation set metadata ...")
    print("=" * 60)
    eval_set = download_evaluation_set()
    print(f"  {len(eval_set)} instructions in evaluation set")

    # Step 2: Download all model review files
    print("\n" + "=" * 60)
    print("Downloading model review files from kaistAI/FLASK ...")
    print("=" * 60)
    all_reviews = download_all_reviews()
    print(f"\n  Downloaded reviews for {len(all_reviews)} models")

    # Step 3: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)
    payloads = build_response_matrices(all_reviews, eval_set)

    for name, payload in sorted(payloads.items()):
        n_s, n_i = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {name}: {n_s} models x {n_i} items, {nan_pct:.1%} missing")

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
    print(f"  Source: {SRC_BASE_URL}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for flask.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
