#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate JudgeBench data to torch-measure-data.

Downloads per-judge evaluation results from the ScalerLab/JudgeBench GitHub
repository, pivots into response matrices (judges x pairs), and uploads
.pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_judgebench_data.py

Source data:
    - Items: ScalerLab/JudgeBench on HuggingFace (350 GPT + 270 Claude pairs)
    - Results: ScalerLab/JudgeBench on GitHub (per-judge evaluation outputs)

JudgeBench evaluates LLM-as-judge capabilities on challenging response pairs
with objectively verifiable correctness labels across 4 categories:
knowledge (MMLU-Pro), reasoning (LiveBench), math (LiveBench + MMLU-Pro),
and coding (LiveCodeBench).

Each item is a (question, response_A, response_B) triple with a ground-truth
label (A>B or B>A).  Each judge produces two judgments per pair (original and
swapped order).  A judge is scored as correct (1.0) if its net preference
aligns with the ground truth, incorrect (0.0) if it opposes, and NaN on ties.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # judge identifiers
        "item_ids": list[str],             # pair_id strings
        "subject_metadata": list[dict],    # judge metadata (judge_name, judge_model)
        "item_metadata": list[dict],       # per-item metadata (source, category)
    }
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote as urlquote

import requests
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_HF_REPO = "ScalerLab/JudgeBench"
SRC_GH_REPO = "ScalerLab/JudgeBench"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_judgebench_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# We use the GPT split (350 pairs, 33 judges) which has the most judge coverage.
RESPONSE_MODEL = "gpt-4o-2024-05-13"

# Category mapping (source -> category)
_KNOWLEDGE_SOURCES = [
    "mmlu-pro-biology",
    "mmlu-pro-business",
    "mmlu-pro-chemistry",
    "mmlu-pro-computer science",
    "mmlu-pro-economics",
    "mmlu-pro-engineering",
    "mmlu-pro-health",
    "mmlu-pro-history",
    "mmlu-pro-law",
    "mmlu-pro-other",
    "mmlu-pro-philosophy",
    "mmlu-pro-physics",
    "mmlu-pro-psychology",
]

SOURCE_TO_CATEGORY: dict[str, str] = {}
for _s in _KNOWLEDGE_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "knowledge"
for _s in ["livebench-math", "mmlu-pro-math"]:
    SOURCE_TO_CATEGORY[_s] = "math"
SOURCE_TO_CATEGORY["livebench-reasoning"] = "reasoning"
SOURCE_TO_CATEGORY["livecodebench"] = "coding"

CATEGORIES = ["knowledge", "reasoning", "math", "coding"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _gh_raw_url(path: str) -> str:
    """Build a raw GitHub URL, encoding path components."""
    encoded = urlquote(path, safe="/")
    return f"https://raw.githubusercontent.com/{SRC_GH_REPO}/main/{encoded}"


def list_output_files() -> list[str]:
    """List all JSONL output files for the target response model from GitHub."""
    api_url = f"https://api.github.com/repos/{SRC_GH_REPO}/contents/outputs"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    target_prefix = f"dataset=judgebench,response_model={RESPONSE_MODEL},"
    return sorted(
        f["name"]
        for f in files
        if f["name"].startswith(target_prefix) and f["name"].endswith(".jsonl")
    )


def parse_filename(filename: str) -> dict[str, str]:
    """Parse a JudgeBench output filename into its components."""
    name = filename.replace(".jsonl", "")
    parts = name.split(",")
    result = {}
    for part in parts:
        key, value = part.split("=", 1)
        result[key] = value
    return result


def download_output_file(filename: str) -> list[dict]:
    """Download and parse a single JSONL output file from GitHub."""
    url = _gh_raw_url(f"outputs/{filename}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    return [json.loads(line) for line in lines]


def flip_decision(decision: str) -> str:
    """Flip a judgment decision: A>B <-> B>A."""
    if decision == "A>B":
        return "B>A"
    elif decision == "B>A":
        return "A>B"
    return decision


def extract_decision(response_text: str) -> str | None:
    """Extract a decision (A>B or B>A) from a judge's text response.

    Handles formats like 'Output (a)', 'Output (b)', 'A>B', 'B>A', etc.
    """
    if not response_text:
        return None

    text = response_text.strip().lower()

    # Check for explicit A>B / B>A patterns
    if "a>b" in text or "a > b" in text:
        return "A>B"
    if "b>a" in text or "b > a" in text:
        return "B>A"

    # Check for 'Output (a)' / 'Output (b)' patterns
    if "output (a)" in text:
        return "A>B"
    if "output (b)" in text:
        return "B>A"

    # Check for 'Response A' / 'Response B' patterns
    if re.search(r"\bresponse\s*a\b", text):
        return "A>B"
    if re.search(r"\bresponse\s*b\b", text):
        return "B>A"

    # Check if judgment has a 'decision' field already parsed
    return None


def score_judge_on_pair(pair: dict, reverse_order: bool = True) -> float | None:
    """Score a judge's performance on a single pair.

    Uses the same logic as the original JudgeBench metrics:
    - With reverse_order: run twice (original + swapped), net vote determines
      correctness.  +1 for matching ground truth, -1 for opposing.
      Correct if net > 0, incorrect if net < 0, None (tie) if net == 0.
    - Without reverse_order: single judgment, binary match.

    Returns 1.0 (correct), 0.0 (incorrect), or None (inconclusive/null).
    """
    label = pair["label"]
    judgments = pair.get("judgments", [])

    if not judgments:
        return None

    if reverse_order and len(judgments) >= 2:
        # Two judgments: original order + swapped order
        decision_1 = extract_decision(judgments[0].get("judgment", {}).get("response", ""))
        decision_2_raw = extract_decision(judgments[1].get("judgment", {}).get("response", ""))

        # The second judgment was made with swapped order, so flip its decision
        decision_2 = flip_decision(decision_2_raw) if decision_2_raw else None

        counter = 0
        if decision_1 is not None:
            if decision_1 == label:
                counter += 1
            elif decision_1 == flip_decision(label):
                counter -= 1
        if decision_2 is not None:
            if decision_2 == label:
                counter += 1
            elif decision_2 == flip_decision(label):
                counter -= 1

        if counter > 0:
            return 1.0
        elif counter < 0:
            return 0.0
        else:
            return None  # tie / inconclusive
    else:
        # Single judgment
        decision = extract_decision(judgments[0].get("judgment", {}).get("response", ""))
        if decision is None:
            return None
        return 1.0 if decision == label else 0.0


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def build_response_matrix(
    judge_results: dict[str, list[dict]],
    item_ids: list[str] | None = None,
    category_filter: str | None = None,
) -> dict:
    """Build a judges x pairs binary response matrix.

    Parameters
    ----------
    judge_results : dict
        {judge_key: [list of pair dicts with judgments]} for each judge.
    item_ids : list[str] or None
        Ordered pair IDs.  If None, inferred from first judge.
    category_filter : str or None
        If provided, only include pairs in this category.

    Returns
    -------
    dict with keys: data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    # Determine item ordering from first judge
    if item_ids is None:
        first_judge_data = next(iter(judge_results.values()))
        item_ids = [p["pair_id"] for p in first_judge_data]

    # Build pair_id -> source mapping
    first_judge_data = next(iter(judge_results.values()))
    id_to_source = {p["pair_id"]: p["source"] for p in first_judge_data}
    id_to_category = {
        pid: SOURCE_TO_CATEGORY.get(src, "unknown")
        for pid, src in id_to_source.items()
    }

    # Apply category filter
    if category_filter is not None:
        item_ids = [pid for pid in item_ids if id_to_category.get(pid) == category_filter]

    item_id_to_idx = {pid: idx for idx, pid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Sort judges alphabetically
    judge_keys = sorted(judge_results.keys())
    n_judges = len(judge_keys)

    # Build the matrix
    data = torch.full((n_judges, n_items), float("nan"), dtype=torch.float32)

    for j, judge_key in enumerate(judge_keys):
        pairs = judge_results[judge_key]
        # Build pair_id -> pair mapping for this judge
        pair_map = {p["pair_id"]: p for p in pairs}

        for pid, idx in item_id_to_idx.items():
            if pid in pair_map:
                score = score_judge_on_pair(pair_map[pid])
                if score is not None:
                    data[j, idx] = score

    # Build subject metadata
    subject_metadata = []
    for judge_key in judge_keys:
        parts = judge_key.split("/", 1)
        subject_metadata.append(
            {
                "judge_name": parts[0] if len(parts) >= 1 else judge_key,
                "judge_model": parts[1] if len(parts) >= 2 else judge_key,
            }
        )

    # Build item metadata
    item_metadata = []
    for pid in item_ids:
        source = id_to_source.get(pid, "")
        item_metadata.append(
            {
                "source": source,
                "category": SOURCE_TO_CATEGORY.get(source, "unknown"),
            }
        )

    return {
        "data": data,
        "subject_ids": judge_keys,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: List output files
    print("=" * 60)
    print("Listing output files from ScalerLab/JudgeBench ...")
    print("=" * 60)

    output_files = list_output_files()
    print(f"  Found {len(output_files)} output files for {RESPONSE_MODEL}")

    # Step 2: Download and parse output files
    print("\n" + "=" * 60)
    print("Downloading judge output files ...")
    print("=" * 60)

    judge_results: dict[str, list[dict]] = {}
    for i, filename in enumerate(output_files):
        meta = parse_filename(filename)
        judge_key = f"{meta['judge_name']}/{meta['judge_model']}"
        try:
            pairs = download_output_file(filename)
            judge_results[judge_key] = pairs
            print(f"  [{i + 1}/{len(output_files)}] {judge_key}: {len(pairs)} pairs")
        except Exception as e:
            print(f"  [{i + 1}/{len(output_files)}] Warning: failed {judge_key}: {e}")

    print(f"\n  Downloaded results for {len(judge_results)} judges")

    # Determine global item ordering from first judge
    first_judge_data = next(iter(judge_results.values()))
    item_ids = [p["pair_id"] for p in first_judge_data]
    print(f"  Total items: {len(item_ids)}")

    # Step 3: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # --- All categories ---
    print("\n--- judgebench/all ---")
    payloads["judgebench/all"] = build_response_matrix(
        judge_results, item_ids=item_ids
    )
    n_s, n_i = payloads["judgebench/all"]["data"].shape
    print(f"  {n_s} judges x {n_i} pairs")

    # --- Per-category splits ---
    for category in CATEGORIES:
        name = f"judgebench/{category}"
        print(f"\n--- {name} ---")
        payloads[name] = build_response_matrix(
            judge_results, item_ids=item_ids, category_filter=category
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
    print(f"  Source: {SRC_HF_REPO} (HuggingFace) + {SRC_GH_REPO} (GitHub)")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for judgebench.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
