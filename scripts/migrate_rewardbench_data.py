#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate RewardBench per-judge results to torch-measure-data.

Downloads per-model JSON files from ``allenai/reward-bench-results`` on
HuggingFace (``eval-set-scores/`` directory), builds a binary response
matrix (reward_models x items), and uploads ``.pt`` files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_rewardbench_data.py

Source data: https://huggingface.co/datasets/allenai/reward-bench-results
    eval-set-scores/{org}/{model}.json

Each JSON file contains per-item binary (0/1) results for a single reward
model across 2,985 (prompt, chosen, rejected) items from the RewardBench
evaluation set.  Items span 23 subsets covering chat, safety, reasoning,
and coding tasks.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, binary 0/1
        "subject_ids": list[str],          # reward model names (org/model)
        "item_ids": list[str],             # "{subset}:{id}" strings
        "subject_metadata": list[dict],    # structured model metadata
    }
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import requests
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_RESULTS_REPO = "allenai/reward-bench-results"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_rewardbench_migration"

# HuggingFace API endpoints
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"

# Reference model that has the 'id' field for constructing item_ids
REFERENCE_MODEL_PATH = "eval-set-scores/openai/gpt-4o-2024-05-13.json"

# RewardBench subset categories (for metadata)
SUBSET_CATEGORIES = {
    "chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "chat_hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

# Invert: subset -> category
_SUBSET_TO_CATEGORY = {}
for cat, subs in SUBSET_CATEGORIES.items():
    for s in subs:
        _SUBSET_TO_CATEGORY[s] = cat


# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------


def _build_subject_metadata(
    model_names: list[str],
    model_types: list[str],
) -> list[dict]:
    """Build structured metadata for each subject (reward model)."""
    metadata = []
    for name, mtype in zip(model_names, model_types):
        parts = name.split("/", 1)
        org = parts[0] if len(parts) > 1 else ""
        short_name = parts[1] if len(parts) > 1 else parts[0]
        metadata.append(
            {
                "model": name,
                "org": org,
                "model_type": mtype,
                "short_name": short_name,
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def list_all_model_files() -> list[str]:
    """List all JSON files under eval-set-scores/ in the HF results repo."""
    print("Enumerating model files from HuggingFace API ...")
    tree_url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/eval-set-scores"
    resp = requests.get(tree_url, timeout=30)
    resp.raise_for_status()
    orgs = [d["path"] for d in resp.json() if d["type"] == "directory"]
    print(f"  Found {len(orgs)} organizations")

    all_files = []
    for org_path in orgs:
        url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/{org_path}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        files = [
            f["path"]
            for f in resp.json()
            if f["type"] == "file" and f["path"].endswith(".json")
        ]
        all_files.extend(files)

    print(f"  Found {len(all_files)} total JSON files")
    return all_files


def download_model_json(file_path: str) -> dict | None:
    """Download and parse a single model's JSON results file."""
    url = f"{HF_RESOLVE_BASE}/{HF_RESULTS_REPO}/resolve/main/{file_path}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    WARNING: Failed to download {file_path}: {e}")
        return None


def get_reference_item_ids() -> tuple[list[str], list[str]]:
    """Download the reference model to extract canonical item IDs and subsets.

    Returns (item_ids, subsets) where item_ids are "{subset}:{id}" strings.
    """
    print(f"Downloading reference model ({REFERENCE_MODEL_PATH}) for item IDs ...")
    data = download_model_json(REFERENCE_MODEL_PATH)
    if data is None:
        raise RuntimeError("Failed to download reference model")

    ids = data["id"]
    subsets = data["subset"]
    item_ids = [f"{subset}:{item_id}" for subset, item_id in zip(subsets, ids)]
    print(f"  {len(item_ids)} items across {len(set(subsets))} subsets")
    return item_ids, subsets


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_response_matrix(
    all_files: list[str],
    ref_subsets: list[str],
) -> tuple[list[str], list[str], list[list[int]], list[dict]]:
    """Download all model files and build the response matrix.

    Returns (subject_ids, model_types, rows, subject_metadata) where rows
    is a list of lists (one per model) of binary 0/1 values.
    """
    n_items = len(ref_subsets)
    subject_ids: list[str] = []
    model_types: list[str] = []
    rows: list[list[int]] = []
    skipped = 0

    for i, fpath in enumerate(all_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Downloading {i + 1}/{len(all_files)} ...")

        data = download_model_json(fpath)
        if data is None:
            skipped += 1
            continue

        results = data.get("results")
        if results is None or len(results) != n_items:
            print(
                f"    WARNING: {fpath} has {len(results) if results else 0} "
                f"results (expected {n_items}), skipping"
            )
            skipped += 1
            continue

        # Verify subset ordering matches reference
        file_subsets = data.get("subset", [])
        if file_subsets and file_subsets != ref_subsets:
            print(f"    WARNING: {fpath} has different subset ordering, skipping")
            skipped += 1
            continue

        model_name = data.get("model", fpath.split("/", 1)[-1].replace(".json", ""))
        model_type = data.get("model_type", "Unknown")

        subject_ids.append(model_name)
        model_types.append(model_type)
        rows.append([int(r) for r in results])

    print(f"  Downloaded: {len(subject_ids)}, Skipped: {skipped}")
    return subject_ids, model_types, rows, _build_subject_metadata(subject_ids, model_types)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def build_payload(
    subject_ids: list[str],
    item_ids: list[str],
    rows: list[list[int]],
    subject_metadata: list[dict],
) -> dict:
    """Build the .pt payload dict."""
    data = torch.tensor(rows, dtype=torch.float32)
    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


def build_subset_payload(
    payload: dict,
    subsets: list[str],
    target_subsets: list[str],
) -> dict:
    """Extract a subset of items from the full payload."""
    mask = [s in target_subsets for s in subsets]
    indices = [i for i, m in enumerate(mask) if m]

    data = payload["data"][:, indices]
    item_ids = [payload["item_ids"][i] for i in indices]

    return {
        "data": data,
        "subject_ids": payload["subject_ids"],
        "item_ids": item_ids,
        "subject_metadata": payload["subject_metadata"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Enumerate files
    print("=" * 60)
    print("Step 1: Enumerating model files")
    print("=" * 60)
    all_files = list_all_model_files()

    # Step 2: Get reference item IDs
    print("\n" + "=" * 60)
    print("Step 2: Getting reference item IDs")
    print("=" * 60)
    item_ids, ref_subsets = get_reference_item_ids()

    # Step 3: Download all models and build matrix
    print("\n" + "=" * 60)
    print("Step 3: Downloading all model results")
    print("=" * 60)
    subject_ids, model_types, rows, subject_metadata = build_response_matrix(
        all_files, ref_subsets
    )

    # Step 4: Build payloads
    print("\n" + "=" * 60)
    print("Step 4: Building response matrices")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # Full matrix
    print("\n--- rewardbench/results (all items) ---")
    full_payload = build_payload(subject_ids, item_ids, rows, subject_metadata)
    payloads["rewardbench/results"] = full_payload
    n_s, n_i = full_payload["data"].shape
    acc = full_payload["data"].mean().item()
    print(f"  {n_s} reward models x {n_i} items")
    print(f"  Overall accuracy: {acc:.3f}")

    # Per-category splits
    for cat_name, cat_subsets in SUBSET_CATEGORIES.items():
        key = f"rewardbench/{cat_name}"
        print(f"\n--- {key} ---")
        cat_payload = build_subset_payload(full_payload, ref_subsets, cat_subsets)
        payloads[key] = cat_payload
        n_s, n_i = cat_payload["data"].shape
        acc = cat_payload["data"].mean().item()
        print(f"  {n_s} reward models x {n_i} items")
        print(f"  Category accuracy: {acc:.3f}")

    # Step 5: Save and upload
    print("\n" + "=" * 60)
    print("Step 5: Saving and uploading")
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
        )

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {HF_RESULTS_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for rewardbench.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")

    # Print model type distribution
    from collections import Counter

    type_counts = Counter(model_types)
    print("\nModel type distribution:")
    for mtype, count in type_counts.most_common():
        print(f"  {mtype}: {count}")

    # Print subset item counts
    from collections import Counter as C2

    subset_counts = C2(ref_subsets)
    print("\nSubset item counts:")
    for subset, count in sorted(subset_counts.items()):
        cat = _SUBSET_TO_CATEGORY.get(subset, "?")
        print(f"  {subset}: {count} items ({cat})")

    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
