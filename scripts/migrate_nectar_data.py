#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Berkeley-NEST Nectar data to torch-measure-data.

Downloads the Nectar dataset (berkeley-nest/Nectar) from HuggingFace Hub via
streaming, builds a response matrix (models x prompts), and uploads .pt files
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_nectar_data.py

Source data: https://huggingface.co/datasets/berkeley-nest/Nectar

Nectar contains 182,954 prompts, each with 7 ranked responses from diverse
models (GPT-4, GPT-3.5-turbo, Llama-2-7B-chat, Mistral-7B-instruct, etc.),
ranked 1-7 by GPT-4.

The response matrix has:
- **Rows (subjects)**: LLM model names (39 unique models).
- **Columns (items)**: Individual prompts (identified by index).
- **Values**: Normalized rank scores in [0, 1], where rank 1 -> 1.0 and
  rank 7 -> 0.0, computed as (7 - rank) / 6.  NaN for missing entries
  (not every model responds to every prompt).

Because the full dataset is large (183K prompts x 39 models), we process
all prompts but the resulting matrix will be sparse (~18% filled per model
on average).  A 50K-prompt random subset is also produced for lighter use.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # prompt identifiers (index-based)
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import random
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_DATASET = "berkeley-nest/Nectar"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_nectar_migration"

# Maximum number of prompts to process (None = all).
# Set to 50_000 for the subset version.
MAX_PROMPTS_FULL = None  # process all
MAX_PROMPTS_SUBSET = 50_000

RANDOM_SEED = 42

# Rank normalization: rank 1 -> 1.0, rank 7 -> 0.0
MAX_RANK = 7


def normalize_rank(rank: float) -> float:
    """Convert rank (1=best, 7=worst) to score (1.0=best, 0.0=worst)."""
    return (MAX_RANK - rank) / (MAX_RANK - 1)


# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_ORG_MAP: dict[str, str] = {
    "gpt-4": "OpenAI",
    "gpt-3.5": "OpenAI",
    "mistral": "Mistral AI",
    "llama": "Meta",
    "vicuna": "LMSYS",
    "alpaca": "Stanford",
    "koala": "Berkeley",
    "mpt": "MosaicML",
    "wizardlm": "Microsoft",
    "dolly": "Databricks",
    "chatglm": "Tsinghua",
    "starchat": "HuggingFace",
    "falcon": "TII",
    "bard": "Google",
    "fastchat": "LMSYS",
    "oasst": "LAION",
    "claude": "Anthropic",
    "anthropic": "Anthropic",
    "rwkv": "RWKV",
    "guanaco": "UW",
    "stablelm": "Stability AI",
    "gpt4all": "Nomic AI",
    "palm": "Google",
    "pythia": "EleutherAI",
    "ultralm": "Microsoft",
}


def _infer_org(model_name: str) -> str:
    """Infer organization from model name."""
    lower = model_name.lower()
    for prefix, org in _ORG_MAP.items():
        if lower.startswith(prefix):
            return org
    return ""


def _build_subject_metadata(model_names: list[str]) -> list[dict]:
    """Build structured metadata for each model."""
    metadata = []
    for name in model_names:
        metadata.append(
            {
                "model": name,
                "org": _infer_org(name),
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Data loading & matrix construction
# ---------------------------------------------------------------------------


def stream_nectar_data() -> list[dict]:
    """Stream all rows from Nectar into a list of lightweight records.

    Each record is: {prompt_idx: int, model: str, rank: float}
    We collect all prompts and their response metadata (model + rank).
    """
    print(f"Streaming {SRC_DATASET} ...")
    ds = load_dataset(SRC_DATASET, split="train", streaming=True)

    # Collect: for each prompt, the (model, rank) pairs
    all_records: list[list[tuple[str, float]]] = []
    n_prompts = 0

    for i, sample in enumerate(ds):
        answers = sample["answers"]
        record = []
        for ans in answers:
            record.append((ans["model"], float(ans["rank"])))
        all_records.append(record)
        n_prompts += 1

        if n_prompts % 20_000 == 0:
            print(f"  Streamed {n_prompts:,} prompts ...")

    print(f"  Total: {n_prompts:,} prompts streamed.")
    return all_records


def build_response_matrix(
    all_records: list[list[tuple[str, float]]],
    prompt_indices: list[int] | None = None,
) -> dict:
    """Build a response matrix from collected records.

    Parameters
    ----------
    all_records : list of list of (model, rank) tuples
        One entry per prompt.
    prompt_indices : list of int or None
        If provided, only include these prompt indices.
        If None, include all prompts.

    Returns
    -------
    dict
        Payload with data, subject_ids, item_ids, subject_metadata.
    """
    # Determine which prompts to use
    if prompt_indices is not None:
        selected = prompt_indices
    else:
        selected = list(range(len(all_records)))

    # Collect all model names across selected prompts
    model_set: set[str] = set()
    for idx in selected:
        for model, _rank in all_records[idx]:
            model_set.add(model)

    model_names = sorted(model_set)
    model_to_idx = {m: i for i, m in enumerate(model_names)}

    n_subjects = len(model_names)
    n_items = len(selected)

    print(f"  Building matrix: {n_subjects} models x {n_items} prompts ...")

    # Initialize with NaN
    data = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)

    # Fill in normalized ranks
    for col, prompt_idx in enumerate(selected):
        for model, rank in all_records[prompt_idx]:
            row = model_to_idx[model]
            data[row, col] = normalize_rank(rank)

    # Build item IDs
    item_ids = [f"prompt_{idx}" for idx in selected]
    subject_metadata = _build_subject_metadata(model_names)

    return {
        "data": data,
        "subject_ids": model_names,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Stream all data
    print("=" * 60)
    print("Step 1: Streaming Nectar dataset ...")
    print("=" * 60)
    all_records = stream_nectar_data()
    n_total = len(all_records)

    payloads: dict[str, dict] = {}

    # Step 2: Build full response matrix
    print("\n" + "=" * 60)
    print("Step 2: Building full response matrix ...")
    print("=" * 60)
    print(f"\n--- nectar/all ({n_total:,} prompts) ---")
    payloads["nectar/all"] = build_response_matrix(all_records)
    n_s, n_i = payloads["nectar/all"]["data"].shape
    nan_pct = torch.isnan(payloads["nectar/all"]["data"]).float().mean().item()
    print(f"  Shape: {n_s} x {n_i}, {nan_pct:.1%} missing")

    # Step 3: Build 50K subset
    print(f"\n--- nectar/50k (random {MAX_PROMPTS_SUBSET:,} prompts) ---")
    rng = random.Random(RANDOM_SEED)
    if n_total > MAX_PROMPTS_SUBSET:
        subset_indices = sorted(rng.sample(range(n_total), MAX_PROMPTS_SUBSET))
    else:
        subset_indices = list(range(n_total))
    payloads["nectar/50k"] = build_response_matrix(all_records, subset_indices)
    n_s, n_i = payloads["nectar/50k"]["data"].shape
    nan_pct = torch.isnan(payloads["nectar/50k"]["data"]).float().mean().item()
    print(f"  Shape: {n_s} x {n_i}, {nan_pct:.1%} missing")

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Step 3: Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        filesize_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {n_sub} x {n_items}, {nan_pct:.1%} missing, {filesize_mb:.1f} MB")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )
        print(f"    -> uploaded to {DST_REPO}/{filename}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_DATASET}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for nectar.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}, missing={nan_pct:.1%}")
    print(f"\nModel list:")
    for name in sorted(payloads.values(), key=lambda p: p["data"].shape[1], reverse=True):
        for sid in name["subject_ids"]:
            pass
    # Print models from the full dataset
    full = payloads["nectar/all"]
    for i, sid in enumerate(full["subject_ids"]):
        org = full["subject_metadata"][i]["org"]
        # Count non-NaN entries for this model
        non_nan = (~torch.isnan(full["data"][i])).sum().item()
        print(f"  {sid} ({org}): {non_nan:,} prompts answered")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
