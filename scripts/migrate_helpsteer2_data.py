#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate NVIDIA HelpSteer2 human preference data to torch-measure-data.

Downloads HelpSteer2 from HuggingFace, builds per-attribute response matrices
(2 responses per prompt x prompts), and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_helpsteer2_data.py

Source data: https://huggingface.co/datasets/nvidia/HelpSteer2

HelpSteer2 contains ~21K samples (~10K prompts x 2 responses each), rated by
human annotators on 5 attributes (helpfulness, correctness, coherence,
complexity, verbosity) on a 0-4 integer scale.

Since model identities are NOT available in HelpSteer2 (responses are anonymous),
we build the response matrices as:
    - Rows (subjects): response_0, response_1 (the two anonymous responses per prompt)
    - Columns (items): prompts identified by integer index
    - Values: human ratings normalized to [0, 1] (from original 0-4 scale)

We produce one .pt file per attribute plus an overall (mean across attributes).

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (2, n_prompts), float32
        "subject_ids": list[str],          # ["response_0", "response_1"]
        "item_ids": list[str],             # prompt indices as strings
        "subject_metadata": list[dict],    # metadata per subject
    }
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_helpsteer2_migration"

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
MAX_SCORE = 4  # Original scores are 0-4


# ---------------------------------------------------------------------------
# Data loading and pivot
# ---------------------------------------------------------------------------


def load_helpsteer2() -> pd.DataFrame:
    """Download HelpSteer2 from HuggingFace and return as a DataFrame."""
    print("Downloading nvidia/HelpSteer2 from HuggingFace ...")
    ds_train = load_dataset("nvidia/HelpSteer2", split="train")
    ds_val = load_dataset("nvidia/HelpSteer2", split="validation")

    df_train = pd.DataFrame(ds_train)
    df_val = pd.DataFrame(ds_val)
    df = pd.concat([df_train, df_val], ignore_index=True)

    print(f"  Loaded {len(df)} total rows ({len(df_train)} train + {len(df_val)} validation)")
    print(f"  Unique prompts: {df['prompt'].nunique()}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def build_paired_matrices(df: pd.DataFrame) -> dict[str, dict]:
    """Build per-attribute response matrices from paired responses.

    Groups rows by prompt, assigns response_0/response_1 within each group,
    and pivots into (2 x n_prompts) matrices for each attribute.

    Returns dict mapping attribute name -> payload dict.
    """
    # Assign response index within each prompt group
    df = df.copy()
    df["response_idx"] = df.groupby("prompt").cumcount()

    # Keep only prompts with exactly 2 responses
    prompt_counts = df.groupby("prompt").size()
    valid_prompts = prompt_counts[prompt_counts == 2].index
    df = df[df["prompt"].isin(valid_prompts)].copy()

    # Create a stable prompt ID
    unique_prompts = sorted(df["prompt"].unique())
    prompt_to_id = {p: str(i) for i, p in enumerate(unique_prompts)}
    df["prompt_id"] = df["prompt"].map(prompt_to_id)

    n_prompts = len(unique_prompts)
    print(f"  Paired prompts (exactly 2 responses): {n_prompts}")

    subject_ids = ["response_0", "response_1"]
    item_ids = [str(i) for i in range(n_prompts)]
    subject_metadata = [
        {"response_index": 0, "description": "First response in pair"},
        {"response_index": 1, "description": "Second response in pair"},
    ]

    payloads: dict[str, dict] = {}

    for attr in ATTRIBUTES:
        # Pivot: rows=response_idx, columns=prompt_id, values=attribute score
        pivot = df.pivot_table(
            values=attr,
            index="response_idx",
            columns="prompt_id",
            aggfunc="first",
        )
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        # Normalize to [0, 1]
        data = torch.tensor(pivot.values, dtype=torch.float32) / MAX_SCORE

        payloads[attr] = {
            "data": data,
            "subject_ids": subject_ids,
            "item_ids": list(pivot.columns),
            "subject_metadata": subject_metadata,
        }

    # Overall: mean across all attributes
    all_attrs = torch.stack([payloads[attr]["data"] for attr in ATTRIBUTES])
    overall_data = all_attrs.mean(dim=0)

    payloads["overall"] = {
        "data": overall_data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }

    return payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading HelpSteer2 ...")
    print("=" * 60)
    df = load_helpsteer2()

    # Step 2: Build response matrices
    print("\n" + "=" * 60)
    print("Building per-attribute response matrices ...")
    print("=" * 60)
    payloads = build_paired_matrices(df)

    for name, payload in sorted(payloads.items()):
        n_s, n_i = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  helpsteer2/{name}: {n_s} x {n_i}, {nan_pct:.1%} missing")

    # Step 3: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"helpsteer2/{name}.pt"
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
        print(f"    Uploaded to {DST_REPO}/{filename}")

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: nvidia/HelpSteer2")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for helpsteer2.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  helpsteer2/{name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
