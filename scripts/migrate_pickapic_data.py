#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Pick-a-Pic v2 pairwise comparisons to torch-measure-data.

Downloads the Pick-a-Pic v2 dataset (no-images variant) from HuggingFace,
reservoir-samples 100K labeled comparisons, converts them to the
torch_measure PairwiseComparisons .pt format, and uploads to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_pickapic_data.py

Source data:
    yuvalkirstain/pickapic_v2_no_images on HuggingFace Hub.
    ~1M human preference judgments over text-to-image model outputs.
    Users compare pairs of generated images for a given text prompt and pick
    the preferred image (or declare a tie).  14 unique text-to-image models.

    Key columns:
        caption: text prompt
        model_0, model_1: names of the two models that generated the images
        label_0, label_1: preference floats (1.0/0.0 or 0.5/0.5 for ties)
        best_image_uid: UID of the preferred image (or "tie")
        user_id: anonymous user identifier
        ranking_id: unique comparison identifier
        has_label: whether the comparison has a preference label

Destination .pt file format (consumed by torch_measure.datasets.load via
PairwiseComparisons):
    {
        "subject_a": torch.LongTensor,       # (n_comparisons,) indices into subject_ids
        "subject_b": torch.LongTensor,       # (n_comparisons,) indices into subject_ids
        "outcome": torch.Tensor,             # (n_comparisons,) 1.0=a wins, 0.0=b wins, 0.5=tie
        "subject_ids": list[str],            # unique model names (sorted)
        "item_ids": list[str],              # unique prompt identifiers (ranking_id)
        "item_contents": list[str],         # text prompts (captions)
        "item_idx": torch.LongTensor,       # (n_comparisons,) indices into item_ids
        "subject_metadata": list[dict],      # per-subject metadata
        "comparison_metadata": list[dict],   # per-comparison metadata
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

SRC_REPO = "yuvalkirstain/pickapic_v2_no_images"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_pickapic_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SAMPLE_SIZE = 100_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pickapic_data():
    """Download Pick-a-Pic v2 (no images) from HuggingFace and return a DataFrame."""
    from datasets import load_dataset

    print("Downloading Pick-a-Pic v2 (no images) from HuggingFace ...")
    ds = load_dataset(SRC_REPO, split="train", token=HF_TOKEN)
    print(f"  Total rows: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")
    return ds


def filter_labeled(ds) -> "Dataset":  # noqa: F821
    """Keep only rows that have a valid preference label."""
    print("\nFiltering to labeled comparisons ...")
    # Keep rows where has_label is True and label_0 is not null
    ds_labeled = ds.filter(lambda x: x["has_label"] is True)
    print(f"  Labeled rows: {len(ds_labeled):,}")
    return ds_labeled


def reservoir_sample(ds, n: int, seed: int) -> "Dataset":  # noqa: F821
    """Reservoir-sample n rows from the dataset."""
    print(f"\nReservoir-sampling {n:,} comparisons (seed={seed}) ...")
    ds_sampled = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    print(f"  Sampled rows: {len(ds_sampled):,}")
    return ds_sampled


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def build_pairwise_payload(ds) -> dict:
    """Convert a HuggingFace Dataset into the .pt pairwise payload format."""
    print("\nBuilding pairwise comparison payload ...")

    # Collect all unique model names
    all_models = set()
    for row in ds:
        all_models.add(row["model_0"])
        all_models.add(row["model_1"])
    subject_ids = sorted(all_models)
    sid_to_idx = {s: i for i, s in enumerate(subject_ids)}
    print(f"  Unique models (subjects): {len(subject_ids)}")
    for s in subject_ids:
        print(f"    {s}")

    n = len(ds)

    # Map models to subject indices
    subject_a_list = []
    subject_b_list = []
    outcome_list = []
    item_ids: list[str] = []
    item_contents: list[str] = []
    item_idx_list: list[int] = []
    comparison_metadata: list[dict] = []

    seen_items: dict[str, int] = {}

    for i, row in enumerate(ds):
        model_a = row["model_0"]
        model_b = row["model_1"]
        label_0 = row["label_0"]
        label_1 = row["label_1"]

        subject_a_list.append(sid_to_idx[model_a])
        subject_b_list.append(sid_to_idx[model_b])

        # Encode outcome: label_0=1.0 means image_0 (model_a) wins
        # label_0=0.0 means image_1 (model_b) wins; 0.5/0.5 = tie
        outcome_list.append(float(label_0))

        # Item tracking (prompt deduplication by ranking_id)
        ranking_id = str(row["ranking_id"])
        caption = row.get("caption", "") or ""

        if ranking_id not in seen_items:
            seen_items[ranking_id] = len(item_ids)
            item_ids.append(ranking_id)
            item_contents.append(caption)
        item_idx_list.append(seen_items[ranking_id])

        # Per-comparison metadata
        comparison_metadata.append({
            "user_id": row.get("user_id", 0),
            "model_0": model_a,
            "model_1": model_b,
            "label_0": label_0,
            "label_1": label_1,
            "best_image_uid": row.get("best_image_uid", ""),
        })

        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,}/{n:,} rows")

    subject_a = torch.tensor(subject_a_list, dtype=torch.long)
    subject_b = torch.tensor(subject_b_list, dtype=torch.long)
    outcome = torch.tensor(outcome_list, dtype=torch.float32)
    item_idx = torch.tensor(item_idx_list, dtype=torch.long)

    # Subject metadata
    subject_metadata = [{"model": s} for s in subject_ids]

    # Summary statistics
    n_comparisons = len(outcome)
    n_a_wins = (outcome == 1.0).sum().item()
    n_b_wins = (outcome == 0.0).sum().item()
    n_ties = (outcome == 0.5).sum().item()
    print(f"\n  Comparisons: {n_comparisons:,}")
    print(f"  Unique prompts: {len(item_ids):,}")
    print(f"  Outcomes: {n_a_wins:,.0f} image_0 wins, {n_b_wins:,.0f} image_1 wins, {n_ties:,.0f} ties")

    return {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "outcome": outcome,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "item_idx": item_idx,
        "subject_metadata": subject_metadata,
        "comparison_metadata": comparison_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Step 1: Downloading Pick-a-Pic v2 ...")
    print("=" * 60)
    ds = load_pickapic_data()

    # Step 2: Filter to labeled rows
    print("\n" + "=" * 60)
    print("Step 2: Filtering to labeled comparisons ...")
    print("=" * 60)
    ds_labeled = filter_labeled(ds)

    # Step 3: Sample
    print("\n" + "=" * 60)
    print("Step 3: Sampling ...")
    print("=" * 60)
    ds_sampled = reservoir_sample(ds_labeled, SAMPLE_SIZE, RANDOM_SEED)

    # Step 4: Build payload
    print("\n" + "=" * 60)
    print("Step 4: Building pairwise payload ...")
    print("=" * 60)
    payload = build_pairwise_payload(ds_sampled)

    # Step 5: Save and upload
    print("\n" + "=" * 60)
    print("Step 5: Saving and uploading ...")
    print("=" * 60)

    filename = "pickapic/sampled_100k.pt"
    local_path = TMP_DIR / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, local_path)

    n_comparisons = len(payload["outcome"])
    n_subjects = len(payload["subject_ids"])
    n_items = len(payload["item_ids"])
    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  {filename}: {n_comparisons:,} comparisons, {n_subjects} models, {n_items:,} prompts, {file_size_mb:.1f} MB")

    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=DST_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}/{filename}")
    print(f"  Models: {n_subjects}")
    print(f"  Comparisons: {n_comparisons:,}")
    print(f"  Unique prompts: {n_items:,}")
    print(f"\n  Update pickapic.py with:")
    print(f"    n_subjects={n_subjects}, n_comparisons={n_comparisons}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
