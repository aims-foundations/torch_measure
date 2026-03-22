#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate BeaverTails (PKU-Alignment) data to torch-measure-data.

Downloads the PKU-Alignment/BeaverTails dataset from HuggingFace Hub,
builds binary safety classification response matrices (harm categories x
QA pairs), and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_beavertails_data.py

Source data: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
    330K+ QA pairs with binary safety annotations across 14 harm categories.
    We use the 330k_test split (33,432 QA pairs).

Response matrices:
    - beavertails/all.pt — all 15 classifiers (14 harm categories + overall
      is_safe) x 33,432 QA pairs.  Binary {0, 1} where 1 = unsafe.
    - beavertails/overall.pt — overall is_safe label only (1 x 33,432).
    - beavertails/<category>.pt — per-category binary label (1 x 33,432).

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # harm category names
        "item_ids": list[str],             # QA pair identifiers
        "subject_metadata": list[dict],    # per-category metadata
        "item_metadata": list[dict],       # per-item metadata (prompt, response)
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

SRC_REPO = "PKU-Alignment/BeaverTails"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_beavertails_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# The 14 harm categories in canonical order.
HARM_CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]

# Registry-friendly short names.
CATEGORY_NAME_MAP = {
    "animal_abuse": "animal_abuse",
    "child_abuse": "child_abuse",
    "controversial_topics,politics": "controversial_topics",
    "discrimination,stereotype,injustice": "discrimination",
    "drug_abuse,weapons,banned_substance": "drug_abuse",
    "financial_crime,property_crime,theft": "financial_crime",
    "hate_speech,offensive_language": "hate_speech",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "non_violent_unethical",
    "privacy_violation": "privacy_violation",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexually_explicit",
    "terrorism,organized_crime": "terrorism",
    "violence,aiding_and_abetting,incitement": "violence",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_beavertails():
    """Load BeaverTails 330k_test split and build response matrices.

    Returns dict mapping name -> payload dict.
    """
    from datasets import load_dataset

    print("Loading PKU-Alignment/BeaverTails (330k_test split) ...")
    ds = load_dataset(SRC_REPO, split="330k_test", token=HF_TOKEN)
    n_items = len(ds)
    print(f"  Loaded {n_items:,} QA pairs")

    # Subject IDs: overall + 14 harm categories
    all_subject_ids = ["overall"] + HARM_CATEGORIES
    n_subjects_all = len(all_subject_ids)

    # Build the combined matrix: (15 x n_items)
    data_all = torch.zeros((n_subjects_all, n_items), dtype=torch.float32)

    # Build item IDs and metadata
    item_ids = []
    item_metadata = []

    for i, example in enumerate(ds):
        # Item ID
        item_ids.append(f"bt_{i:06d}")

        # Item metadata
        item_metadata.append({
            "prompt": example["prompt"],
            "response": example["response"],
        })

        # Overall safety: 1 = unsafe (is_safe=False), 0 = safe (is_safe=True)
        is_unsafe = 0.0 if example["is_safe"] else 1.0
        data_all[0, i] = is_unsafe

        # Per-category labels
        category_dict = example["category"]
        for j, cat in enumerate(HARM_CATEGORIES):
            # category dict maps category name -> bool (True = unsafe)
            data_all[j + 1, i] = 1.0 if category_dict.get(cat, False) else 0.0

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,}/{n_items:,} ...")

    print(f"  Combined matrix shape: {data_all.shape[0]} x {data_all.shape[1]}")

    # Subject metadata
    subject_metadata_all = [{"category": "overall", "type": "aggregate"}]
    for cat in HARM_CATEGORIES:
        subject_metadata_all.append({"category": cat, "type": "harm_category"})

    # Print statistics
    n_unsafe_overall = int(data_all[0].sum().item())
    print(f"\n  Overall unsafe: {n_unsafe_overall:,}/{n_items:,} "
          f"({n_unsafe_overall / n_items * 100:.1f}%)")
    for j, cat in enumerate(HARM_CATEGORIES):
        n_flagged = int(data_all[j + 1].sum().item())
        print(f"  {cat}: {n_flagged:,}/{n_items:,} "
              f"({n_flagged / n_items * 100:.1f}%)")

    # Build payloads
    payloads: dict[str, dict] = {}

    # --- Combined (all 15 classifiers) ---
    payloads["beavertails/all"] = {
        "data": data_all,
        "subject_ids": all_subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata_all,
        "item_metadata": item_metadata,
    }

    # --- Overall only ---
    payloads["beavertails/overall"] = {
        "data": data_all[0:1],  # (1, n_items)
        "subject_ids": ["overall"],
        "item_ids": item_ids,
        "subject_metadata": [{"category": "overall", "type": "aggregate"}],
        "item_metadata": item_metadata,
    }

    # --- Per-category ---
    for j, cat in enumerate(HARM_CATEGORIES):
        short = CATEGORY_NAME_MAP[cat]
        name = f"beavertails/{short}"
        payloads[name] = {
            "data": data_all[j + 1: j + 2],  # (1, n_items)
            "subject_ids": [cat],
            "item_ids": item_ids,
            "subject_metadata": [{"category": cat, "type": "harm_category"}],
            "item_metadata": item_metadata,
        }

    return payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build response matrices
    print("=" * 60)
    print("Building BeaverTails response matrices ...")
    print("=" * 60)
    payloads = load_beavertails()

    # Step 2: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        n_flagged = int(payload["data"].sum().item())
        total = n_sub * n_items
        print(f"  {filename}: {n_sub} x {n_items}, "
              f"{n_flagged:,}/{total:,} flagged ({n_flagged / total * 100:.1f}%)")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for beavertails.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
