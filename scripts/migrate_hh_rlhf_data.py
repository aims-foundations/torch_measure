#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Anthropic HH-RLHF data to torch-measure-data.

Downloads human preference pairs from the Anthropic/hh-rlhf HuggingFace
dataset, converts them into PairwiseComparisons .pt payloads, and uploads
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_hh_rlhf_data.py

Source data: https://huggingface.co/datasets/Anthropic/hh-rlhf
    ~161K chosen/rejected conversation pairs across "helpful" and "harmless"
    subsets.  Each sample contains a chosen conversation (preferred by a human)
    and a rejected conversation.

    The HF repo contains separate JSONL.gz files per subset:
        harmless-base/{train,test}.jsonl.gz
        helpful-base/{train,test}.jsonl.gz
        helpful-online/{train,test}.jsonl.gz
        helpful-rejection-sampled/{train,test}.jsonl.gz

Destination .pt file format (consumed by torch_measure.datasets.load via
PairwiseComparisons):
    {
        "subject_a": torch.LongTensor,    # (n_pairs,) index 0 = "chosen"
        "subject_b": torch.LongTensor,    # (n_pairs,) index 1 = "rejected"
        "outcome": torch.FloatTensor,     # (n_pairs,) all 1.0 (chosen wins)
        "subject_ids": list[str],         # ["chosen", "rejected"]
        "item_ids": list[str],            # per-pair identifiers
        "item_contents": list[str],       # chosen conversation text
        "comparison_metadata": list[dict], # {"rejected": str} per pair
    }
"""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_SOURCE = "Anthropic/hh-rlhf"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_hh_rlhf_migration"

# Mapping from subset name to the list of JSONL.gz file prefixes in the HF repo.
SUBSET_FILES = {
    "helpful": [
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ],
    "harmless": [
        "harmless-base",
    ],
}

# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def build_pairwise_payload(
    chosen: list[str],
    rejected: list[str],
    subset_name: str,
) -> dict:
    """Build a PairwiseComparisons-compatible payload from chosen/rejected pairs.

    Parameters
    ----------
    chosen : list[str]
        Chosen (preferred) conversations.
    rejected : list[str]
        Rejected conversations.
    subset_name : str
        Name of the subset ("helpful" or "harmless").

    Returns
    -------
    dict
        Payload dict compatible with ``_load_pairwise`` in ``_loader.py``.
    """
    n = len(chosen)
    assert len(rejected) == n, f"Mismatch: {len(chosen)} chosen vs {len(rejected)} rejected"

    # subject_ids: index 0 = "chosen", index 1 = "rejected"
    subject_ids = ["chosen", "rejected"]

    # Every comparison: chosen (index 0) vs rejected (index 1), outcome = 1.0
    subject_a = torch.zeros(n, dtype=torch.long)   # chosen
    subject_b = torch.ones(n, dtype=torch.long)     # rejected
    outcome = torch.ones(n, dtype=torch.float32)     # chosen always wins

    # Item IDs: sequential identifiers
    item_ids = [f"{subset_name}_{i:06d}" for i in range(n)]

    # Item index: each comparison maps to its own unique item
    item_idx = torch.arange(n, dtype=torch.long)

    # Store chosen text as item_contents, rejected text in comparison_metadata
    item_contents = chosen
    comparison_metadata = [{"rejected": r} for r in rejected]

    return {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "outcome": outcome,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "item_idx": item_idx,
        "comparison_metadata": comparison_metadata,
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def load_jsonl_gz(path: str) -> list[dict]:
    """Load a gzipped JSONL file and return list of dicts."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_and_split() -> dict[str, tuple[list[str], list[str]]]:
    """Download Anthropic HH-RLHF JSONL files and split into helpful/harmless.

    Downloads individual JSONL.gz files from the HF repo using
    ``hf_hub_download``, then groups by subset.

    Returns
    -------
    dict[str, tuple[list[str], list[str]]]
        Mapping from subset name to (chosen_list, rejected_list).
    """
    result: dict[str, tuple[list[str], list[str]]] = {}

    for subset_name, file_prefixes in SUBSET_FILES.items():
        print(f"\n  Loading subset: {subset_name} ...")
        all_chosen: list[str] = []
        all_rejected: list[str] = []

        for prefix in file_prefixes:
            for split in ["train", "test"]:
                filename = f"{prefix}/{split}.jsonl.gz"
                print(f"    Downloading {filename} ...")

                local_path = hf_hub_download(
                    repo_id=HF_SOURCE,
                    filename=filename,
                    repo_type="dataset",
                )

                rows = load_jsonl_gz(local_path)
                n_before = len(all_chosen)
                for row in rows:
                    all_chosen.append(row["chosen"])
                    all_rejected.append(row["rejected"])
                n_added = len(all_chosen) - n_before
                print(f"      {split}: {n_added} pairs")

        print(f"    Total {subset_name}: {len(all_chosen)} pairs")
        result[subset_name] = (all_chosen, all_rejected)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading Anthropic HH-RLHF ...")
    print("=" * 60)
    subsets = download_and_split()

    payloads: dict[str, dict] = {}

    # Step 2: Build pairwise payloads
    print("\n" + "=" * 60)
    print("Building pairwise comparison payloads ...")
    print("=" * 60)

    for subset_name, (chosen, rejected) in subsets.items():
        print(f"\n--- hh_rlhf/{subset_name} ---")
        payload = build_pairwise_payload(chosen, rejected, subset_name)
        payloads[f"hh_rlhf/{subset_name}"] = payload
        n_pairs = payload["outcome"].shape[0]
        print(f"  {n_pairs} preference pairs, 2 subjects (chosen, rejected)")

    # Step 3: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_pairs = payload["outcome"].shape[0]
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {n_pairs} pairs, {file_size_mb:.1f} MB")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {HF_SOURCE}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for hh_rlhf.py registry):")
    for name, payload in sorted(payloads.items()):
        n_pairs = payload["outcome"].shape[0]
        print(f"  {name}: n_subjects=2, n_items={n_pairs}, n_comparisons={n_pairs}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
