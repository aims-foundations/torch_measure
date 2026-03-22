#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate PKU-SafeRLHF data to torch-measure-data.

Downloads expert comparison pairs from PKU-Alignment/PKU-SafeRLHF on
HuggingFace Hub, converts them into PairwiseComparisons .pt payloads
(separately for helpfulness and safety preferences), and uploads to
HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_pku_saferlhf_data.py

Source data: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
    ~82K expert comparison pairs across 3 Alpaca model variants.  Each sample
    contains a prompt, two responses (response_0, response_1), and dual
    annotations: ``better_response_id`` (helpfulness) and ``safer_response_id``
    (safety/harmlessness).

    Configurations:
        - default: All Alpaca variants combined (73907 train + 8211 test = 82118)
        - alpaca-7b: 27393 train + 3036 test = 30429
        - alpaca2-7b: 25564 train + 2848 test = 28412
        - alpaca3-8b: 20950 train + 2327 test = 23277

Destination .pt file format (consumed by torch_measure.datasets.load via
PairwiseComparisons):
    {
        "subject_a": torch.LongTensor,    # (n_pairs,) index of subject a
        "subject_b": torch.LongTensor,    # (n_pairs,) index of subject b
        "outcome": torch.FloatTensor,     # (n_pairs,) 1.0 = subject_a wins
        "subject_ids": list[str],         # ["response_0", "response_1"]
        "item_ids": list[str],            # per-pair identifiers
        "item_idx": torch.LongTensor,     # (n_pairs,) item index per comparison
        "item_contents": list[str],       # prompt text per pair
        "comparison_metadata": list[dict], # per-pair safety & response metadata
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

HF_SOURCE = "PKU-Alignment/PKU-SafeRLHF"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_pku_saferlhf_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Configurations to process.  "default" combines all three Alpaca variants.
CONFIGS = {
    "default": ("helpfulness", "safety"),
    "alpaca-7b": ("alpaca7b_helpfulness", "alpaca7b_safety"),
    "alpaca2-7b": ("alpaca2_7b_helpfulness", "alpaca2_7b_safety"),
    "alpaca3-8b": ("alpaca3_8b_helpfulness", "alpaca3_8b_safety"),
}

# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def build_pairwise_payload(
    rows: list[dict],
    preference_key: str,
    subset_label: str,
) -> dict:
    """Build a PairwiseComparisons-compatible payload.

    Parameters
    ----------
    rows : list[dict]
        List of dataset rows (each a dict with prompt, responses, annotations).
    preference_key : str
        Either ``"better_response_id"`` (helpfulness) or ``"safer_response_id"``
        (safety).
    subset_label : str
        Label used for item IDs (e.g., ``"helpfulness"``, ``"alpaca7b_safety"``).

    Returns
    -------
    dict
        Payload dict compatible with ``_load_pairwise`` in ``_loader.py``.
    """
    n = len(rows)

    # subject_ids: index 0 = "response_0", index 1 = "response_1"
    subject_ids = ["response_0", "response_1"]

    subject_a = torch.empty(n, dtype=torch.long)
    subject_b = torch.empty(n, dtype=torch.long)
    outcome = torch.empty(n, dtype=torch.float32)
    item_idx = torch.arange(n, dtype=torch.long)

    item_ids: list[str] = []
    item_contents: list[str] = []
    comparison_metadata: list[dict] = []

    for i, row in enumerate(rows):
        preferred = row[preference_key]  # 0 or 1

        # subject_a = preferred response, subject_b = non-preferred response
        # outcome = 1.0 means subject_a wins
        subject_a[i] = preferred
        subject_b[i] = 1 - preferred
        outcome[i] = 1.0

        item_ids.append(f"{subset_label}_{i:06d}")
        item_contents.append(row["prompt"])

        meta = {
            "response_0": row["response_0"],
            "response_1": row["response_1"],
            "prompt_source": row.get("prompt_source", ""),
            "response_0_source": row.get("response_0_source", ""),
            "response_1_source": row.get("response_1_source", ""),
            "is_response_0_safe": row.get("is_response_0_safe"),
            "is_response_1_safe": row.get("is_response_1_safe"),
            "response_0_severity_level": row.get("response_0_severity_level"),
            "response_1_severity_level": row.get("response_1_severity_level"),
            "better_response_id": row.get("better_response_id"),
            "safer_response_id": row.get("safer_response_id"),
        }

        # Include harm categories if present
        if "response_0_harm_category" in row:
            meta["response_0_harm_category"] = row["response_0_harm_category"]
        if "response_1_harm_category" in row:
            meta["response_1_harm_category"] = row["response_1_harm_category"]

        comparison_metadata.append(meta)

    return {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "outcome": outcome,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_idx": item_idx,
        "item_contents": item_contents,
        "comparison_metadata": comparison_metadata,
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def load_config(config_name: str) -> list[dict]:
    """Load all rows (train+test) from a PKU-SafeRLHF configuration.

    Parameters
    ----------
    config_name : str
        HuggingFace config name (e.g., ``"default"``, ``"alpaca-7b"``).

    Returns
    -------
    list[dict]
        Combined train+test rows as list of dicts.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_SOURCE, name=config_name, token=HF_TOKEN if HF_TOKEN else None)

    rows: list[dict] = []
    for split_name in ["train", "test"]:
        if split_name in ds:
            split_ds = ds[split_name]
            print(f"    {split_name}: {len(split_ds)} rows")
            for row in split_ds:
                rows.append(dict(row))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict] = {}

    # Step 1: Process each configuration
    print("=" * 60)
    print("Downloading and processing PKU-SafeRLHF ...")
    print("=" * 60)

    for config_name, (helpfulness_name, safety_name) in CONFIGS.items():
        print(f"\n--- Config: {config_name} ---")
        rows = load_config(config_name)
        print(f"  Total rows: {len(rows)}")

        # Build helpfulness payload
        print(f"\n  Building helpfulness payload ({helpfulness_name}) ...")
        helpfulness_payload = build_pairwise_payload(
            rows, "better_response_id", helpfulness_name,
        )
        payloads[f"pku_saferlhf/{helpfulness_name}"] = helpfulness_payload
        n_pairs = helpfulness_payload["outcome"].shape[0]
        print(f"    {n_pairs} preference pairs, 2 subjects (response_0, response_1)")

        # Build safety payload
        print(f"\n  Building safety payload ({safety_name}) ...")
        safety_payload = build_pairwise_payload(
            rows, "safer_response_id", safety_name,
        )
        payloads[f"pku_saferlhf/{safety_name}"] = safety_payload
        n_pairs = safety_payload["outcome"].shape[0]
        print(f"    {n_pairs} preference pairs, 2 subjects (response_0, response_1)")

    # Step 2: Save and upload
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
            token=HF_TOKEN,
        )

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {HF_SOURCE}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for pku_saferlhf.py registry):")
    for name, payload in sorted(payloads.items()):
        n_pairs = payload["outcome"].shape[0]
        print(f"  {name}: n_subjects=2, n_items={n_pairs}, n_comparisons={n_pairs}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
