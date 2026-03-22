#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Prometheus evaluation data to torch-measure-data.

Downloads the Feedback-Collection and Preference-Collection datasets from
prometheus-eval on HuggingFace Hub, pivots into response matrices
(criteria x instances), and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_prometheus_data.py

Source data:
    - Feedback-Collection: prometheus-eval/Feedback-Collection (100K instances)
      GPT-4 evaluates model responses on 996 custom rubric criteria, scored 1-5.
    - Preference-Collection: prometheus-eval/Preference-Collection (200K instances)
      GPT-4 pairwise preferences between two responses on custom rubrics.

Response matrix structure:
    - Feedback: Rows = criteria, Columns = instance indices, Values = [0,1] normalized
    - Preference: Rows = criteria, Columns = instance indices, Values = binary {0,1}

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # criteria names
        "item_ids": list[str],             # instance index strings
        "subject_metadata": list[dict],    # per-criteria metadata
        "item_metadata": list[dict],       # per-instance metadata
    }
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_FEEDBACK_REPO = "prometheus-eval/Feedback-Collection"
SRC_PREFERENCE_REPO = "prometheus-eval/Preference-Collection"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_prometheus_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_feedback_collection() -> list[dict]:
    """Load Feedback-Collection from HuggingFace and return list of records."""
    from datasets import load_dataset

    ds = load_dataset(SRC_FEEDBACK_REPO, split="train", token=HF_TOKEN)
    records = []
    criteria_instance_count: dict[str, int] = defaultdict(int)

    for item in ds:
        criteria = item.get("orig_criteria", "")
        score_str = item.get("orig_score", "")
        instruction = item.get("orig_instruction", "")

        if not criteria or not score_str:
            continue

        try:
            score = int(score_str)
        except (ValueError, TypeError):
            continue

        if score < 1 or score > 5:
            continue

        instance_idx = criteria_instance_count[criteria]
        criteria_instance_count[criteria] += 1

        records.append({
            "criteria": criteria,
            "instance_idx": instance_idx,
            "score": score,
            "instruction": instruction or "",
        })

    return records


def load_preference_collection() -> list[dict]:
    """Load Preference-Collection from HuggingFace and return list of records."""
    from datasets import load_dataset

    ds = load_dataset(SRC_PREFERENCE_REPO, split="train", token=HF_TOKEN)
    records = []
    criteria_instance_count: dict[str, int] = defaultdict(int)

    for item in ds:
        criteria = item.get("orig_criteria", "")
        preference = item.get("orig_preference", "")
        score_a_str = item.get("orig_score_A", "")
        score_b_str = item.get("orig_score_B", "")
        instruction = item.get("orig_instruction", "")

        if not criteria or not preference:
            continue

        if preference == "B":
            binary_pref = 1.0
        elif preference == "A":
            binary_pref = 0.0
        else:
            continue

        try:
            score_a = int(score_a_str) if score_a_str else None
            score_b = int(score_b_str) if score_b_str else None
        except (ValueError, TypeError):
            score_a = None
            score_b = None

        instance_idx = criteria_instance_count[criteria]
        criteria_instance_count[criteria] += 1

        records.append({
            "criteria": criteria,
            "instance_idx": instance_idx,
            "preference": binary_pref,
            "score_a": score_a,
            "score_b": score_b,
            "instruction": instruction or "",
        })

    return records


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def build_feedback_payload(records: list[dict]) -> dict:
    """Build criteria x instances response matrix from Feedback-Collection.

    Values are normalized to [0, 1] via (score - 1) / 4.
    """
    # Determine unique criteria and max instance count
    criteria_set: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in records:
        criteria_set[rec["criteria"]][rec["instance_idx"]] = rec

    criteria_names = sorted(criteria_set.keys())
    max_instances = max(len(v) for v in criteria_set.values())

    n_criteria = len(criteria_names)
    n_items = max_instances

    # Build matrix
    data = torch.full((n_criteria, n_items), float("nan"), dtype=torch.float32)
    criteria_to_idx = {c: i for i, c in enumerate(criteria_names)}

    for rec in records:
        ci = criteria_to_idx[rec["criteria"]]
        ii = rec["instance_idx"]
        if ii < n_items:
            data[ci, ii] = (rec["score"] - 1) / 4.0

    # Build metadata
    subject_ids = criteria_names
    item_ids = [str(i) for i in range(n_items)]

    subject_metadata = []
    for criteria in criteria_names:
        n_instances = len(criteria_set[criteria])
        scores = [criteria_set[criteria][k]["score"] for k in criteria_set[criteria]]
        subject_metadata.append({
            "criteria": criteria,
            "n_instances": n_instances,
            "mean_score": sum(scores) / len(scores) if scores else 0,
        })

    item_metadata = [{"instance_idx": i} for i in range(n_items)]

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


def build_preference_payload(records: list[dict]) -> dict:
    """Build criteria x instances response matrix from Preference-Collection.

    Values are binary: 1.0 = B preferred, 0.0 = A preferred.
    """
    criteria_set: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in records:
        criteria_set[rec["criteria"]][rec["instance_idx"]] = rec

    criteria_names = sorted(criteria_set.keys())
    max_instances = max(len(v) for v in criteria_set.values())

    n_criteria = len(criteria_names)
    n_items = max_instances

    # Build matrix
    data = torch.full((n_criteria, n_items), float("nan"), dtype=torch.float32)
    criteria_to_idx = {c: i for i, c in enumerate(criteria_names)}

    for rec in records:
        ci = criteria_to_idx[rec["criteria"]]
        ii = rec["instance_idx"]
        if ii < n_items:
            data[ci, ii] = rec["preference"]

    # Build metadata
    subject_ids = criteria_names
    item_ids = [str(i) for i in range(n_items)]

    subject_metadata = []
    for criteria in criteria_names:
        n_instances = len(criteria_set[criteria])
        prefs = [criteria_set[criteria][k]["preference"] for k in criteria_set[criteria]]
        subject_metadata.append({
            "criteria": criteria,
            "n_instances": n_instances,
            "pref_b_rate": sum(prefs) / len(prefs) if prefs else 0,
        })

    item_metadata = [{"instance_idx": i} for i in range(n_items)]

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict] = {}

    # Step 1: Load and build Feedback-Collection
    print("=" * 60)
    print("Loading Feedback-Collection from prometheus-eval ...")
    print("=" * 60)

    feedback_records = load_feedback_collection()
    print(f"  Loaded {len(feedback_records):,} feedback records")

    print("\nBuilding feedback response matrix ...")
    payloads["prometheus/feedback"] = build_feedback_payload(feedback_records)
    n_s, n_i = payloads["prometheus/feedback"]["data"].shape
    print(f"  {n_s} criteria x {n_i} instances")

    # Step 2: Load and build Preference-Collection
    print("\n" + "=" * 60)
    print("Loading Preference-Collection from prometheus-eval ...")
    print("=" * 60)

    preference_records = load_preference_collection()
    print(f"  Loaded {len(preference_records):,} preference records")

    print("\nBuilding preference response matrix ...")
    payloads["prometheus/preference"] = build_preference_payload(preference_records)
    n_s, n_i = payloads["prometheus/preference"]["data"].shape
    print(f"  {n_s} criteria x {n_i} instances")

    # Step 3: Save and upload
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

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source (feedback):   {SRC_FEEDBACK_REPO}")
    print(f"  Source (preference): {SRC_PREFERENCE_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for prometheus.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
