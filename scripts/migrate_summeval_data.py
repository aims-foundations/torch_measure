#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate SummEval data to torch-measure-data.

Downloads the original SummEval annotations (Yale-LILY/SummEval) which
contain per-annotator scores, builds 2D response matrices (models x documents)
for each quality dimension (coherence, consistency, fluency, relevance) and
overall (mean of 4 dimensions) for both expert and crowd annotations, plus a
3D expert tensor for G-theory, and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_summeval_data.py

Source data:
    - Original annotations: Yale-LILY/SummEval on GitHub
      (model_annotations.aligned.jsonl via Google Cloud Storage)
    - HuggingFace mirror: mteb/summeval (aggregated scores only)

SummEval evaluates text summarization quality across 100 CNN/DailyMail source
documents, each summarized by 16 models.  Each summary is annotated by
3 experts and 5 crowd workers on 4 quality dimensions: coherence, consistency,
fluency, and relevance (1-5 Likert scale).

Destination .pt file format (2D, consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # document IDs
        "subject_metadata": list[dict],    # per-model metadata
        "item_metadata": list[dict],       # per-document metadata
    }

3D tensor .pt file format:
    {
        "data": torch.Tensor,             # (n_subjects, n_items, n_experts), float32
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # document IDs
        "expert_ids": list[str],           # expert identifiers
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/sfr-summarization-repo-research/"
    "model_annotations.aligned.jsonl"
)
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_summeval_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_annotations() -> list[dict]:
    """Download the original SummEval per-annotator annotations."""
    print(f"  Downloading from {ANNOTATIONS_URL} ...")
    response = urlopen(ANNOTATIONS_URL)
    data = response.read().decode("utf-8")
    records = [json.loads(line) for line in data.strip().split("\n") if line.strip()]
    print(f"  Downloaded {len(records)} annotation records")
    return records


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_response_matrices(records: list[dict]) -> dict[str, dict]:
    """Build all response matrices from raw annotation records.

    Each record has:
        - model_id: str (model identifier)
        - id: str (document identifier)
        - expert_annotations: list of 3 dicts with {coherence, consistency,
          fluency, relevance}
        - turker_annotations: list of 5 dicts with {coherence, consistency,
          fluency, relevance}

    Returns a dict of dataset_name -> payload dict.
    """
    # Collect unique model IDs and document IDs
    model_ids = sorted(set(r["model_id"] for r in records))
    doc_ids = sorted(set(r["id"] for r in records))

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    doc_to_idx = {d: i for i, d in enumerate(doc_ids)}

    n_models = len(model_ids)
    n_docs = len(doc_ids)
    n_experts = 3
    n_turkers = 5

    print(f"  Models:    {n_models}")
    print(f"  Documents: {n_docs}")
    print(f"  Experts:   {n_experts}")
    print(f"  Turkers:   {n_turkers}")

    # Initialize matrices for expert and crowd annotations
    # Per-dimension: (models x docs) averaged across annotators
    expert_dim = {
        dim: torch.full((n_models, n_docs), float("nan"), dtype=torch.float32)
        for dim in DIMENSIONS
    }
    crowd_dim = {
        dim: torch.full((n_models, n_docs), float("nan"), dtype=torch.float32)
        for dim in DIMENSIONS
    }

    # 3D expert tensor: (models x docs x experts), mean-of-4-dimensions per expert
    expert_3d = torch.full(
        (n_models, n_docs, n_experts), float("nan"), dtype=torch.float32
    )

    # Fill matrices
    for rec in records:
        m_idx = model_to_idx[rec["model_id"]]
        d_idx = doc_to_idx[rec["id"]]

        # Expert annotations
        expert_anns = rec.get("expert_annotations", [])
        for dim in DIMENSIONS:
            scores = [a[dim] for a in expert_anns if dim in a]
            if scores:
                expert_dim[dim][m_idx, d_idx] = sum(scores) / len(scores)

        # Per-expert mean-of-4-dimensions for 3D tensor
        for e_idx, ann in enumerate(expert_anns):
            if e_idx < n_experts:
                dim_scores = [ann[dim] for dim in DIMENSIONS if dim in ann]
                if dim_scores:
                    expert_3d[m_idx, d_idx, e_idx] = sum(dim_scores) / len(dim_scores)

        # Crowd (turker) annotations
        turker_anns = rec.get("turker_annotations", [])
        for dim in DIMENSIONS:
            scores = [a[dim] for a in turker_anns if dim in a]
            if scores:
                crowd_dim[dim][m_idx, d_idx] = sum(scores) / len(scores)

    # Build subject and item metadata
    subject_metadata = [{"model": m} for m in model_ids]
    item_metadata = [{"document": d} for d in doc_ids]
    expert_ids = [f"expert_{i}" for i in range(n_experts)]

    # Build payloads
    payloads: dict[str, dict] = {}

    # Expert per-dimension
    for dim in DIMENSIONS:
        payloads[f"summeval/expert_{dim}"] = {
            "data": expert_dim[dim],
            "subject_ids": model_ids,
            "item_ids": doc_ids,
            "subject_metadata": subject_metadata,
            "item_metadata": item_metadata,
        }

    # Expert overall (mean of 4 dimensions)
    expert_overall = torch.stack(
        [expert_dim[dim] for dim in DIMENSIONS], dim=-1
    ).mean(dim=-1)
    payloads["summeval/expert_overall"] = {
        "data": expert_overall,
        "subject_ids": model_ids,
        "item_ids": doc_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }

    # Crowd per-dimension
    for dim in DIMENSIONS:
        payloads[f"summeval/crowd_{dim}"] = {
            "data": crowd_dim[dim],
            "subject_ids": model_ids,
            "item_ids": doc_ids,
            "subject_metadata": subject_metadata,
            "item_metadata": item_metadata,
        }

    # Crowd overall (mean of 4 dimensions)
    crowd_overall = torch.stack(
        [crowd_dim[dim] for dim in DIMENSIONS], dim=-1
    ).mean(dim=-1)
    payloads["summeval/crowd_overall"] = {
        "data": crowd_overall,
        "subject_ids": model_ids,
        "item_ids": doc_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }

    # Expert 3D tensor
    payloads["summeval/expert_3d"] = {
        "data": expert_3d,
        "subject_ids": model_ids,
        "item_ids": doc_ids,
        "expert_ids": expert_ids,
        "subject_metadata": subject_metadata,
    }

    return payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotations
    print("=" * 60)
    print("Downloading SummEval annotations ...")
    print("=" * 60)
    records = download_annotations()

    # Step 2: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)
    payloads = build_response_matrices(records)

    # Step 3: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)

    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        data = payload["data"]
        shape_str = " x ".join(str(s) for s in data.shape)
        nan_pct = torch.isnan(data).float().mean().item()
        valid_vals = data[~torch.isnan(data)]
        print(
            f"  {filename}: {shape_str}, "
            f"{nan_pct:.1%} missing, "
            f"range [{valid_vals.min():.2f}, {valid_vals.max():.2f}]"
        )

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
    print(f"  Source: {ANNOTATIONS_URL}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for summeval.py registry):")
    for name, payload in sorted(payloads.items()):
        shape_str = " x ".join(str(s) for s in payload["data"].shape)
        print(f"  {name}: {shape_str}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
