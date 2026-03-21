#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Indeterminacy Experiments data to torch-measure-data.

Downloads multi-judge LLM evaluation data from the
``lguerdan/indeterminacy-experiments`` HuggingFace dataset, builds response
matrices (judges x items), and uploads ``.pt`` files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_indeterminacy_data.py

Source data: https://huggingface.co/datasets/lguerdan/indeterminacy-experiments
    "Validating LLM-as-a-Judge under Rating Indeterminacy" (NeurIPS 2025)

Dataset structure (36 rows = 9 judges x 4 task groups):
    - model_info: {provider, model} identifying the LLM judge
    - resp_table: [200 items, 3 scales, 4 categories, R repetitions]
        Binary indicators for forced-choice / multi-label ratings.
        R = 10 for most judges, 8 for o3-mini.
    - p_judge_hat: Estimated model parameters (theta, F, omega, etc.)

Rating tasks: 7 meaningful (group, scale) combinations with >= 2 categories.
    Group 0: scale 0 (124 items, 2 cats), scale 1 (47 items, 2 cats)
    Group 1: scale 0 (178 items, 2 cats)
    Group 2: scale 0 (55 items, 2 cats), scale 1 (172 items, 2 cats)
    Group 3: scale 0 (118 items, 3 cats), scale 1 (108 items, 2 cats)

Response encoding: P(category 0) averaged across repetitions, giving
continuous [0, 1] values per (judge, item). NaN for items without
responses in a given group.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # LLM judge model names
        "item_ids": list[str],             # item index strings
        "subject_metadata": list[dict],    # structured judge metadata
    }
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_SOURCE = "lguerdan/indeterminacy-experiments"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_indeterminacy_migration"

N_JUDGES = 9
N_ITEMS = 200
N_GROUPS = 4

# Registry-friendly group names.
GROUP_NAMES = {
    0: "group_0",
    1: "group_1",
    2: "group_2",
    3: "group_3",
}

# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_ORG_MAP: dict[str, str] = {
    "claude": "Anthropic",
    "gpt": "OpenAI",
    "o3": "OpenAI",
    "deepseek": "DeepSeek",
    "mistral": "Mistral",
    "llama": "Meta",
}


def _infer_org(model_name: str) -> str:
    for prefix, org in _ORG_MAP.items():
        if prefix in model_name.lower():
            return org
    return ""


def _build_subject_metadata(
    model_names: list[str], providers: list[str]
) -> list[dict]:
    metadata = []
    for name, prov in zip(model_names, providers):
        metadata.append(
            {
                "model": name,
                "org": _infer_org(name),
                "provider": prov,
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Response matrix construction
# ---------------------------------------------------------------------------


def build_group_matrix(ds, group_idx: int) -> np.ndarray:
    """Build a (9 judges x 200 items) response matrix for one task group.

    For each (judge, item), find the first active scale and compute
    P(category 0) = proportion of repetitions where category 0 was selected,
    among reps that have any selection.  Items with no active scale get NaN.
    """
    matrix = np.full((N_JUDGES, N_ITEMS), np.nan)

    for m in range(N_JUDGES):
        row_idx = group_idx * N_JUDGES + m
        rt = np.array(ds[row_idx]["resp_table"])  # [200, 3, 4, reps]

        for i in range(N_ITEMS):
            for s in range(3):
                item_scale = rt[i, s, :, :]  # [4, reps]
                if item_scale.sum() > 0:
                    # Reps that have at least one category selected
                    reps_with_response = item_scale.sum(axis=0) > 0  # [reps]
                    if reps_with_response.sum() > 0:
                        cat0_selected = item_scale[0, reps_with_response]
                        matrix[m, i] = cat0_selected.mean()
                    break  # use first active scale

    return matrix


def build_payload(
    matrix: np.ndarray,
    model_names: list[str],
    providers: list[str],
    item_ids: list[str],
) -> dict:
    """Package a response matrix into a .pt-ready payload."""
    data = torch.tensor(matrix, dtype=torch.float32)
    subject_metadata = _build_subject_metadata(model_names, providers)
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

    # Step 1: Download
    print("=" * 60)
    print("Downloading lguerdan/indeterminacy-experiments ...")
    print("=" * 60)
    ds = load_dataset(HF_SOURCE, split="test")
    print(f"  Loaded {len(ds)} rows")

    # Extract model info (same order repeats 4 times)
    model_names: list[str] = []
    providers: list[str] = []
    for m in range(N_JUDGES):
        mi = ds[m]["model_info"]
        model_names.append(mi["model"])
        providers.append(mi["provider"])
    print(f"  Judges ({N_JUDGES}): {model_names}")

    payloads: dict[str, dict] = {}

    # Step 2: Build per-group response matrices
    print("\n" + "=" * 60)
    print("Building per-group response matrices ...")
    print("=" * 60)

    combined_matrices = []
    combined_item_ids = []

    for g in range(N_GROUPS):
        group_name = GROUP_NAMES[g]
        matrix = build_group_matrix(ds, g)
        item_ids = [f"g{g}_item_{i:03d}" for i in range(N_ITEMS)]

        name = f"indeterminacy/{group_name}"
        print(f"\n--- {name} ---")
        n_nan = np.isnan(matrix).sum()
        total = matrix.size
        print(f"  {N_JUDGES} judges x {N_ITEMS} items")
        print(f"  NaN: {n_nan}/{total} ({n_nan / total:.1%})")
        print(f"  Value range: [{np.nanmin(matrix):.3f}, {np.nanmax(matrix):.3f}]")
        print(f"  Mean: {np.nanmean(matrix):.3f}")

        payloads[name] = build_payload(matrix, model_names, providers, item_ids)
        combined_matrices.append(matrix)
        combined_item_ids.extend(item_ids)

    # Step 3: Build combined response matrix (all groups)
    print("\n" + "=" * 60)
    print("Building combined response matrix ...")
    print("=" * 60)

    combined = np.concatenate(combined_matrices, axis=1)  # 9 x 800
    name = "indeterminacy/all"
    print(f"\n--- {name} ---")
    n_nan = np.isnan(combined).sum()
    total = combined.size
    print(f"  {N_JUDGES} judges x {combined.shape[1]} items")
    print(f"  NaN: {n_nan}/{total} ({n_nan / total:.1%})")
    print(f"  Value range: [{np.nanmin(combined):.3f}, {np.nanmax(combined):.3f}]")
    print(f"  Mean: {np.nanmean(combined):.3f}")

    payloads[name] = build_payload(
        combined, model_names, providers, combined_item_ids
    )

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for payload_name, payload in sorted(payloads.items()):
        filename = f"{payload_name}.pt"
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

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {HF_SOURCE}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for indeterminacy.py registry):")
    for payload_name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {payload_name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
