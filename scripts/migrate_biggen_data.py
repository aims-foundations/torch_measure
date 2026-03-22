#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate BiGGen-Bench Results to torch-measure-data.

Downloads the ``prometheus-eval/BiGGen-Bench-Results`` dataset from
HuggingFace Hub (``llm_as_a_judge`` split), builds per-judge 2D response
matrices and a combined 3D tensor, and uploads .pt files to HuggingFace Hub.

BiGGen-Bench (NAACL 2025) evaluates frontier LLMs on 695 instances across
70 tasks and 8 capabilities, scored by 5 different LLM judges on a 1-5 scale.
This fully crossed (examinee x item x judge) design is ideal for
Generalizability Theory (G-theory) studies.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_biggen_data.py

Source data: prometheus-eval/BiGGen-Bench-Results on HuggingFace Hub.

The 5 judges and their score columns:
    - gpt4_score           -> GPT-4-1106
    - gpt4_04_turbo_score  -> GPT-4-Turbo-2024-04-09
    - claude_score         -> Claude-3-Opus
    - prometheus_8x7b_score      -> Prometheus-2-8x7B (list of 5, averaged)
    - prometheus_8x7b_bgb_score  -> Prometheus-2-8x7B-BGB (list of 5, averaged)

Per-judge files: biggen/{judge}.pt — 2D (n_subjects x n_items) response matrices
Combined file:   biggen/all_judges.pt — 3D (n_subjects x n_items x n_judges) tensor

Scores are normalized from 1-5 to [0,1] via (score - 1) / 4.

Destination .pt file format (per-judge):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # task/instance identifiers
        "subject_metadata": list[dict],    # per-model metadata
    }

Combined .pt file format:
    {
        "data": torch.Tensor,             # (n_subjects, n_items, n_judges), float32
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # task/instance identifiers
        "judge_ids": list[str],            # judge model names
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "prometheus-eval/BiGGen-Bench-Results"
SRC_SPLIT = "llm_as_a_judge"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_biggen_migration"

# Judge columns -> short names and display names
JUDGES = {
    "gpt4_score": ("gpt4", "GPT-4-1106"),
    "gpt4_04_turbo_score": ("gpt4_turbo", "GPT-4-Turbo-2024-04-09"),
    "claude_score": ("claude", "Claude-3-Opus"),
    "prometheus_8x7b_score": ("prometheus", "Prometheus-2-8x7B"),
    "prometheus_8x7b_bgb_score": ("prometheus_bgb", "Prometheus-2-8x7B-BGB"),
}

# Organization inference from model name
_ORG_MAP: dict[str, str] = {
    "Claude": "Anthropic",
    "claude": "Anthropic",
    "GPT": "OpenAI",
    "gpt": "OpenAI",
    "Gemini": "Google",
    "gemini": "Google",
    "gemma": "Google",
    "Llama": "Meta",
    "Meta-Llama": "Meta",
    "CodeLlama": "Meta",
    "Mistral": "Mistral AI",
    "Mixtral": "Mistral AI",
    "mistral": "Mistral AI",
    "Qwen": "Alibaba",
    "qwen": "Alibaba",
    "Yi": "01.AI",
    "SOLAR": "Upstage",
    "OLMo": "AI2",
    "Phi": "Microsoft",
    "phi": "Microsoft",
    "Orca": "Microsoft",
    "tulu": "AI2",
    "codetulu": "AI2",
    "Starling": "UC Berkeley",
    "openchat": "OpenChat",
    "OpenHermes": "NousResearch",
    "Nous-Hermes": "NousResearch",
    "zephyr": "HuggingFace",
    "aya": "Cohere",
    "c4ai": "Cohere",
    "llemma": "EleutherAI",
    "DeepSeek": "DeepSeek",
    "davinci": "OpenAI",
}


def _infer_org(model_name: str) -> str:
    """Infer organization from model name."""
    for prefix, org in _ORG_MAP.items():
        if prefix.lower() in model_name.lower():
            return org
    return ""


def _build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each subject (model)."""
    import re

    param_re = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")
    metadata = []
    for sid in subject_ids:
        param_count = None
        match = param_re.search(sid)
        if match:
            num = match.group(1)
            if num.endswith(".0"):
                num = num[:-2]
            param_count = f"{num}B"

        metadata.append(
            {
                "model": sid,
                "org": _infer_org(sid),
                "param_count": param_count,
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading BiGGen-Bench-Results ...")
    print("=" * 60)
    from datasets import load_dataset

    ds = load_dataset(SRC_REPO, split=SRC_SPLIT)
    df = ds.to_pandas()
    print(f"  Loaded {len(df)} rows")
    print(f"  Models: {df['model_name'].nunique()}")
    print(f"  Items (by id): {df['id'].nunique()}")
    print(f"  Capabilities: {sorted(df['capability'].unique())}")

    # Step 2: Prepare score columns
    # For prometheus judges, scores are lists of 5 — take the mean.
    print("\n" + "=" * 60)
    print("Preparing score columns ...")
    print("=" * 60)

    def safe_mean(x):
        """Average a list/array of scores, returning NaN for None/empty."""
        if x is None:
            return np.nan
        arr = np.array(x, dtype=float)
        if len(arr) == 0 or np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmean(arr))

    # Create scalar score columns for all judges
    score_cols = {}
    for col, (short_name, display_name) in JUDGES.items():
        if col in ("prometheus_8x7b_score", "prometheus_8x7b_bgb_score"):
            df[f"_score_{short_name}"] = df[col].apply(safe_mean)
        else:
            df[f"_score_{short_name}"] = df[col].astype(float)
        score_cols[short_name] = f"_score_{short_name}"
        n_missing = df[f"_score_{short_name}"].isna().sum()
        print(f"  {short_name} ({display_name}): {n_missing} missing values")

    # Step 3: Build sorted subject/item lists
    subject_ids = sorted(df["model_name"].unique())
    item_ids = sorted(df["id"].unique())
    n_subjects = len(subject_ids)
    n_items = len(item_ids)
    print(f"\n  n_subjects = {n_subjects}, n_items = {n_items}")

    subject_to_idx = {s: i for i, s in enumerate(subject_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    subject_metadata = _build_subject_metadata(subject_ids)

    # Step 4: Build per-judge 2D response matrices and combined 3D tensor
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    judge_short_names = [short for _, (short, _) in JUDGES.items()]
    judge_display_names = [display for _, (_, display) in JUDGES.items()]
    n_judges = len(judge_short_names)

    # Initialize 3D tensor with NaN
    all_data = torch.full((n_subjects, n_items, n_judges), float("nan"), dtype=torch.float32)

    payloads_2d: dict[str, dict] = {}

    for j_idx, (col, (short_name, display_name)) in enumerate(JUDGES.items()):
        print(f"\n--- biggen/{short_name} ---")
        score_col = score_cols[short_name]

        # Build 2D matrix via pivot
        pivot = pd.pivot_table(
            df,
            values=score_col,
            index="model_name",
            columns="id",
            aggfunc="mean",
        )
        # Reindex to ensure consistent ordering
        pivot = pivot.reindex(index=subject_ids, columns=item_ids)

        # Normalize from 1-5 to [0,1]: (score - 1) / 4
        raw_data = torch.tensor(pivot.values, dtype=torch.float32)
        normalized_data = (raw_data - 1.0) / 4.0

        # Clamp to [0,1] for safety (NaN stays NaN)
        normalized_data = normalized_data.clamp(0.0, 1.0)

        n_nan = torch.isnan(normalized_data).sum().item()
        n_total = normalized_data.numel()
        print(f"  Shape: {normalized_data.shape}")
        print(f"  Missing: {n_nan}/{n_total} ({100*n_nan/n_total:.2f}%)")
        print(f"  Range: [{normalized_data[~torch.isnan(normalized_data)].min():.4f}, "
              f"{normalized_data[~torch.isnan(normalized_data)].max():.4f}]")

        payloads_2d[f"biggen/{short_name}"] = {
            "data": normalized_data,
            "subject_ids": subject_ids,
            "item_ids": item_ids,
            "subject_metadata": subject_metadata,
        }

        # Fill 3D tensor
        all_data[:, :, j_idx] = normalized_data

    # Build combined 3D payload
    print(f"\n--- biggen/all_judges ---")
    print(f"  Shape: {all_data.shape}  ({n_subjects} subjects x {n_items} items x {n_judges} judges)")
    n_nan_3d = torch.isnan(all_data).sum().item()
    n_total_3d = all_data.numel()
    print(f"  Missing: {n_nan_3d}/{n_total_3d} ({100*n_nan_3d/n_total_3d:.2f}%)")

    payload_3d = {
        "data": all_data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "judge_ids": judge_display_names,
        "subject_metadata": subject_metadata,
    }

    # Step 5: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)

    # Per-judge 2D files
    for name, payload in sorted(payloads_2d.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_it = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {n_sub} x {n_it}, {nan_pct:.1%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )

    # Combined 3D file
    filename_3d = "biggen/all_judges.pt"
    local_path_3d = TMP_DIR / filename_3d
    local_path_3d.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload_3d, local_path_3d)

    nan_pct_3d = torch.isnan(all_data).float().mean().item()
    print(f"  {filename_3d}: {n_subjects} x {n_items} x {n_judges}, {nan_pct_3d:.1%} missing")

    upload_file(
        path_or_fileobj=str(local_path_3d),
        path_in_repo=filename_3d,
        repo_id=DST_REPO,
        repo_type="dataset",
    )

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO} (split={SRC_SPLIT})")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads_2d) + 1}")
    print(f"\n  Dimensions:")
    print(f"    n_subjects = {n_subjects}")
    print(f"    n_items    = {n_items}")
    print(f"    n_judges   = {n_judges}")
    print(f"\n  Judges: {judge_display_names}")
    print(f"\n  Per-judge files:")
    for name in sorted(payloads_2d):
        print(f"    {name}.pt")
    print(f"    biggen/all_judges.pt (3D combined)")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
