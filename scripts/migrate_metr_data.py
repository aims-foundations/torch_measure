#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate METR eval-analysis-public data to torch-measure-data.

Downloads runs.jsonl from the METR/eval-analysis-public GitHub repo,
pivots into response matrices (agents x tasks), and uploads .pt files
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_metr_data.py

Source data: https://github.com/METR/eval-analysis-public
    reports/time-horizon-1-0/data/raw/runs.jsonl

Each JSONL row represents a single evaluation run with fields:
    - alias: public model name (e.g., "Claude 3 Opus")
    - task_id: unique task identifier (e.g., "acdc_bug/fix_checkpointing")
    - score_binarized: binary pass/fail (0 or 1)
    - score_cont: continuous score (0.0 to 1.0)
    - task_source: "HCAST", "RE-Bench", or "SWAA"
    - human_minutes, task_family, scaffold, model, etc.

Multiple runs per (agent, task) pair are averaged to produce
continuous [0, 1] pass-rate and score matrices.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model aliases
        "item_ids": list[str],             # task_id strings
        "subject_metadata": list[dict],    # structured model metadata
    }
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METR_JSONL_URL = (
    "https://raw.githubusercontent.com/METR/eval-analysis-public/"
    "main/reports/time-horizon-1-0/data/raw/runs.jsonl"
)
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_metr_migration"

# Task source names as they appear in the data.
TASK_SOURCES = ["HCAST", "RE-Bench", "SWAA"]

# Registry-friendly names for per-source splits.
SOURCE_NAME_MAP = {
    "HCAST": "hcast",
    "RE-Bench": "rebench",
    "SWAA": "swaa",
}

# ---------------------------------------------------------------------------
# Subject metadata parsing
# ---------------------------------------------------------------------------

# Known org mappings from model alias.
_ORG_MAP: dict[str, str] = {
    "Claude": "Anthropic",
    "GPT": "OpenAI",
    "o1": "OpenAI",
    "o3": "OpenAI",
    "o4": "OpenAI",
    "Gemini": "Google",
    "DeepSeek": "DeepSeek",
    "Grok": "xAI",
    "Qwen": "Alibaba",
    "Llama": "Meta",
    "Human": "Human",
    "davinci": "OpenAI",
}

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")


def _infer_org(alias: str) -> str:
    """Infer organization from model alias."""
    for prefix, org in _ORG_MAP.items():
        if prefix.lower() in alias.lower():
            return org
    return ""


def _build_subject_metadata(df: pd.DataFrame, subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each subject (agent).

    Uses the first row per alias to extract model, scaffold, etc.
    """
    first_rows = df.drop_duplicates(subset="alias").set_index("alias")
    metadata = []
    for sid in subject_ids:
        row = first_rows.loc[sid] if sid in first_rows.index else None
        model = row["model"] if row is not None else ""
        scaffold = row["scaffold"] if row is not None else ""

        param_count = None
        match = _PARAM_RE.search(sid)
        if match:
            num = match.group(1)
            if num.endswith(".0"):
                num = num[:-2]
            param_count = f"{num}B"

        metadata.append(
            {
                "model": model,
                "org": _infer_org(sid),
                "scaffold": scaffold,
                "param_count": param_count,
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def pivot_to_payload(
    df: pd.DataFrame,
    value_col: str,
) -> dict:
    """Pivot a long-format DataFrame into a response matrix payload.

    Aggregates multiple runs per (alias, task_id) by taking the mean.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: alias, task_id, and *value_col*.
    value_col : str
        Column to pivot (e.g., "score_binarized" or "score_cont").
    """
    pivot = pd.pivot_table(df, values=value_col, index="alias", columns="task_id", aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    subject_ids = list(pivot.index)
    item_ids = list(pivot.columns)
    data = torch.tensor(pivot.values, dtype=torch.float32)

    subject_metadata = _build_subject_metadata(df, subject_ids)

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def download_runs() -> pd.DataFrame:
    """Download runs.jsonl from the METR GitHub repo."""
    print(f"Downloading {METR_JSONL_URL} ...")
    df = pd.read_json(METR_JSONL_URL, lines=True)
    print(f"  Loaded {len(df)} rows, {df['alias'].nunique()} agents, {df['task_id'].nunique()} tasks")
    print(f"  Task sources: {df['task_source'].value_counts().to_dict()}")
    return df


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading METR runs.jsonl ...")
    print("=" * 60)
    df = download_runs()

    payloads: dict[str, dict] = {}

    # Step 2: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    # --- All tasks ---
    print("\n--- metr/all (pass rate) ---")
    payloads["metr/all"] = pivot_to_payload(df, "score_binarized")
    n_s, n_i = payloads["metr/all"]["data"].shape
    print(f"  {n_s} agents x {n_i} tasks")

    print("\n--- metr/all_score (continuous) ---")
    payloads["metr/all_score"] = pivot_to_payload(df, "score_cont")
    n_s, n_i = payloads["metr/all_score"]["data"].shape
    print(f"  {n_s} agents x {n_i} tasks")

    # --- Per task-source splits ---
    for source in TASK_SOURCES:
        short = SOURCE_NAME_MAP[source]
        df_src = df[df["task_source"] == source]
        if df_src.empty:
            print(f"\n  Warning: no rows for task_source={source!r}, skipping")
            continue

        # Pass rate
        name_pr = f"metr/{short}"
        print(f"\n--- {name_pr} (pass rate) ---")
        payloads[name_pr] = pivot_to_payload(df_src, "score_binarized")
        n_s, n_i = payloads[name_pr]["data"].shape
        print(f"  {n_s} agents x {n_i} tasks")

        # Continuous score
        name_sc = f"metr/{short}_score"
        print(f"\n--- {name_sc} (continuous) ---")
        payloads[name_sc] = pivot_to_payload(df_src, "score_cont")
        n_s, n_i = payloads[name_sc]["data"].shape
        print(f"  {n_s} agents x {n_i} tasks")

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
        )

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {METR_JSONL_URL}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for metr.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
