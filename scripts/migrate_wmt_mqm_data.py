#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate WMT MQM data to torch-measure-data.

Downloads expert human MQM annotations from RicardoRei/wmt-mqm-human-evaluation
on HuggingFace Hub, pivots into response matrices (systems x segments),
and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_wmt_mqm_data.py

Source data:
    - RicardoRei/wmt-mqm-human-evaluation (WMT 2020-2022 MQM annotations)

WMT MQM evaluates machine translation systems via expert human annotators who
mark fine-grained error categories.  Each segment receives a quality score
(continuous, typically <= 0, where 0 = no errors).  The response matrix for
each (year, language pair) has:
    - Rows: MT systems (translation engines)
    - Columns: source segments
    - Values: mean MQM score across annotators

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # system names
        "item_ids": list[str],             # segment identifiers (hashed src text)
        "subject_metadata": list[dict],    # per-system metadata
        "item_metadata": list[dict],       # per-segment metadata (domain, src, ref)
    }
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "RicardoRei/wmt-mqm-human-evaluation"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_wmt_mqm_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Year + language-pair combinations present in the dataset.
YEAR_LP_COMBOS = [
    (2020, "en-de"),
    (2020, "zh-en"),
    (2021, "en-de"),
    (2021, "en-ru"),
    (2021, "zh-en"),
    (2022, "en-de"),
    (2022, "en-ru"),
    (2022, "zh-en"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _src_hash(src_text: str) -> str:
    """Create a short deterministic ID from source text."""
    return hashlib.sha256(src_text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_wmt_mqm() -> pd.DataFrame:
    """Load the full WMT MQM dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading WMT MQM dataset from HuggingFace ...")
    ds = load_dataset(SRC_REPO, split="train")
    df = ds.to_pandas()
    print(f"  Loaded {len(df)} rows")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Language pairs: {sorted(df['lp'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def build_response_matrix(
    df: pd.DataFrame,
    year: int,
    lp: str,
) -> dict:
    """Build a systems x segments response matrix for one (year, lp) split.

    Parameters
    ----------
    df : pd.DataFrame
        Full WMT MQM dataframe.
    year : int
        WMT year (2020, 2021, or 2022).
    lp : str
        Language pair (e.g. "en-de", "zh-en", "en-ru").

    Returns
    -------
    dict with keys: data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    subset = df[(df["year"] == year) & (df["lp"] == lp)].copy()

    # Create a stable segment ID from source text
    subset["seg_id"] = subset["src"].apply(_src_hash)

    # Aggregate: mean score per (system, segment) across annotators
    agg = subset.groupby(["system", "seg_id"]).agg(
        score=("score", "mean"),
        annotators=("annotators", "first"),
        domain=("domain", "first"),
        src=("src", "first"),
        ref=("ref", "first"),
    ).reset_index()

    # Pivot into matrix
    pivot = agg.pivot(index="system", columns="seg_id", values="score")
    pivot = pivot.sort_index()  # sort systems alphabetically
    pivot = pivot[sorted(pivot.columns)]  # sort segments

    system_names = list(pivot.index)
    seg_ids = list(pivot.columns)

    n_systems = len(system_names)
    n_segments = len(seg_ids)

    # Build tensor
    data = torch.tensor(pivot.values, dtype=torch.float32)

    # Build subject metadata
    subject_metadata = []
    for sys_name in system_names:
        subject_metadata.append({
            "system": sys_name,
            "year": year,
            "lp": lp,
        })

    # Build item metadata (segment-level info from first occurrence)
    seg_info = agg.drop_duplicates(subset="seg_id").set_index("seg_id")
    item_metadata = []
    for seg_id in seg_ids:
        row = seg_info.loc[seg_id] if seg_id in seg_info.index else {}
        item_metadata.append({
            "domain": row.get("domain", "") if isinstance(row, pd.Series) else "",
            "src": row.get("src", "") if isinstance(row, pd.Series) else "",
            "ref": row.get("ref", "") if isinstance(row, pd.Series) else "",
        })

    return {
        "data": data,
        "subject_ids": system_names,
        "item_ids": seg_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load full dataset
    print("=" * 60)
    print("Step 1: Loading WMT MQM dataset")
    print("=" * 60)
    df = load_wmt_mqm()

    # Step 2: Build response matrices per (year, lp)
    print("\n" + "=" * 60)
    print("Step 2: Building response matrices")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    for year, lp in YEAR_LP_COMBOS:
        lp_underscore = lp.replace("-", "_")
        name = f"wmt_mqm/{year}_{lp_underscore}"
        print(f"\n--- {name} ---")

        payload = build_response_matrix(df, year, lp)
        payloads[name] = payload

        n_s, n_i = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {n_s} systems x {n_i} segments, {nan_pct:.1%} missing")

    # Step 3: Save and upload
    print("\n" + "=" * 60)
    print("Step 3: Saving and uploading")
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
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for wmt_mqm.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
