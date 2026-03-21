#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate MT-Bench GPT-4 judgment data to torch-measure-data.

Downloads GPT-4 single-answer judgments from the FastChat HuggingFace Space,
pivots into response matrices (models x questions), and uploads .pt files
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_mtbench_data.py

Source data:
    https://huggingface.co/spaces/lmsys/mt-bench
    GPT-4 single-answer grading on 80 multi-turn questions across 8 categories.
    Scores are 1-10 (with -1 for errors).

MT-Bench evaluates LLMs on multi-turn conversation quality. Each model is
scored by GPT-4 on two turns per question. This script produces:

1. **mtbench/score** — Continuous scores normalized to [0, 1] (divide by 10),
   averaging turn-1 and turn-2 scores per (model, question).
2. **mtbench/score_t1** — Turn-1 only scores, normalized to [0, 1].
3. **mtbench/score_t2** — Turn-2 only scores, normalized to [0, 1].
4. **mtbench/binary** — Binary version: score >= 0.5 (i.e. raw >= 5) -> 1, else 0.
5. **mtbench/binary_t1** — Turn-1 only binary.
6. **mtbench/binary_t2** — Turn-2 only binary.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # question identifiers (e.g., "q81_writing")
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JUDGMENT_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/model_judgment/gpt-4_single.jsonl"
)
QUESTION_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/question.jsonl"
)
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_mtbench_migration"

# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_ORG_MAP: dict[str, str] = {
    "gpt": "OpenAI",
    "claude": "Anthropic",
    "llama": "Meta",
    "vicuna": "LMSYS",
    "alpaca": "Stanford",
    "palm": "Google",
    "falcon": "TII",
    "mpt": "MosaicML",
    "dolly": "Databricks",
    "chatglm": "Tsinghua",
    "koala": "UC Berkeley",
    "guanaco": "UW",
    "tulu": "AI2",
    "wizardlm": "Microsoft",
    "baize": "UCSD",
    "oasst": "LAION",
    "stablelm": "Stability AI",
    "rwkv": "RWKV",
    "fastchat": "LMSYS",
    "nous": "NousResearch",
    "h2ogpt": "H2O.ai",
}


def _infer_org(model_name: str) -> str:
    """Infer organization from model name."""
    lower = model_name.lower()
    for prefix, org in _ORG_MAP.items():
        if prefix in lower:
            return org
    return ""


def _build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each model."""
    metadata = []
    for sid in subject_ids:
        metadata.append(
            {
                "model": sid,
                "org": _infer_org(sid),
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_jsonl(url: str) -> list[dict]:
    """Download a JSONL file and return list of dicts."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req)
    data = response.read().decode("utf-8")
    return [json.loads(line) for line in data.strip().split("\n")]


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def pivot_to_payload(
    df: pd.DataFrame,
    question_meta: dict[int, str],
) -> dict:
    """Pivot a long-format DataFrame into a response matrix payload.

    Expects columns: model, question_id, score.
    Aggregates multiple rows (e.g., turn-1 and turn-2) by mean.
    Normalizes scores to [0, 1] by dividing by 10.
    Rows with score == -1 are treated as missing (NaN).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: model, question_id, score.
    question_meta : dict
        Maps question_id -> category string.
    """
    # Replace -1 with NaN (error scores)
    df = df.copy()
    df.loc[df["score"] < 0, "score"] = float("nan")

    # Normalize to [0, 1]
    df["score"] = df["score"] / 10.0

    # Create item label: "q{id}_{category}"
    df["item_id"] = df["question_id"].map(
        lambda qid: f"q{qid}_{question_meta.get(qid, 'unknown')}"
    )

    # Pivot: aggregate by mean (handles multiple turns if both present)
    pivot = pd.pivot_table(
        df, values="score", index="model", columns="item_id", aggfunc="mean"
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    subject_ids = list(pivot.index)
    item_ids = list(pivot.columns)
    data = torch.tensor(pivot.values, dtype=torch.float32)

    subject_metadata = _build_subject_metadata(subject_ids)

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


def binarize_payload(payload: dict, threshold: float = 0.5) -> dict:
    """Create a binary version of a payload: score >= threshold -> 1, else 0.

    NaN values remain NaN.
    """
    data = payload["data"].clone()
    mask = ~torch.isnan(data)
    binary = torch.full_like(data, float("nan"))
    binary[mask] = (data[mask] >= threshold).float()

    return {
        "data": binary,
        "subject_ids": list(payload["subject_ids"]),
        "item_ids": list(payload["item_ids"]),
        "subject_metadata": list(payload["subject_metadata"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading MT-Bench data ...")
    print("=" * 60)

    judgments = _download_jsonl(JUDGMENT_URL)
    questions = _download_jsonl(QUESTION_URL)

    # Build question metadata map: question_id -> category
    question_meta = {q["question_id"]: q["category"] for q in questions}

    df = pd.DataFrame(judgments)
    print(f"  Loaded {len(df)} judgment rows")
    print(f"  {df['model'].nunique()} models, {df['question_id'].nunique()} questions")
    print(f"  Turns: {sorted(df['turn'].unique())}")
    print(f"  Score range: {df['score'].min()} to {df['score'].max()}")

    payloads: dict[str, dict] = {}

    # Step 2: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    # --- Both turns averaged ---
    print("\n--- mtbench/score (both turns, continuous) ---")
    payloads["mtbench/score"] = pivot_to_payload(df, question_meta)
    n_s, n_i = payloads["mtbench/score"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

    print("\n--- mtbench/binary (both turns, binary >= 0.5) ---")
    payloads["mtbench/binary"] = binarize_payload(payloads["mtbench/score"])
    n_s, n_i = payloads["mtbench/binary"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

    # --- Turn 1 only ---
    df_t1 = df[df["turn"] == 1]
    print("\n--- mtbench/score_t1 (turn 1 only, continuous) ---")
    payloads["mtbench/score_t1"] = pivot_to_payload(df_t1, question_meta)
    n_s, n_i = payloads["mtbench/score_t1"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

    print("\n--- mtbench/binary_t1 (turn 1 only, binary >= 0.5) ---")
    payloads["mtbench/binary_t1"] = binarize_payload(payloads["mtbench/score_t1"])
    n_s, n_i = payloads["mtbench/binary_t1"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

    # --- Turn 2 only ---
    df_t2 = df[df["turn"] == 2]
    print("\n--- mtbench/score_t2 (turn 2 only, continuous) ---")
    payloads["mtbench/score_t2"] = pivot_to_payload(df_t2, question_meta)
    n_s, n_i = payloads["mtbench/score_t2"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

    print("\n--- mtbench/binary_t2 (turn 2 only, binary >= 0.5) ---")
    payloads["mtbench/binary_t2"] = binarize_payload(payloads["mtbench/score_t2"])
    n_s, n_i = payloads["mtbench/binary_t2"]["data"].shape
    print(f"  {n_s} models x {n_i} questions")

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
    print(f"  Source: {JUDGMENT_URL}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for mtbench.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
