#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Arena-Hard-Auto v0.1 data to torch-measure-data.

Downloads judgment data from lmarena-ai/arena-hard-auto on HuggingFace,
builds a response matrix (models x prompts), and uploads .pt files
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_arena_hard_data.py

Source data: https://huggingface.co/datasets/lmarena-ai/arena-hard-auto
    data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview/*.jsonl

Arena-Hard-Auto (LMSYS) evaluates models by having an LLM judge
(GPT-4-Turbo / gpt-4-1106-preview) compare each model's response against
a fixed baseline (GPT-4-0314) on 500 challenging prompts.

Each judgment row contains two "games" (swapped order):
    Game 0: A = model, B = baseline
    Game 1: A = baseline, B = model

Scores are 5-level: A>>B, A>B, A=B (or A~=B), B>A, B>>A.
We map these to numeric scores from the model's perspective:
    Game 0: A>>B=1.0, A>B=0.75, A=B=0.5, B>A=0.25, B>>A=0.0
    Game 1: A>>B=0.0, A>B=0.25, A=B=0.5, B>A=0.75, B>>A=1.0
Then average the two games per (model, prompt) pair.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # prompt/question identifiers
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "lmarena-ai/arena-hard-auto"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_arena_hard_migration"

# We use the gpt-4-1106-preview judge — the primary/default judge for v0.1.
JUDGE = "gpt-4-1106-preview"
VERSION = "arena-hard-v0.1"

# Score mapping from the MODEL's perspective.
# Game 0: A = model, B = baseline.
SCORE_MAP_GAME0 = {
    "A>>B": 1.0,
    "A>B": 0.75,
    "A=B": 0.5,
    "B>A": 0.25,
    "B>>A": 0.0,
}

# Game 1: A = baseline, B = model (inverted).
SCORE_MAP_GAME1 = {
    "A>>B": 0.0,
    "A>B": 0.25,
    "A=B": 0.5,
    "B>A": 0.75,
    "B>>A": 1.0,
}

# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_ORG_MAP: dict[str, str] = {
    "claude": "Anthropic",
    "gpt": "OpenAI",
    "o1": "OpenAI",
    "o3": "OpenAI",
    "o4": "OpenAI",
    "gemini": "Google",
    "gemma": "Google",
    "deepseek": "DeepSeek",
    "grok": "xAI",
    "qwen": "Alibaba",
    "llama": "Meta",
    "mistral": "Mistral",
    "mixtral": "Mistral",
    "command": "Cohere",
    "dbrx": "Databricks",
    "phi": "Microsoft",
    "yi": "01.AI",
    "starling": "Berkeley",
    "zephyr": "HuggingFace",
    "wizardlm": "Microsoft",
    "athene": "NexusFlow",
    "internlm": "Shanghai AI Lab",
    "tulu": "AI2",
    "snowflake": "Snowflake",
    "davinci": "OpenAI",
    "chatgpt": "OpenAI",
}

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")


def _infer_org(model_name: str) -> str:
    """Infer organization from model name."""
    name_lower = model_name.lower()
    for prefix, org in _ORG_MAP.items():
        if prefix in name_lower:
            return org
    return ""


def _build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build metadata for each model."""
    metadata = []
    for sid in subject_ids:
        param_count = None
        match = _PARAM_RE.search(sid)
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
# Data download and parsing
# ---------------------------------------------------------------------------


def clone_data() -> Path:
    """Clone the HF dataset repo and return the judgment directory."""
    clone_dir = TMP_DIR / "arena-hard-data"
    judgment_dir = clone_dir / "data" / VERSION / "model_judgment" / JUDGE

    if judgment_dir.exists():
        print(f"  Using cached clone at {clone_dir}")
        return judgment_dir

    print(f"  Cloning {SRC_REPO} to {clone_dir} ...")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"https://huggingface.co/datasets/{SRC_REPO}", str(clone_dir)],
        check=True,
        capture_output=True,
    )
    return judgment_dir


def parse_judgments(judgment_dir: Path) -> pd.DataFrame:
    """Parse all judgment JSONL files into a long-format DataFrame.

    Returns DataFrame with columns: model, question_id, score
    where score is the average of game 0 and game 1 mapped scores.
    """
    records = []
    jsonl_files = sorted(judgment_dir.glob("*.jsonl"))
    print(f"  Found {len(jsonl_files)} model judgment files")

    for fpath in jsonl_files:
        model_name = fpath.stem  # e.g., "claude-3-opus-20240229"
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                question_id = row["uid"]
                games = row["games"]

                if len(games) < 2:
                    continue

                score0_label = games[0]["score"]
                score1_label = games[1]["score"]

                # Skip if either game has a None score (judge failure)
                if score0_label is None or score1_label is None:
                    continue

                s0 = SCORE_MAP_GAME0.get(score0_label)
                s1 = SCORE_MAP_GAME1.get(score1_label)

                if s0 is None or s1 is None:
                    print(f"    Warning: unknown score label in {fpath.name}: "
                          f"game0={score0_label!r}, game1={score1_label!r}")
                    continue

                avg_score = (s0 + s1) / 2.0

                records.append(
                    {
                        "model": model_name,
                        "question_id": question_id,
                        "score": avg_score,
                    }
                )

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} (model, question) judgments")
    print(f"  Models: {df['model'].nunique()}, Questions: {df['question_id'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Pivot & payload
# ---------------------------------------------------------------------------


def build_payload(df: pd.DataFrame) -> dict:
    """Build a response matrix payload from long-format judgments."""
    pivot = pd.pivot_table(
        df, values="score", index="model", columns="question_id", aggfunc="mean"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("Downloading Arena-Hard-Auto data ...")
    print("=" * 60)
    judgment_dir = clone_data()

    # Step 2: Parse judgments
    print("\n" + "=" * 60)
    print("Parsing judgment files ...")
    print("=" * 60)
    df = parse_judgments(judgment_dir)

    # Step 3: Build response matrix
    print("\n" + "=" * 60)
    print("Building response matrix ...")
    print("=" * 60)
    payload = build_payload(df)
    n_sub, n_items = payload["data"].shape
    nan_pct = torch.isnan(payload["data"]).float().mean().item()
    print(f"  arena_hard/judgments: {n_sub} models x {n_items} prompts, {nan_pct:.1%} missing")

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)

    filename = "arena_hard/judgments.pt"
    local_path = TMP_DIR / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, local_path)
    print(f"  Saved to {local_path}")

    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=DST_REPO,
        repo_type="dataset",
    )
    print(f"  Uploaded to {DST_REPO}/{filename}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print(f"  Source: {SRC_REPO} (judge={JUDGE})")
    print(f"  Destination: {DST_REPO}/{filename}")
    print(f"  Dimensions: n_subjects={n_sub}, n_items={n_items}")
    print(f"  Missing: {nan_pct:.1%}")
    print(f"\n  Subject IDs (first 10): {payload['subject_ids'][:10]}")
    data_flat = payload["data"][~torch.isnan(payload["data"])]
    print(f"\n  Score stats: mean={data_flat.mean():.4f}, "
          f"min={data_flat.min():.4f}, max={data_flat.max():.4f}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")

    # Print dimensions for registry
    print(f"\n--- For arena_hard.py registry ---")
    print(f"  n_subjects={n_sub}")
    print(f"  n_items={n_items}")


if __name__ == "__main__":
    main()
