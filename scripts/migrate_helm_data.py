#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate HELM response matrices from stair-lab/reeval to aims-foundation/torch-measure-data.

Downloads the full response matrix from stair-lab/reeval (private),
splits it by benchmark, repackages each into the standardized .pt dict format,
and uploads to aims-foundation/torch-measure-data (dataset repo).

Usage:
    export HF_TOKEN=hf_xxxxx  # token with read access to stair-lab, write access to aims-foundation
    python scripts/migrate_helm_data.py

The .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,        # shape (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],     # model names
        "item_ids": list[str],        # question identifiers (benchmark_name:index)
        "item_contents": list[str],          # question text
        "subject_metadata": list[dict],      # structured model metadata
    }
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "stair-lab/reeval"
DST_REPO = "sangttruong/torch-measure-data"
SRC_FILE = "resmat.pkl"
TMP_DIR = Path("/tmp/torch_measure_migration")

# Mapping from benchmark names in the source data to our registry names.
# Keys = benchmark name in column tuple index [1], Values = filename stem in DST_REPO.
BENCHMARK_MAP = {
    "mmlu": "helm/mmlu",
    "gsm": "helm/gsm8k",
    "truthful_qa": "helm/truthfulqa",
    "boolq": "helm/boolq",
    "commonsense": "helm/commonsense",
    "math": "helm/math",
    "lsat_qa": "helm/lsat_qa",
    "med_qa": "helm/med_qa",
    "legalbench": "helm/legalbench",
    "imdb": "helm/imdb",
    "civil_comments": "helm/civil_comments",
    "bbq": "helm/bbq",
    "air_bench_2024": "helm/air_bench_2024",
    "babi_qa": "helm/babi_qa",
    "raft": "helm/raft",
    "wikifact": "helm/wikifact",
    "synthetic_reasoning": "helm/synthetic_reasoning",
    "entity_matching": "helm/entity_matching",
    "entity_data_imputation": "helm/entity_data_imputation",
    "dyck_language_np=3": "helm/dyck_language",
    "legal_support": "helm/legal_support",
    "thai_exam": "helm/thai_exam",
}


# Org display names for cleaner subject_contents.
_ORG_DISPLAY = {
    "together": "Together AI",
    "aisingapore": "AI Singapore",
    "eleutherai": "EleutherAI",
    "allenai": "Allen AI",
    "openthaigpt": "OpenThaiGPT",
    "openai": "OpenAI",
    "cohere": "Cohere",
    "sambanova": "SambaNova",
    "writer": "Writer",
    "damo": "DAMO Academy",
    "mistralai": "Mistral AI",
    "sail": "SAIL",
    "AlephAlpha": "Aleph Alpha",
    "qwen": "Qwen",
    "scb10x": "SCB 10X",
    "meta": "Meta",
    "mosaicml": "MosaicML",
    "microsoft": "Microsoft",
    "ai21": "AI21 Labs",
    "lmsys": "LMSYS",
    "stanford": "Stanford",
    "google": "Google",
    "tiiuae": "TII UAE",
    "databricks": "Databricks",
    "01-ai": "01.AI",
    "anthropic": "Anthropic",
    "deepseek-ai": "DeepSeek",
    "snowflake": "Snowflake",
    "amazon": "Amazon",
    "upstage": "Upstage",
}

# Regex to extract parameter count from model name (e.g., "7b", "70b", "175b", "530B").
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")

# Keywords indicating an instruction-tuned / chat model.
_INSTRUCT_KEYWORDS = {"instruct", "chat", "it", "sft", "rlhf", "v1.3", "beta"}


def _extract_param_count(name: str) -> str | None:
    """Extract parameter count string from a model name, e.g., '7B', '70B'."""
    match = _PARAM_RE.search(name)
    if match:
        num = match.group(1)
        # Normalize: drop trailing .0
        if num.endswith(".0"):
            num = num[:-2]
        return f"{num}B"
    return None


def _is_instruct_model(name: str) -> bool:
    """Heuristic: is this an instruction-tuned or chat model?"""
    parts = set(name.lower().replace("-", " ").replace("_", " ").split())
    return bool(parts & _INSTRUCT_KEYWORDS)


def build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata dicts for each model.

    Each dict has keys: org, model, param_count, is_instruct.
    """
    metadata = []
    for sid in subject_ids:
        org_raw = sid.split("/")[0] if "/" in sid else ""
        org = _ORG_DISPLAY.get(org_raw, org_raw)
        model = sid.split("/", 1)[1] if "/" in sid else sid
        param_count = _extract_param_count(sid)
        is_instruct = _is_instruct_model(sid)
        metadata.append({
            "org": org,
            "model": model,
            "param_count": param_count,
            "is_instruct": is_instruct,
        })
    return metadata


def load_source() -> pd.DataFrame:
    """Download and load the source response matrix."""
    print(f"Downloading {SRC_FILE} from {SRC_REPO}...")
    path = hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset")
    with open(path, "rb") as f:
        df = pickle.load(f)
    print(f"  Shape: {df.shape} ({df.shape[0]} models x {df.shape[1]} items)")
    print(f"  NaN: {df.isna().sum().sum() / df.size:.1%}")
    return df


def split_by_benchmark(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the full response matrix into per-benchmark DataFrames."""
    benchmarks: dict[str, list[int]] = {}
    for i, col in enumerate(df.columns):
        bench = col[1]  # column tuple: (question_text, benchmark_name, split_name)
        benchmarks.setdefault(bench, []).append(i)

    result = {}
    for bench, col_indices in sorted(benchmarks.items()):
        sub = df.iloc[:, col_indices]
        result[bench] = sub
    return result


def make_payload(df_sub: pd.DataFrame, bench_name: str) -> dict:
    """Convert a per-benchmark DataFrame into a .pt payload dict."""
    data = torch.tensor(df_sub.values, dtype=torch.float32)
    subject_ids = list(df_sub.index.astype(str))
    # Item IDs: use "benchmark:index" format for compact, unique identifiers
    item_ids = [f"{bench_name}:{i}" for i in range(df_sub.shape[1])]
    # Item contents: extract question text from column tuples (question_text, benchmark, split)
    item_contents = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df_sub.columns]
    # Subject metadata: structured model info
    subject_metadata = build_subject_metadata(subject_ids)

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "subject_metadata": subject_metadata,
    }


def make_all_payload(df: pd.DataFrame) -> dict:
    """Create the aggregated 'all' payload from the full DataFrame."""
    data = torch.tensor(df.values, dtype=torch.float32)
    subject_ids = list(df.index.astype(str))
    item_ids = [f"{col[1]}:{i}" for i, col in enumerate(df.columns)]
    item_contents = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
    subject_metadata = build_subject_metadata(subject_ids)

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "subject_metadata": subject_metadata,
    }


def ensure_repo(api: HfApi) -> None:
    """Create the destination repo if it doesn't exist."""
    try:
        api.repo_info(DST_REPO, repo_type="dataset")
        print(f"Destination repo {DST_REPO} already exists.")
    except Exception:
        print(f"Creating dataset repo {DST_REPO}...")
        api.create_repo(DST_REPO, repo_type="dataset", private=False)
        print(f"  Created {DST_REPO}")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # Step 1: Ensure destination repo exists
    ensure_repo(api)

    # Step 2: Download source data
    df = load_source()

    # Step 3: Split by benchmark
    benchmarks = split_by_benchmark(df)
    print(f"\nFound {len(benchmarks)} benchmarks:")
    for bench, sub in sorted(benchmarks.items()):
        name = BENCHMARK_MAP.get(bench, f"helm/{bench}")
        print(f"  {name}: {sub.shape[0]} models x {sub.shape[1]} items")

    # Step 4: Save and upload per-benchmark .pt files
    print("\nSaving and uploading per-benchmark files...")
    for bench, sub in sorted(benchmarks.items()):
        name = BENCHMARK_MAP.get(bench, f"helm/{bench}")
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        payload = make_payload(sub, bench)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {n_sub}x{n_items}, {nan_pct:.0%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )

    # Step 5: Save and upload the "all" aggregate
    print("\nSaving and uploading helm/all.pt...")
    all_payload = make_all_payload(df)
    all_path = TMP_DIR / "helm" / "all.pt"
    all_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_payload, all_path)

    n_sub, n_items = all_payload["data"].shape
    nan_pct = torch.isnan(all_payload["data"]).float().mean().item()
    print(f"  helm/all.pt: {n_sub}x{n_items}, {nan_pct:.0%} missing")

    upload_file(
        path_or_fileobj=str(all_path),
        path_in_repo="helm/all.pt",
        repo_id=DST_REPO,
        repo_type="dataset",
    )

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}/{SRC_FILE}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Benchmarks uploaded: {len(benchmarks)} + 1 aggregate")
    print(f"  Total: {df.shape[0]} models x {df.shape[1]} items")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
