#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate OpenLLM Leaderboard response matrices from stair-lab to torch-measure-data.

Downloads long-format data from stair-lab/zero_shot_open_llm_leaderboard,
pivots into per-benchmark response matrices, and uploads to HuggingFace.

Usage:
    export HF_TOKEN=hf_xxxxx
    python scripts/migrate_openllm_data.py

Two-pass streaming approach — never holds more than one shard in memory:
  Pass 1: Collect unique (model_name, benchmark, item_id) sets.
  Pass 2: Pre-allocate tensors, fill values shard by shard.

Available benchmarks in actual data: bbh, mmlu_pro.
"""

from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

# Unbuffered output so we can monitor progress
print = lambda *a, **kw: (sys.stdout.write(" ".join(str(x) for x in a) + kw.get("end", "\n")), sys.stdout.flush())  # noqa: E731

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "stair-lab/zero_shot_open_llm_leaderboard"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path("/tmp/torch_measure_migration")

# All parquet files across all splits
_SPLIT_FILES = [
    *[f"data/train-{i:05d}-of-00007.parquet" for i in range(7)],
    *[f"data/validation-{i:05d}-of-00002.parquet" for i in range(2)],
    *[f"data/val2-{i:05d}-of-00003.parquet" for i in range(3)],
    *[f"data/val3-{i:05d}-of-00003.parquet" for i in range(3)],
    "data/val4-00000-of-00001.parquet",
]

_META_RE = re.compile(r"\((\d+), (\d+), '(.+)', \('([^']+)', '([^']+)'\)\)")


def _get_shard_path(filename: str) -> str:
    """Get local path for a shard (downloads if not cached)."""
    return hf_hub_download(SRC_REPO, filename, repo_type="dataset")


def _extract_shard(path: str) -> pd.DataFrame:
    """Read parquet, extract metadata fields via regex, return compact DataFrame."""
    raw = pd.read_parquet(path, columns=["label", "metadata"])
    n_raw = len(raw)

    extracted = raw["metadata"].str.extract(_META_RE.pattern)
    extracted.columns = ["subject_idx", "item_idx", "model_name", "benchmark", "item_id"]
    valid = extracted["model_name"].notna()
    n_valid = int(valid.sum())

    result = pd.DataFrame({
        "label": raw.loc[valid, "label"].values,
        "model_name": extracted.loc[valid, "model_name"].values,
        "benchmark": extracted.loc[valid, "benchmark"].values,
        "item_id": extracted.loc[valid, "item_id"].values,
    })
    del raw, extracted
    return result, n_raw, n_valid


def load_metadata() -> tuple[list[dict], dict[str, list[str]]]:
    """Download and load models_info.json and questions.json."""
    print("Downloading metadata...")
    models_path = hf_hub_download(SRC_REPO, "metadata/models_info.json", repo_type="dataset")
    with open(models_path) as f:
        raw_models = json.load(f)
    _safe_ns = {"__builtins__": {}, "nan": None, "inf": None}
    models_info = [eval(m, _safe_ns) if isinstance(m, str) else m for m in raw_models]  # noqa: S307
    print(f"  Models metadata: {len(models_info)}")

    questions_path = hf_hub_download(SRC_REPO, "metadata/questions.json", repo_type="dataset")
    with open(questions_path) as f:
        questions = json.load(f)
    for k, v in questions.items():
        print(f"  {k}: {len(v)} items")

    return models_info, questions


def pass1_collect_keys() -> tuple[set[str], dict[str, set[str]], int]:
    """Pass 1: Collect unique model names and item IDs per benchmark."""
    all_models: set[str] = set()
    bench_items: dict[str, set[str]] = {}
    total_valid = 0

    print(f"\n=== Pass 1: Scanning {len(_SPLIT_FILES)} shards for keys ===")
    for i, filename in enumerate(_SPLIT_FILES):
        path = _get_shard_path(filename)
        chunk, n_raw, n_valid = _extract_shard(path)
        total_valid += n_valid

        all_models.update(chunk["model_name"].unique())
        for bench in chunk["benchmark"].unique():
            if bench not in bench_items:
                bench_items[bench] = set()
            bench_items[bench].update(
                chunk.loc[chunk["benchmark"] == bench, "item_id"].unique()
            )

        del chunk
        gc.collect()
        print(f"  [{i+1}/{len(_SPLIT_FILES)}] {filename}: {n_raw:,} -> {n_valid:,}")

    print(f"\nPass 1 done: {total_valid:,} valid rows")
    print(f"  Models: {len(all_models)}")
    for b, items in sorted(bench_items.items()):
        print(f"  {b}: {len(items)} items")

    return all_models, bench_items, total_valid


def pass2_fill_tensors(
    subject_to_row: dict[str, int],
    bench_item_to_col: dict[str, dict[str, int]],
    tensors: dict[str, torch.Tensor],
) -> None:
    """Pass 2: Fill pre-allocated tensors from each shard."""
    print(f"\n=== Pass 2: Filling tensors from {len(_SPLIT_FILES)} shards ===")
    for i, filename in enumerate(_SPLIT_FILES):
        path = _get_shard_path(filename)
        chunk, _, _ = _extract_shard(path)

        for bench, tensor in tensors.items():
            mask = chunk["benchmark"].values == bench
            if not mask.any():
                continue
            sub = chunk.loc[mask]
            rows = np.array([subject_to_row[m] for m in sub["model_name"].values], dtype=np.int64)
            cols = np.array([bench_item_to_col[bench][it] for it in sub["item_id"].values], dtype=np.int64)
            vals = sub["label"].values.astype(np.float32)
            tensor[rows, cols] = torch.from_numpy(vals)

        del chunk
        gc.collect()
        print(f"  [{i+1}/{len(_SPLIT_FILES)}] {filename}: done")


def build_subject_metadata(subject_ids: list[str], models_info: list[dict]) -> list[dict]:
    """Build structured metadata for each model."""
    info_by_name = {m["model_name"]: m for m in models_info}
    metadata = []
    for sid in subject_ids:
        info = info_by_name.get(sid, {})
        org = sid.split("/")[0] if "/" in sid else ""
        model = sid.split("/", 1)[1] if "/" in sid else sid
        metadata.append({
            "org": org,
            "model": model,
            "param_count_b": info.get("#Params (B)"),
            "is_official": info.get("Official Providers", False),
            "upload_date": info.get("Upload To Hub Date"),
            "base_model": info.get("Base Model"),
        })
    return metadata


def build_item_contents(item_ids: list[str], questions: dict[str, list[str]], benchmark: str) -> list[str]:
    """Map item IDs to question text from questions.json."""
    bench_questions = questions.get(benchmark, [])
    contents = []
    for iid in item_ids:
        try:
            idx = int(iid.rsplit("_", 1)[1])
            if idx < len(bench_questions):
                contents.append(str(bench_questions[idx]))
            else:
                contents.append("")
        except (ValueError, IndexError):
            contents.append("")
    return contents


def ensure_repo(api: HfApi) -> None:
    """Create the destination repo if it doesn't exist."""
    try:
        api.repo_info(DST_REPO, repo_type="dataset")
        print(f"Repo {DST_REPO} exists.")
    except Exception:
        print(f"Creating {DST_REPO}...")
        api.create_repo(DST_REPO, repo_type="dataset", private=False)


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ensure_repo(api)

    # Step 1: Metadata
    models_info, questions = load_metadata()

    # Step 2: Pass 1 — collect keys (low memory: only sets of strings)
    all_models_set, bench_items_set, total_valid = pass1_collect_keys()

    # Build indexes
    all_subject_ids = sorted(all_models_set)
    subject_to_row = {sid: i for i, sid in enumerate(all_subject_ids)}
    n_subjects = len(all_subject_ids)

    benchmarks = sorted(bench_items_set.keys())
    bench_item_ids: dict[str, list[str]] = {}
    bench_item_to_col: dict[str, dict[str, int]] = {}
    for bench in benchmarks:
        items = sorted(bench_items_set[bench])
        bench_item_ids[bench] = items
        bench_item_to_col[bench] = {iid: j for j, iid in enumerate(items)}

    del all_models_set, bench_items_set
    gc.collect()

    # Step 3: Pre-allocate NaN tensors
    tensors: dict[str, torch.Tensor] = {}
    for bench in benchmarks:
        n_items = len(bench_item_ids[bench])
        tensors[bench] = torch.full((n_subjects, n_items), float("nan"))
        print(f"  Allocated {bench}: {n_subjects}x{n_items}")

    # Step 4: Pass 2 — fill tensors shard by shard
    pass2_fill_tensors(subject_to_row, bench_item_to_col, tensors)

    # Step 5: Subject metadata
    subject_metadata = build_subject_metadata(all_subject_ids, models_info)

    # Step 6: Save & upload per-benchmark
    all_data_parts = []
    all_item_ids_combined: list[str] = []
    all_item_contents_combined: list[str] = []

    # --- Save all files locally first ---
    print("\nSaving locally...")
    local_files: list[tuple[str, str]] = []  # (local_path, repo_path)

    for bench in benchmarks:
        data = tensors[bench]
        item_ids = bench_item_ids[bench]
        item_contents = build_item_contents(item_ids, questions, bench)
        density = (~torch.isnan(data)).float().mean().item()
        n_sub, n_items = data.shape

        payload = {
            "data": data,
            "subject_ids": all_subject_ids,
            "item_ids": item_ids,
            "item_contents": item_contents,
            "subject_metadata": subject_metadata,
        }

        fname = f"openllm/{bench}.pt"
        local_path = TMP_DIR / fname
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)
        local_files.append((str(local_path), fname))
        print(f"  {fname}: {n_sub}x{n_items}, density={density:.1%}")

        all_data_parts.append(data)
        all_item_ids_combined.extend(item_ids)
        all_item_contents_combined.extend(item_contents)

    # Aggregate "all"
    print("\nBuilding openllm/all.pt...")
    all_data = torch.cat(all_data_parts, dim=1)
    n_sub, n_items = all_data.shape
    density = (~torch.isnan(all_data)).float().mean().item()

    all_payload = {
        "data": all_data,
        "subject_ids": all_subject_ids,
        "item_ids": all_item_ids_combined,
        "item_contents": all_item_contents_combined,
        "subject_metadata": subject_metadata,
    }

    all_path = TMP_DIR / "openllm" / "all.pt"
    torch.save(all_payload, all_path)
    local_files.append((str(all_path), "openllm/all.pt"))
    print(f"  openllm/all.pt: {n_sub}x{n_items}, density={density:.1%}")

    # --- Upload all files ---
    print("\nUploading to HuggingFace...")
    for local_path, repo_path in local_files:
        print(f"  Uploading {repo_path}...")
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=DST_REPO,
            repo_type="dataset",
        )
        print(f"  Done: {repo_path}")

    # Step 8: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Benchmarks: {len(benchmarks)} + 1 aggregate")
    print(f"  Models: {n_subjects}")
    print(f"  Total items: {n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
