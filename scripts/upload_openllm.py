#!/usr/bin/env python3
"""Build mmlu_pro.pt + all.pt from the existing bbh.pt and re-parsed data, then upload all."""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

print = lambda *a, **kw: (sys.stdout.write(" ".join(str(x) for x in a) + kw.get("end", "\n")), sys.stdout.flush())

SRC_REPO = "stair-lab/zero_shot_open_llm_leaderboard"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path("/tmp/torch_measure_migration")
TOKEN = os.environ.get("HF_TOKEN", "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")

_SPLIT_FILES = [
    *[f"data/train-{i:05d}-of-00007.parquet" for i in range(7)],
    *[f"data/validation-{i:05d}-of-00002.parquet" for i in range(2)],
    *[f"data/val2-{i:05d}-of-00003.parquet" for i in range(3)],
    *[f"data/val3-{i:05d}-of-00003.parquet" for i in range(3)],
    "data/val4-00000-of-00001.parquet",
]

_META_RE = re.compile(r"\((\d+), (\d+), '(.+)', \('([^']+)', '([^']+)'\)\)")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing bbh.pt for shared metadata
    bbh_path = TMP_DIR / "openllm" / "bbh.pt"
    bbh_data = torch.load(bbh_path, weights_only=False)
    all_subject_ids = bbh_data["subject_ids"]
    subject_metadata = bbh_data["subject_metadata"]
    subject_to_row = {sid: i for i, sid in enumerate(all_subject_ids)}
    n_subjects = len(all_subject_ids)
    print(f"Loaded bbh.pt: {n_subjects} subjects, {bbh_data['data'].shape[1]} items")

    # Load questions.json for item contents
    questions_path = hf_hub_download(SRC_REPO, "metadata/questions.json", repo_type="dataset")
    with open(questions_path) as f:
        questions = json.load(f)

    # Build mmlu_pro: single pass, only extract mmlu_pro rows
    mmlu_pro_path = TMP_DIR / "openllm" / "mmlu_pro.pt"
    if mmlu_pro_path.exists():
        print(f"mmlu_pro.pt already exists, loading...")
        mmlu_pro_data = torch.load(mmlu_pro_path, weights_only=False)
    else:
        print(f"\nBuilding mmlu_pro from {len(_SPLIT_FILES)} shards...")
        # Pass 1: collect item IDs
        item_ids_set: set[str] = set()
        for i, filename in enumerate(_SPLIT_FILES):
            path = hf_hub_download(SRC_REPO, filename, repo_type="dataset")
            raw = pd.read_parquet(path, columns=["metadata"])
            extracted = raw["metadata"].str.extract(_META_RE.pattern)
            extracted.columns = ["si", "ii", "mn", "bench", "iid"]
            mask = extracted["bench"] == "mmlu_pro"
            item_ids_set.update(extracted.loc[mask, "iid"].unique())
            del raw, extracted
            gc.collect()
            print(f"  P1 [{i+1}/{len(_SPLIT_FILES)}] {filename}")

        item_ids = sorted(item_ids_set)
        item_to_col = {iid: j for j, iid in enumerate(item_ids)}
        n_items = len(item_ids)
        print(f"  mmlu_pro items: {n_items}")

        # Pass 2: fill tensor
        data = torch.full((n_subjects, n_items), float("nan"))
        for i, filename in enumerate(_SPLIT_FILES):
            path = hf_hub_download(SRC_REPO, filename, repo_type="dataset")
            raw = pd.read_parquet(path, columns=["label", "metadata"])
            extracted = raw["metadata"].str.extract(_META_RE.pattern)
            extracted.columns = ["si", "ii", "mn", "bench", "iid"]
            mask = extracted["bench"] == "mmlu_pro"
            if mask.any():
                rows = np.array([subject_to_row[m] for m in extracted.loc[mask, "mn"].values], dtype=np.int64)
                cols = np.array([item_to_col[it] for it in extracted.loc[mask, "iid"].values], dtype=np.int64)
                vals = raw.loc[mask, "label"].values.astype(np.float32)
                data[rows, cols] = torch.from_numpy(vals)
            del raw, extracted
            gc.collect()
            print(f"  P2 [{i+1}/{len(_SPLIT_FILES)}] {filename}")

        # Item contents
        bench_questions = questions.get("mmlu_pro", [])
        item_contents = []
        for iid in item_ids:
            try:
                idx = int(iid.rsplit("_", 1)[1])
                item_contents.append(str(bench_questions[idx]) if idx < len(bench_questions) else "")
            except (ValueError, IndexError):
                item_contents.append("")

        density = (~torch.isnan(data)).float().mean().item()
        print(f"  mmlu_pro: {n_subjects}x{n_items}, density={density:.1%}")

        mmlu_pro_data = {
            "data": data,
            "subject_ids": all_subject_ids,
            "item_ids": item_ids,
            "item_contents": item_contents,
            "subject_metadata": subject_metadata,
        }
        torch.save(mmlu_pro_data, mmlu_pro_path)
        print(f"  Saved mmlu_pro.pt")

    # Build all.pt
    print("\nBuilding openllm/all.pt...")
    all_data = torch.cat([bbh_data["data"], mmlu_pro_data["data"]], dim=1)
    all_item_ids = list(bbh_data["item_ids"]) + list(mmlu_pro_data["item_ids"])
    all_item_contents = list(bbh_data["item_contents"]) + list(mmlu_pro_data["item_contents"])
    n_sub, n_items = all_data.shape
    density = (~torch.isnan(all_data)).float().mean().item()

    all_payload = {
        "data": all_data,
        "subject_ids": all_subject_ids,
        "item_ids": all_item_ids,
        "item_contents": all_item_contents,
        "subject_metadata": subject_metadata,
    }
    all_path = TMP_DIR / "openllm" / "all.pt"
    torch.save(all_payload, all_path)
    print(f"  openllm/all.pt: {n_sub}x{n_items}, density={density:.1%}")

    # Upload all three files
    print("\nUploading to HuggingFace...")
    for fname in ["openllm/bbh.pt", "openllm/mmlu_pro.pt", "openllm/all.pt"]:
        local_path = TMP_DIR / fname
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fname,
            repo_id=DST_REPO,
            repo_type="dataset",
            token=TOKEN,
        )
        print(f"  Done: {fname}")

    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Benchmarks: bbh, mmlu_pro + all")
    print(f"  Models: {n_subjects}")
    print(f"  Total items: {n_items}")


if __name__ == "__main__":
    main()
