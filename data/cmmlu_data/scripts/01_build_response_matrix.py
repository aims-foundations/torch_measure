#!/usr/bin/env python3
"""
Build task metadata and item content for CMMLU (Chinese MMLU) from HuggingFace.

Data source:
  haonan-li/cmmlu on HuggingFace
  - 67 subject configs covering Chinese academic/professional exams
  - Loaded from the cmmlu_v1_0_1.zip file (CSV files inside)
  - Columns: Question, A, B, C, D, Answer
  - Splits: test and dev

No per-model prediction data is available, so we build:
  - task_metadata.csv  : item_id, question (first 200 chars), answer_key,
                         config, split, source_dataset, language
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : empty placeholder
  - response_matrix.csv: placeholder with item_ids and answer keys only

All paths use Path(__file__).resolve().parent.parent
"""

import sys
import warnings
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_cmmlu_zip():
    """Download the CMMLU zip from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset"
    )
    return zip_path


def load_cmmlu(zip_path):
    """Load all CMMLU subjects from the zip file."""
    all_rows = []
    item_id = 0

    with zipfile.ZipFile(zip_path) as zf:
        csv_files = sorted([
            n for n in zf.namelist()
            if n.endswith(".csv") and "/" in n
        ])

        for csv_name in csv_files:
            parts = csv_name.split("/")
            if len(parts) < 2:
                continue
            split_name = parts[0]  # "test" or "dev"
            config = parts[1].replace(".csv", "")

            print(f"  Loading {split_name}/{config}...", end=" ")
            try:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # Normalize column names
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl == "question":
                    col_map[c] = "Question"
                elif cl == "answer":
                    col_map[c] = "Answer"
                elif cl in ("a", "b", "c", "d"):
                    col_map[c] = cl.upper()
            df = df.rename(columns=col_map)

            n_items = 0
            for _, row in df.iterrows():
                q = str(row.get("Question", ""))
                a_key = str(row.get("Answer", "")).strip()
                opt_a = str(row.get("A", ""))
                opt_b = str(row.get("B", ""))
                opt_c = str(row.get("C", ""))
                opt_d = str(row.get("D", ""))

                options_text = (
                    f"A: {opt_a}\nB: {opt_b}\n"
                    f"C: {opt_c}\nD: {opt_d}"
                )
                full_content = f"{q}\n\n{options_text}"

                all_rows.append({
                    "item_id": f"cmmlu_{item_id:06d}",
                    "question_short": q[:200],
                    "answer_key": a_key,
                    "config": config,
                    "split": split_name,
                    "source_dataset": "haonan-li/cmmlu",
                    "language": "chinese",
                    "full_content": full_content,
                })
                item_id += 1
                n_items += 1

            print(f"{n_items} items")

    return all_rows


def main():
    print("=" * 70)
    print("CMMLU Task Metadata Builder")
    print("=" * 70)

    print("\nDownloading CMMLU zip from HuggingFace...")
    zip_path = download_cmmlu_zip()
    print(f"  Zip path: {zip_path}")

    print("\nLoading CMMLU data...")
    rows = load_cmmlu(zip_path)
    if not rows:
        print("ERROR: No data loaded.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\nTotal items loaded: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Configs: {df['config'].nunique()}")

    # ── task_metadata.csv ──
    meta_cols = [
        "item_id", "question_short", "answer_key", "config",
        "split", "source_dataset", "language",
    ]
    meta_df = df[meta_cols].copy()
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved task_metadata.csv: {meta_path}")
    print(f"  Shape: {meta_df.shape}")

    # ── item_content.csv ──
    content_df = df[["item_id", "full_content"]].copy()
    content_df.columns = ["item_id", "content"]
    content_path = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(content_path, index=False)
    print(f"Saved item_content.csv: {content_path}")
    print(f"  Shape: {content_df.shape}")

    # ── response_matrix.csv (gold answers only, no model responses) ──
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) > 0:
        rm = test_df[["item_id", "answer_key"]].set_index("item_id")
        rm.columns = ["gold_answer"]
        rm_path = PROCESSED_DIR / "response_matrix.csv"
        rm.to_csv(rm_path)
        print(f"Saved response_matrix.csv (gold answers): {rm_path}")
        print(f"  Shape: {rm.shape}")
        print(f"  Answer distribution:")
        print(f"    {test_df['answer_key'].value_counts().to_dict()}")
    else:
        print("WARNING: No test split items found.")

    # ── model_summary.csv (placeholder) ──
    summary_df = pd.DataFrame(columns=[
        "model", "source", "overall_accuracy", "n_items", "notes"
    ])
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model_summary.csv (empty): {summary_path}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {len(df)}")
    print(f"  Test items:  {len(test_df)}")
    print(f"  Configs:     {df['config'].nunique()}")
    print(f"  Language:    Chinese")
    print(f"  Has model predictions: No")
    print(f"  Has gold answers: Yes")

    if len(test_df) > 0:
        print(f"\n  Per-config test item counts:")
        for cfg in sorted(test_df["config"].unique()):
            n = (test_df["config"] == cfg).sum()
            print(f"    {cfg:50s}: {n} items")

    print("\nDone!")


if __name__ == "__main__":
    main()
