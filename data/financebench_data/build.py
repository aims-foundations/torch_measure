"""
Build FinanceBench response matrix from per-model results.

Data source:
  - https://github.com/patronus-ai/financebench
  - 16 model configs x 150 financial QA items
  - results/ contains JSONL files named {model}_{config}.jsonl
  - Each line has: financebench_id, question, label ("Correct Answer" or not)

Score format:
  - Binary 0/1: whether label == "Correct Answer"

Outputs:
  - raw/financebench/: Cloned GitHub repo
  - processed/response_matrix.csv: Models (rows) x items (columns)
  - processed/item_content.csv: Per-item metadata (question text)
"""

import json
import os
import subprocess
import sys

import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPO_URL = "https://github.com/patronus-ai/financebench.git"


def main():
    print("FinanceBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Clone repo
    fb_dir = os.path.join(RAW_DIR, "financebench")
    if not os.path.exists(fb_dir):
        print("  Cloning FinanceBench...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, fb_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            sys.exit(1)

    # Step 2: Find results directory
    results_dir = os.path.join(fb_dir, "results")
    if not os.path.exists(results_dir):
        print(f"  No results directory found at {results_dir}")
        sys.exit(1)

    # Step 3: Load benchmark questions
    data_file = os.path.join(fb_dir, "data", "financebench_open_source.jsonl")
    if not os.path.exists(data_file):
        # Try other locations
        for root, dirs, files in os.walk(fb_dir):
            for f in files:
                if f.endswith(".jsonl"):
                    data_file = os.path.join(root, f)
                    break

    items = {}
    if os.path.exists(data_file):
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                qid = item.get("financebench_id", item.get("id", ""))
                question = item.get("question", "")
                items[str(qid)] = question[:1000]

    # Step 4: Parse model results (JSONL files: {model}_{config}.jsonl)
    print(f"  Reading results from {results_dir}...")
    records = []

    for result_file in sorted(os.listdir(results_dir)):
        if not result_file.endswith(".jsonl"):
            continue
        model_name = result_file.replace(".jsonl", "")
        fpath = os.path.join(results_dir, result_file)

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                qid = str(row.get("financebench_id", row.get("id", "")))
                label = row.get("label", "")
                score = 1.0 if label == "Correct Answer" else 0.0

                records.append({
                    "model": model_name,
                    "item_id": qid,
                    "score": score,
                })

                if qid not in items:
                    question = row.get("question", f"FinanceBench {qid}")
                    items[qid] = question[:1000]

    if not records:
        print("  No results found")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  {len(df)} results, {df['model'].nunique()} models, {df['item_id'].nunique()} items")

    # Step 5: Build and save response matrix
    pivot = df.pivot_table(index="model", columns="item_id", values="score", aggfunc="first")
    pivot.index.name = "model"

    pivot.to_csv(os.path.join(PROCESSED_DIR, "response_matrix.csv"))
    print(f"  Saved response_matrix.csv: {pivot.shape}")

    # Save item content
    item_content = pd.DataFrame([
        {"item_id": k, "content": v} for k, v in items.items()
    ])
    item_content.to_csv(os.path.join(PROCESSED_DIR, "item_content.csv"), index=False)
    print(f"  Saved item_content.csv: {len(item_content)} items")


if __name__ == "__main__":
    main()
