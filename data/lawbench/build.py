"""
Build LawBench response matrix from per-model per-question predictions.

Data source:
  - https://github.com/open-compass/LawBench
  - 51 models x 10K+ Chinese legal items
  - Each model directory under predictions/zero_shot/ contains per-task JSON files

Score format:
  - Binary 0/1: exact match between prediction and reference answer

Outputs:
  - raw/LawBench/: Cloned GitHub repo
  - processed/response_matrix.csv: Models (rows) x items (columns)
  - processed/item_content.csv: Per-item metadata (task + prompt text)
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

REPO_URL = "https://github.com/open-compass/LawBench.git"


def main():
    print("LawBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Clone repo
    lawbench_dir = os.path.join(RAW_DIR, "LawBench")
    if not os.path.exists(lawbench_dir):
        print("  Cloning LawBench...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, lawbench_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            sys.exit(1)

    # Step 2: Find predictions directory
    pred_dir = os.path.join(lawbench_dir, "predictions", "zero_shot")
    if not os.path.exists(pred_dir):
        pred_dir = os.path.join(lawbench_dir, "predictions")
        if not os.path.exists(pred_dir):
            print(f"  No predictions directory found")
            sys.exit(1)

    print(f"  Reading predictions from {pred_dir}...")

    # Step 3: Parse predictions
    records = []
    all_items = {}

    for entry in sorted(os.listdir(pred_dir)):
        model_dir = os.path.join(pred_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        model_name = entry

        for task_file in sorted(os.listdir(model_dir)):
            if not task_file.endswith(".json"):
                continue
            task_name = task_file.replace(".json", "")

            try:
                with open(os.path.join(model_dir, task_file)) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Data can be a list or a dict keyed by string indices
            if isinstance(data, dict):
                items_iter = data.items()
            elif isinstance(data, list):
                items_iter = enumerate(data)
            else:
                continue

            for j, item in items_iter:
                item_id = f"{task_name}_{j}"
                pred = item.get("prediction", "")
                ref = item.get("refr", item.get("reference", ""))
                prompt = item.get("origin_prompt", "")[:500]

                correct = 1.0 if str(pred).strip() == str(ref).strip() else 0.0
                records.append({
                    "model": model_name,
                    "item_id": item_id,
                    "score": correct,
                })

                if item_id not in all_items:
                    all_items[item_id] = f"[{task_name}] {prompt}"

    if not records:
        print("  No predictions found")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  {len(df)} predictions, {df['model'].nunique()} models, {df['item_id'].nunique()} items")

    # Step 4: Build and save response matrix
    pivot = df.pivot_table(index="model", columns="item_id", values="score", aggfunc="first")
    pivot.index.name = "model"

    pivot.to_csv(os.path.join(PROCESSED_DIR, "response_matrix.csv"))
    print(f"  Saved response_matrix.csv: {pivot.shape}")

    item_content = pd.DataFrame([
        {"item_id": k, "content": v} for k, v in all_items.items()
    ])
    item_content.to_csv(os.path.join(PROCESSED_DIR, "item_content.csv"), index=False)
    print(f"  Saved item_content.csv: {len(item_content)} items")


if __name__ == "__main__":
    main()
