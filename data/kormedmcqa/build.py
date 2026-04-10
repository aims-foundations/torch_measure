"""
Build KorMedMCQA response matrix (models x items, binary correct/incorrect).

Data sources:
  1. sean0042/KorMedMCQA on HuggingFace
     - 4 subsets: doctor (435), nurse (878), pharm (885), dentist (811)
     - Test split: 3,009 items total
     - Columns: subject, year, period, q_number, question, A..E, answer (int 1-5)

  2. daekeun-ml/evaluate-llm-on-korean-dataset (GitHub)
     - Per-item model predictions in results/[KorMedMCQA] <model>.csv
     - 7 model result files (as of last check)
     - Columns: id, category, trial, answer, pred, response
     - id is {subset}_{row_index_in_test_split}
     - Some files contain multiple trials per item — we take the first trial.

Outputs:
  - processed/task_metadata.csv : item_id, question, answer, subject, year
  - processed/item_content.csv  : Full item content with options
  - processed/model_summary.csv : Per-model accuracy
  - processed/response_matrix.csv : models (rows) x items (columns) binary {0,1}
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
PROCESSED_DIR = SCRIPT_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_REPO_URL = "https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset.git"
RESULTS_REPO_DIR = RAW_DIR / "evaluate-llm-on-korean-dataset"

SUBSETS = ["doctor", "nurse", "pharm", "dentist"]
LETTER_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def clone_results_repo():
    """Clone daekeun-ml eval repo that hosts per-item prediction CSVs."""
    if RESULTS_REPO_DIR.exists():
        print(f"  Results repo already cloned: {RESULTS_REPO_DIR}")
        return RESULTS_REPO_DIR
    print(f"  Cloning {RESULTS_REPO_URL}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", RESULTS_REPO_URL, str(RESULTS_REPO_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        sys.exit(1)
    return RESULTS_REPO_DIR


def load_test_items():
    """Load test items from sean0042/KorMedMCQA for all subsets."""
    from datasets import load_dataset

    rows = []
    item_idx = 0
    # Map (subset, row_index) -> global item_idx, needed to align model results.
    key_to_idx = {}

    for subset in SUBSETS:
        print(f"  Loading {subset}...", end=" ", flush=True)
        ds = load_dataset("sean0042/KorMedMCQA", subset, split="test")
        for i, row in enumerate(ds):
            ans_int = int(row["answer"])
            ans_letter = LETTER_MAP.get(ans_int, str(row["answer"]))

            options = [row.get("A", ""), row.get("B", ""), row.get("C", ""),
                       row.get("D", ""), row.get("E", "")]
            options_text = "\n".join(
                f"{letter}: {opt}" for letter, opt in zip("ABCDE", options) if opt
            )
            full_content = f"{row['question']}\n\n{options_text}"

            item_id = f"kormedmcqa_{subset}_{i:04d}"
            key_to_idx[(subset, i)] = item_idx
            rows.append({
                "item_id": item_id,
                "subset_row_key": f"{subset}_{i}",
                "subject": subset,
                "year": row.get("year", ""),
                "period": row.get("period", ""),
                "q_number": row.get("q_number", ""),
                "question": row["question"],
                "A": row.get("A", ""),
                "B": row.get("B", ""),
                "C": row.get("C", ""),
                "D": row.get("D", ""),
                "E": row.get("E", ""),
                "answer_int": ans_int,
                "answer_letter": ans_letter,
                "full_content": full_content,
            })
            item_idx += 1
        print(f"{len(ds)} items")

    return rows, key_to_idx


def load_model_results(results_dir: Path, key_to_idx: dict, n_total: int):
    """Load model predictions from results/[KorMedMCQA] *.csv."""
    results_root = results_dir / "results"
    if not results_root.exists():
        print(f"  ERROR: results dir not found: {results_root}")
        return {}

    result_files = sorted(p for p in results_root.glob("[[]KorMedMCQA[]] *.csv"))
    print(f"  Found {len(result_files)} KorMedMCQA result files")

    model_responses = {}
    for f in result_files:
        model_name = f.stem.replace("[KorMedMCQA] ", "").strip()
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"    SKIP {model_name}: read failed: {e}")
            continue

        if not {"id", "answer", "pred"}.issubset(df.columns):
            print(f"    SKIP {model_name}: missing columns")
            continue

        # Deduplicate by id (keep first occurrence)
        df = df.drop_duplicates(subset="id", keep="first")

        scores = np.full(n_total, np.nan, dtype=np.float32)
        mapped = 0
        for _, r in df.iterrows():
            item_id = str(r["id"])
            if "_" not in item_id:
                continue
            subset, idx_str = item_id.rsplit("_", 1)
            try:
                i = int(idx_str)
            except ValueError:
                continue
            key = (subset, i)
            if key not in key_to_idx:
                continue
            gold = str(r["answer"]).strip().upper() if pd.notna(r["answer"]) else ""
            pred = str(r["pred"]).strip().upper() if pd.notna(r["pred"]) else ""
            if not gold:
                continue
            scores[key_to_idx[key]] = 1.0 if pred == gold else 0.0
            mapped += 1

        n_ans = int(np.sum(~np.isnan(scores)))
        acc = float(np.nanmean(scores)) if n_ans > 0 else 0.0
        print(f"    {model_name}: acc={acc:.3f} ({n_ans}/{n_total}, {mapped} rows)")
        model_responses[model_name] = scores.tolist()

    return model_responses


def main():
    print("KorMedMCQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Load test items
    print("Loading KorMedMCQA test items from HuggingFace...")
    rows, key_to_idx = load_test_items()
    n_items = len(rows)
    print(f"\nTotal items: {n_items}")

    # Step 2: Clone results repo
    print("\nCloning results repo...")
    results_dir = clone_results_repo()

    # Step 3: Load model results
    print("\nLoading per-item model predictions...")
    model_responses = load_model_results(results_dir, key_to_idx, n_items)

    if not model_responses:
        print("ERROR: No model responses loaded.")
        sys.exit(1)

    # Step 4: Write task_metadata.csv
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "question", "answer", "subject", "year",
            "period", "q_number", "language", "benchmark",
        ])
        for r in rows:
            writer.writerow([
                r["item_id"], r["question"], r["answer_letter"],
                r["subject"], r["year"], r["period"], r["q_number"],
                "Korean", "KorMedMCQA",
            ])
    print(f"\nSaved task_metadata.csv: {n_items} rows")

    # Step 5: Write item_content.csv
    content_path = PROCESSED_DIR / "item_content.csv"
    with open(content_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "content"])
        for r in rows:
            writer.writerow([r["item_id"], r["full_content"]])
    print(f"Saved item_content.csv: {n_items} rows")

    # Step 6: Build response matrix
    item_ids = [r["item_id"] for r in rows]
    rm_df = pd.DataFrame(
        {model: scores for model, scores in model_responses.items()},
        index=item_ids,
    ).T
    rm_df.index.name = "model"
    rm_path = PROCESSED_DIR / "response_matrix.csv"
    rm_df.to_csv(rm_path)
    print(f"Saved response_matrix.csv: {rm_df.shape}")

    # Step 7: model_summary.csv
    summary_rows = []
    for model, scores in model_responses.items():
        arr = np.array(scores, dtype=float)
        n_ans = int(np.sum(~np.isnan(arr)))
        acc = float(np.nanmean(arr)) if n_ans > 0 else 0.0
        summary_rows.append({
            "model": model,
            "source": "daekeun-ml/evaluate-llm-on-korean-dataset",
            "overall_accuracy": acc,
            "n_items": n_ans,
            "notes": "per-item predictions",
        })
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "model_summary.csv", index=False)
    print(f"Saved model_summary.csv: {len(summary_rows)} models")

    print(f"\nDone! {rm_df.shape[0]} models x {rm_df.shape[1]} items")


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
