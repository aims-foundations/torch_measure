"""
Build IgakuQA response matrix from Japanese medical exam data.

Data source:
  - https://github.com/jungokasai/IgakuQA
  - 5 models x exam years 2018-2022
  - Text-only questions only (no image-dependent questions)
  - Multi-answer questions: prediction must match full answer set (order-insensitive)

Score format:
  - Binary 0/1: whether prediction matches gold answer set

Outputs:
  - raw/IgakuQA/: Cloned GitHub repo
  - processed/response_matrix.csv: Models (rows) x items (columns)
  - processed/item_content.csv: Per-item metadata (problem text, choices, answer)
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPO_URL = "https://github.com/jungokasai/IgakuQA.git"

YEARS = ["2018", "2019", "2020", "2021", "2022"]
MODELS = ["gpt3", "chatgpt", "gpt4", "student-majority", "translate_chatgpt-en"]
MODEL_DISPLAY = {
    "gpt3": "GPT-3",
    "chatgpt": "ChatGPT",
    "gpt4": "GPT-4",
    "student-majority": "Student-Majority",
    "translate_chatgpt-en": "Translate-ChatGPT-EN",
}


def main():
    print("IgakuQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Clone repo
    igaku_dir = os.path.join(RAW_DIR, "IgakuQA")
    if not os.path.exists(igaku_dir):
        print("  Cloning IgakuQA...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, igaku_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            sys.exit(1)

    # Step 2: Load all questions (gold answers + text), text-only only
    print("  Loading questions...")
    questions = {}
    for year in YEARS:
        data_dir = os.path.join(igaku_dir, "data", year)
        if not os.path.exists(data_dir):
            print(f"  WARNING: No data directory for {year}")
            continue
        for fn in sorted(os.listdir(data_dir)):
            if "_" in fn:  # skip metadata/translate files
                continue
            exam_name = fn.replace(".jsonl", "")
            with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if not d["text_only"]:
                        continue
                    questions[d["problem_id"]] = {
                        "answer": sorted(d["answer"]),
                        "problem_text": d["problem_text"],
                        "choices": d["choices"],
                        "points": d.get("points", "1"),
                        "exam": exam_name,
                        "year": year,
                    }

    problem_ids = sorted(questions.keys())
    pid_to_idx = {pid: i for i, pid in enumerate(problem_ids)}
    n_items = len(problem_ids)
    n_models = len(MODELS)

    print(f"  Total text-only items: {n_items}")

    # Step 3: Load baseline results and score
    response_matrix = np.full((n_models, n_items), np.nan)

    for year in YEARS:
        results_dir = os.path.join(igaku_dir, "baseline_results", year)
        if not os.path.exists(results_dir):
            continue
        for fn in sorted(os.listdir(results_dir)):
            base = fn.replace(".jsonl", "")
            parts = base.split("_", 1)
            if len(parts) != 2:
                continue
            exam_name, model_name = parts

            if model_name not in MODELS:
                # Handle "translate_chatgpt-en" which has an extra underscore
                parts2 = base.split("_", 2)
                if len(parts2) == 3:
                    model_name = parts2[1] + "_" + parts2[2]
                if model_name not in MODELS:
                    continue

            model_idx = MODELS.index(model_name)

            with open(os.path.join(results_dir, fn), encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    pid = d["problem_id"]
                    if pid not in pid_to_idx:
                        continue
                    item_idx = pid_to_idx[pid]
                    pred = sorted(d["prediction"].split(","))
                    gold = questions[pid]["answer"]
                    correct = 1 if pred == gold else 0
                    response_matrix[model_idx, item_idx] = correct

    # Step 4: Print statistics
    for i, model in enumerate(MODELS):
        answered = int(np.sum(~np.isnan(response_matrix[i])))
        correct = int(np.nansum(response_matrix[i]))
        acc = correct / answered * 100 if answered > 0 else 0
        print(f"  {MODEL_DISPLAY[model]}: {answered}/{n_items} items, {correct} correct ({acc:.1f}%)")

    # Step 5: Save response matrix
    model_names = [MODEL_DISPLAY[m] for m in MODELS]
    header = ["model_id"] + problem_ids

    with open(os.path.join(PROCESSED_DIR, "response_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, model in enumerate(model_names):
            row = [model]
            for j in range(n_items):
                val = response_matrix[i, j]
                if np.isnan(val):
                    row.append("")
                else:
                    row.append(int(val))
            writer.writerow(row)

    print(f"\n  Saved response_matrix.csv: {n_models} models x {n_items} items")

    # Step 6: Save item_content.csv
    with open(os.path.join(PROCESSED_DIR, "item_content.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "exam", "year", "problem_text", "choices", "answer", "points"])
        for pid in problem_ids:
            q = questions[pid]
            choices_str = " | ".join(q["choices"])
            answer_str = ",".join(q["answer"])
            writer.writerow([pid, q["exam"], q["year"], q["problem_text"], choices_str, answer_str, q["points"]])

    print(f"  Saved item_content.csv: {n_items} items")

    # Summary
    nan_count = int(np.sum(np.isnan(response_matrix)))
    total_cells = n_models * n_items
    print(f"\n  Missing values: {nan_count}/{total_cells} ({nan_count / total_cells * 100:.1f}%)")
    print(f"  Overall accuracy: {np.nanmean(response_matrix) * 100:.1f}%")


if __name__ == "__main__":
    main()
