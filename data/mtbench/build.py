"""
Build MT-Bench response matrix from GPT-4 single-answer judgment data.

Data source:
  - https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl
  - 34 models x 80 questions, GPT-4 absolute scores 1-10

Score format:
  - Raw scores: 1-10 integer (GPT-4 quality rating)
  - Each model is evaluated on 80 multi-turn questions across 8 categories

Outputs:
  - raw/gpt-4_single.jsonl: Raw judgment data
  - processed/response_matrix.csv: Models (rows) x questions (columns), raw 1-10 scores
"""

INFO = {
    'description': 'Build MT-Bench response matrix from GPT-4 single-answer judgment data',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2306.05685',
    'data_source_url': """https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl""",
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-4.0',
    'citation': """@misc{zheng2023judgingllmasajudgemtbenchchatbot,
      title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2306.05685}, 
}""",
    'tags': ['preference', 'pairwise'],
}


import sys
from pathlib import Path
import os
import json
import urllib.request
import urllib.error

import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Data URL
JUDGMENT_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/model_judgment/gpt-4_single.jsonl"
)


def download_data():
    """Download the GPT-4 single-answer judgment JSONL."""
    dest = os.path.join(RAW_DIR, "gpt-4_single.jsonl")

    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  JSONL already exists: {dest}")
        return dest

    print(f"  Downloading from {JUDGMENT_URL}")
    req = urllib.request.Request(JUDGMENT_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  Saved: {dest} ({len(data):,} bytes)")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  ERROR: Download failed: {e}")
        raise

    return dest


def parse_judgments(jsonl_path):
    """Parse JSONL file and extract model, question_id, score tuples."""
    records = []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            model = record.get("model_id", record.get("model", ""))
            question_id = record.get("question_id", "")
            score = record.get("score", None)
            turn = record.get("turn", 1)
            category = record.get("category", "")

            # Some records have rating instead of score
            if score is None:
                score = record.get("rating", None)

            # Extract score from judgment text if needed
            if score is None or score == -1:
                judgment_text = record.get("judgment", "")
                import re

                match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", judgment_text)
                if match:
                    try:
                        score = float(match.group(1))
                    except ValueError:
                        pass

            if model and question_id is not None and score is not None and score != -1:
                try:
                    score = float(score)
                    records.append({
                        "model": model,
                        "question_id": str(question_id),
                        "turn": turn,
                        "category": category,
                        "score": score,
                    })
                except (ValueError, TypeError):
                    pass

    print(f"  Parsed {len(records):,} judgment records")
    return pd.DataFrame(records)


def build_response_matrix(judgments_df):
    """Build response matrix from parsed judgments."""
    print("\nBuilding response matrix...")

    # MT-Bench has 2 turns per question; average across turns for each model-question pair
    avg_scores = (
        judgments_df.groupby(["model", "question_id"])["score"]
        .mean()
        .reset_index()
    )

    # Pivot to matrix
    matrix_df = avg_scores.pivot(index="model", columns="question_id", values="score")
    matrix_df.index.name = "Model"

    # Sort columns numerically if possible
    try:
        sorted_cols = sorted(matrix_df.columns, key=lambda x: int(x))
        matrix_df = matrix_df[sorted_cols]
    except (ValueError, TypeError):
        matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def print_statistics(judgments_df, matrix_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  MT-BENCH STATISTICS")
    print(f"{'='*60}")

    n_models, n_questions = matrix_df.shape
    total_cells = n_models * n_questions
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Models:        {n_models}")
    print(f"    Questions:     {n_questions}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Score distribution
    all_scores = matrix_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Score distribution (1-10 scale):")
        print(f"    Mean:   {np.mean(valid_scores):.2f}")
        print(f"    Median: {np.median(valid_scores):.1f}")
        print(f"    Std:    {np.std(valid_scores):.2f}")
        print(f"    Min:    {np.min(valid_scores):.1f}")
        print(f"    Max:    {np.max(valid_scores):.1f}")

        # Score histogram
        print(f"\n  Score histogram:")
        for score_val in range(1, 11):
            count = np.sum((valid_scores >= score_val - 0.5) & (valid_scores < score_val + 0.5))
            pct = count / len(valid_scores) * 100
            bar = "#" * int(pct)
            print(f"    {score_val:2d}: {count:5,} ({pct:5.1f}%) {bar}")

    # Per-turn analysis
    if "turn" in judgments_df.columns:
        print(f"\n  Per-turn statistics:")
        for turn in sorted(judgments_df["turn"].unique()):
            turn_scores = judgments_df[judgments_df["turn"] == turn]["score"]
            print(
                f"    Turn {turn}: mean={turn_scores.mean():.2f}, "
                f"std={turn_scores.std():.2f}, n={len(turn_scores):,}"
            )

    # Category analysis
    if "category" in judgments_df.columns and judgments_df["category"].notna().any():
        print(f"\n  Per-category mean scores:")
        cat_means = judgments_df.groupby("category")["score"].agg(["mean", "std", "count"])
        cat_means = cat_means.sort_values("mean", ascending=False)
        for cat, row in cat_means.iterrows():
            if cat:
                print(
                    f"    {cat:25s}  mean={row['mean']:.2f}  "
                    f"std={row['std']:.2f}  n={int(row['count']):,}"
                )

    # Per-model stats
    per_model_mean = matrix_df.mean(axis=1).sort_values(ascending=False)
    print(f"\n  Per-model mean score:")
    print(f"    Best:   {per_model_mean.iloc[0]:.2f} ({per_model_mean.index[0]})")
    print(f"    Worst:  {per_model_mean.iloc[-1]:.2f} ({per_model_mean.index[-1]})")
    print(f"    Median: {per_model_mean.median():.2f}")
    print(f"    Std:    {per_model_mean.std():.2f}")

    print(f"\n  All models ranked by mean score:")
    for model, score in per_model_mean.items():
        print(f"    {model:45s}  {score:.2f}")

    # Per-question difficulty
    per_q_mean = matrix_df.mean(axis=0)
    print(f"\n  Per-question difficulty (mean model score):")
    print(f"    Easiest: {per_q_mean.max():.2f} (Q{per_q_mean.idxmax()})")
    print(f"    Hardest: {per_q_mean.min():.2f} (Q{per_q_mean.idxmin()})")
    print(f"    Median:  {per_q_mean.median():.2f}")
    print(f"    Std:     {per_q_mean.std():.2f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("MT-Bench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download
    print("STEP 1: Downloading GPT-4 single-answer judgments")
    print("-" * 60)
    jsonl_path = download_data()

    # Step 2: Parse
    print("\nSTEP 2: Parsing judgment records")
    print("-" * 60)
    judgments_df = parse_judgments(jsonl_path)

    # Step 3: Build matrix
    print("\nSTEP 3: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix(judgments_df)

    # Step 4: Statistics
    print("\nSTEP 4: Detailed statistics")
    print("-" * 60)
    print_statistics(judgments_df, matrix_df)


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
