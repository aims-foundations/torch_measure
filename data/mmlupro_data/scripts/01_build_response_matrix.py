#!/usr/bin/env python3
"""
Build MMLU-Pro response matrices from multiple data sources.

Data sources:
  1. Per-question model outputs (48 models x 12,032 questions) from
     TIGER-AI-Lab/MMLU-Pro GitHub eval_results/
  2. Per-category leaderboard (249 models x 14 categories) from
     TIGER-Lab/mmlu_pro_leaderboard_submission on HuggingFace
  3. Aggregate MMLU-Pro scores (4,576 models) from
     open-llm-leaderboard/contents on HuggingFace

Outputs:
  - response_matrix.csv          : Per-question binary matrix (questions x models)
  - response_matrix_category.csv : Per-category accuracy matrix (categories x models)
  - model_summary.csv            : Summary statistics per model
  - question_metadata.csv        : Question metadata (id, category, answer, src)
"""

import os
import re
import sys
import json
import zipfile
import io
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
EVAL_RESULTS_DIR = RAW_DIR / "eval_results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Helper: extract model name from zip filename
# ──────────────────────────────────────────────────────────────────────
def extract_model_name(zip_filename):
    """Extract a clean model name from the eval_results zip filename."""
    name = zip_filename
    # Remove prefix
    name = re.sub(r"^model_outputs_", "", name)
    # Remove suffix patterns
    name = re.sub(r"_\d+shots\.json\.zip$", "", name)
    name = re.sub(r"_\d+shots\.zip$", "", name)
    name = re.sub(r"_\d+-shots\.zip$", "", name)
    name = re.sub(r"_\d+shots_\d+_\d+_\d+\.zip$", "", name)
    name = re.sub(r"\.zip$", "", name)
    name = re.sub(r"\.json$", "", name)
    return name


def extract_answer_from_pred(pred_text, answer_choices="ABCDEFGHIJ"):
    """Extract predicted answer letter from model prediction field.

    Follows the same logic as TIGER-AI-Lab's compute_accuracy.py:
    - Level 1: regex for 'answer is (X)'
    - Level 2: look for last standalone letter A-J
    - Fallback: random choice (we return None to mark as failed extraction)
    """
    if pred_text is None:
        return None

    pred_str = str(pred_text).strip()
    if len(pred_str) == 1 and pred_str in answer_choices:
        return pred_str

    # Level 1: "answer is (X)" pattern
    match = re.search(r"answer is\s*\(?([A-J])\)?", pred_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Level 2: Look for last standalone capital letter A-J
    matches = re.findall(r"\b([A-J])\b", pred_str)
    if matches:
        return matches[-1]

    # Check if just a letter followed by period/colon
    match = re.search(r"^([A-J])[.\s:]", pred_str)
    if match:
        return match.group(1)

    return None


# ──────────────────────────────────────────────────────────────────────
# PART 1: Build per-question response matrix from GitHub eval_results
# ──────────────────────────────────────────────────────────────────────
def build_per_question_matrix():
    """Parse all 48 model output zips and build binary response matrix."""
    print("=" * 70)
    print("PART 1: Building per-question response matrix from eval_results/")
    print("=" * 70)

    zip_files = sorted(EVAL_RESULTS_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} model output zip files\n")

    if not zip_files:
        print("  No zip files found. Skipping per-question matrix.")
        return None, None

    # We'll collect: {model_name: {question_id: correct_bool}}
    model_results = {}
    question_metadata = {}
    failed_models = []

    for i, zf_path in enumerate(zip_files):
        model_name = extract_model_name(zf_path.name)
        print(f"  [{i+1:2d}/{len(zip_files)}] Processing {model_name}...", end=" ")

        try:
            with zipfile.ZipFile(zf_path) as zf:
                json_files = [n for n in zf.namelist()
                              if n.endswith(".json") and not n.startswith("__MACOSX")]
                if not json_files:
                    print("SKIP (no JSON in zip)")
                    failed_models.append(model_name)
                    continue

                with zf.open(json_files[0]) as f:
                    data = json.load(f)

            if not isinstance(data, list):
                print(f"SKIP (unexpected type: {type(data).__name__})")
                failed_models.append(model_name)
                continue

            correct_count = 0
            total = len(data)
            results_dict = {}

            for item in data:
                if not isinstance(item, dict):
                    continue
                qid = item.get("question_id")
                if qid is None:
                    continue

                # Store metadata on first pass
                if qid not in question_metadata:
                    question_metadata[qid] = {
                        "question_id": qid,
                        "category": item.get("category", ""),
                        "answer": item.get("answer", ""),
                        "answer_index": item.get("answer_index", -1),
                        "src": item.get("src", ""),
                    }

                # Check correctness
                gold = str(item.get("answer", "")).strip().upper()
                pred = item.get("pred")
                extracted = extract_answer_from_pred(pred)

                is_correct = (extracted is not None and extracted == gold)
                results_dict[qid] = int(is_correct)
                if is_correct:
                    correct_count += 1

            model_results[model_name] = results_dict
            acc = correct_count / total * 100 if total > 0 else 0
            print(f"{total} questions, {correct_count} correct ({acc:.1f}%)")

        except Exception as e:
            print(f"ERROR: {e}")
            failed_models.append(model_name)

    if failed_models:
        print(f"\n  Failed models: {failed_models}")

    if not model_results:
        print("  No valid model results. Skipping per-question matrix.")
        return None, None

    # Build the response matrix: rows = questions, columns = models
    all_qids = sorted(question_metadata.keys())
    all_models = sorted(model_results.keys())

    print(f"\n  Building matrix: {len(all_qids)} questions x {len(all_models)} models")

    matrix_data = {}
    for model_name in all_models:
        col = []
        for qid in all_qids:
            col.append(model_results[model_name].get(qid, np.nan))
        matrix_data[model_name] = col

    response_matrix = pd.DataFrame(matrix_data, index=all_qids)
    response_matrix.index.name = "question_id"

    # Build question metadata dataframe
    q_meta_df = pd.DataFrame([question_metadata[qid] for qid in all_qids])
    q_meta_df = q_meta_df.set_index("question_id")

    # Print summary statistics
    per_model_acc = response_matrix.mean()
    print(f"\n  Per-model accuracy range: {per_model_acc.min():.3f} - {per_model_acc.max():.3f}")
    print(f"  Mean accuracy: {per_model_acc.mean():.3f}")

    per_question_diff = response_matrix.mean(axis=1)
    easy = (per_question_diff == 1.0).sum()
    hard = (per_question_diff == 0.0).sum()
    print(f"  Questions solved by all models: {easy}")
    print(f"  Questions solved by no models: {hard}")

    return response_matrix, q_meta_df


# ──────────────────────────────────────────────────────────────────────
# PART 2: Build per-category matrix from TIGER-Lab leaderboard
# ──────────────────────────────────────────────────────────────────────
def build_per_category_matrix():
    """Load the leaderboard CSV with 249 models x 14 categories."""
    print("\n" + "=" * 70)
    print("PART 2: Building per-category matrix from leaderboard_results.csv")
    print("=" * 70)

    csv_path = RAW_DIR / "leaderboard_results.csv"
    if not csv_path.exists():
        print("  leaderboard_results.csv not found. Skipping.")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} models from leaderboard")

    categories = [
        "Biology", "Business", "Chemistry", "Computer Science", "Economics",
        "Engineering", "Health", "History", "Law", "Math",
        "Philosophy", "Physics", "Psychology", "Other"
    ]

    # Clean up: ensure category columns are numeric
    for cat in categories:
        df[cat] = pd.to_numeric(df[cat], errors="coerce")

    # Also ensure Overall is numeric
    df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce")

    # Remove duplicate model names (keep the one with higher Overall score)
    df = df.sort_values("Overall", ascending=False).drop_duplicates(
        subset="Models", keep="first"
    )

    # Build category matrix: rows = categories, columns = models
    cat_matrix = df.set_index("Models")[categories].T
    cat_matrix.index.name = "category"
    print(f"  Category matrix shape: {cat_matrix.shape}")
    print(f"  Models: {cat_matrix.shape[1]}")
    print(f"  Categories: {list(cat_matrix.index)}")

    # Print top models by Overall
    top_models = df.nlargest(10, "Overall")[["Models", "Overall"] + categories]
    print("\n  Top 10 models by Overall accuracy:")
    for _, row in top_models.iterrows():
        print(f"    {row['Models']:40s} {row['Overall']:.4f}")

    return cat_matrix, df


# ──────────────────────────────────────────────────────────────────────
# PART 3: Build model summary
# ──────────────────────────────────────────────────────────────────────
def build_model_summary(response_matrix, q_meta_df, leaderboard_df):
    """Build comprehensive model summary from all data sources."""
    print("\n" + "=" * 70)
    print("PART 3: Building model summary")
    print("=" * 70)

    summaries = []

    # --- Per-question models (48 models) ---
    if response_matrix is not None and q_meta_df is not None:
        print("  Processing per-question models...")
        categories = q_meta_df["category"].unique()

        for model_name in response_matrix.columns:
            row = {"model": model_name, "source": "github_eval_results"}

            # Overall accuracy
            scores = response_matrix[model_name]
            row["overall_accuracy"] = scores.mean()
            row["n_questions"] = int(scores.notna().sum())
            row["n_correct"] = int(scores.sum())

            # Per-category accuracy
            for cat in sorted(categories):
                cat_qids = q_meta_df[q_meta_df["category"] == cat].index
                cat_scores = scores.loc[scores.index.isin(cat_qids)]
                if len(cat_scores) > 0:
                    row[f"cat_{cat}"] = cat_scores.mean()

            summaries.append(row)

    # --- Leaderboard models (249 models) ---
    if leaderboard_df is not None:
        print("  Processing leaderboard models...")

        # Get list of models already added from per-question data
        existing_models = {s["model"] for s in summaries}

        lb_categories = [
            "Biology", "Business", "Chemistry", "Computer Science", "Economics",
            "Engineering", "Health", "History", "Law", "Math",
            "Philosophy", "Physics", "Psychology", "Other"
        ]

        for _, lrow in leaderboard_df.iterrows():
            model_name = lrow["Models"]
            row = {
                "model": model_name,
                "source": "tiger_leaderboard",
                "overall_accuracy": lrow.get("Overall", np.nan),
                "model_size_b": lrow.get("Model Size(B)", ""),
                "data_source": lrow.get("Data Source", ""),
            }
            for cat in lb_categories:
                val = pd.to_numeric(lrow.get(cat), errors="coerce")
                row[f"cat_{cat.lower().replace(' ', '_')}"] = val

            summaries.append(row)

    # --- Open LLM Leaderboard models (aggregate only) ---
    ollm_path = RAW_DIR / "open_llm_leaderboard_mmlu_pro.csv"
    if ollm_path.exists():
        print("  Processing Open LLM Leaderboard models...")
        ollm_df = pd.read_csv(ollm_path)
        for _, orow in ollm_df.iterrows():
            row = {
                "model": orow["fullname"],
                "source": "open_llm_leaderboard",
                "overall_accuracy": orow.get("MMLU-PRO Raw", np.nan),
            }
            summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    print(f"\n  Total model entries in summary: {len(summary_df)}")
    print(f"  By source:")
    print(summary_df["source"].value_counts().to_string())

    return summary_df


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("MMLU-Pro Response Matrix Builder")
    print("================================\n")

    # Part 1: Per-question response matrix
    response_matrix, q_meta_df = build_per_question_matrix()

    # Part 2: Per-category matrix from leaderboard
    cat_result = build_per_category_matrix()
    cat_matrix = None
    leaderboard_df = None
    if cat_result is not None:
        cat_matrix, leaderboard_df = cat_result

    # Part 3: Model summary
    summary_df = build_model_summary(response_matrix, q_meta_df, leaderboard_df)

    # ── Save outputs ──
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    if response_matrix is not None:
        out_path = PROCESSED_DIR / "response_matrix.csv"
        response_matrix.to_csv(out_path)
        print(f"  Saved per-question response matrix: {out_path}")
        print(f"    Shape: {response_matrix.shape} "
              f"({response_matrix.shape[0]} questions x {response_matrix.shape[1]} models)")

        # Also save as compressed for efficiency
        out_path_gz = PROCESSED_DIR / "response_matrix.csv.gz"
        response_matrix.to_csv(out_path_gz, compression="gzip")
        print(f"  Saved compressed: {out_path_gz}")

    if q_meta_df is not None:
        out_path = PROCESSED_DIR / "question_metadata.csv"
        q_meta_df.to_csv(out_path)
        print(f"  Saved question metadata: {out_path}")
        print(f"    Shape: {q_meta_df.shape}")

    if cat_matrix is not None:
        out_path = PROCESSED_DIR / "response_matrix_category.csv"
        cat_matrix.to_csv(out_path)
        print(f"  Saved per-category response matrix: {out_path}")
        print(f"    Shape: {cat_matrix.shape} "
              f"({cat_matrix.shape[0]} categories x {cat_matrix.shape[1]} models)")

    out_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved model summary: {out_path}")
    print(f"    Total entries: {len(summary_df)}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    if response_matrix is not None:
        print(f"\n  Per-Question Response Matrix:")
        print(f"    Questions: {response_matrix.shape[0]}")
        print(f"    Models: {response_matrix.shape[1]}")
        print(f"    Density: {response_matrix.notna().mean().mean():.3f}")
        print(f"    Overall accuracy (mean across models): "
              f"{response_matrix.mean().mean():.4f}")
        print(f"    Model accuracy range: "
              f"{response_matrix.mean().min():.4f} - {response_matrix.mean().max():.4f}")
        print(f"    Question difficulty range: "
              f"{response_matrix.mean(axis=1).min():.4f} - "
              f"{response_matrix.mean(axis=1).max():.4f}")

        # Category-level stats from per-question data
        if q_meta_df is not None:
            print(f"\n    Per-category accuracy (averaged across {response_matrix.shape[1]} models):")
            for cat in sorted(q_meta_df["category"].unique()):
                cat_qids = q_meta_df[q_meta_df["category"] == cat].index
                cat_scores = response_matrix.loc[
                    response_matrix.index.isin(cat_qids)
                ].mean()
                mean_acc = cat_scores.mean()
                n_qs = len(cat_qids)
                print(f"      {cat:20s}: {mean_acc:.4f} ({n_qs} questions)")

    if cat_matrix is not None:
        print(f"\n  Per-Category Leaderboard Matrix:")
        print(f"    Categories: {cat_matrix.shape[0]}")
        print(f"    Models: {cat_matrix.shape[1]}")
        print(f"    Density: {cat_matrix.notna().mean().mean():.3f}")

    print(f"\n  Model Summary:")
    print(f"    Total entries: {len(summary_df)}")
    for src in summary_df["source"].unique():
        n = (summary_df["source"] == src).sum()
        acc_col = summary_df.loc[summary_df["source"] == src, "overall_accuracy"]
        print(f"    {src:25s}: {n:5d} models, "
              f"accuracy range: {acc_col.min():.4f} - {acc_col.max():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
