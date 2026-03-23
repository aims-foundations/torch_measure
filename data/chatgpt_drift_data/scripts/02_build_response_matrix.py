#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process ChatGPT/LLM Drift evaluation data.

Loads evaluation CSVs from raw/LLMDrift/generation/ which contain model responses
across tasks and time periods. Each CSV has: model, date, query, ref_answer, answer,
along with latency and cost metadata.

Builds:
  1. model x task accuracy summary
  2. Per-task accuracy comparison across models
  3. Combined clean evaluation CSV

The original LLMDrift paper (Chen et al., 2023) tracked GPT-3.5 and GPT-4
across different API versions (0301/0314 vs 0613) to measure performance drift.

Saves outputs to processed/.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_DIR = RAW_DIR / "LLMDrift"
GENERATION_DIR = DRIFT_DIR / "generation"


# ---------------------------------------------------------------------------
# Task-specific accuracy functions
# ---------------------------------------------------------------------------

def score_prime(row: pd.Series) -> float | None:
    """Score prime number identification: check if answer contains [Yes] or [No] matching ref."""
    answer = str(row.get("answer", "")).lower()
    ref = str(row.get("ref_answer", "")).lower().strip()

    if ref in ("yes", "true", "1"):
        # Correct if answer contains [yes]
        return 1.0 if "[yes]" in answer else 0.0
    elif ref in ("no", "false", "0"):
        return 1.0 if "[no]" in answer else 0.0
    return None


def score_usmle(row: pd.Series) -> float | None:
    """Score USMLE: check if correct option letter appears in answer."""
    answer = str(row.get("answer", ""))
    ref = str(row.get("ref_answer", "")).strip()
    if not ref:
        return None
    # Reference is typically a letter like (C) or just C
    ref_letter = ref.strip("() ").upper()
    # Look for "The answer is (X)" pattern or just the letter
    pattern = rf"the answer is \(?{re.escape(ref_letter)}\)?"
    if re.search(pattern, answer, re.IGNORECASE):
        return 1.0
    # Fallback: check if the ref answer letter appears prominently
    if f"({ref_letter})" in answer.upper():
        return 1.0
    return 0.0


def score_happy_number(row: pd.Series) -> float | None:
    """Score happy number task: check yes/no match."""
    answer = str(row.get("answer", "")).lower()
    ref = str(row.get("ref_answer", "")).lower().strip()
    if ref in ("yes", "happy", "true", "1"):
        return 1.0 if any(w in answer for w in ["yes", "happy", "[yes]"]) else 0.0
    elif ref in ("no", "not happy", "false", "0"):
        return 1.0 if any(w in answer for w in ["no", "not happy", "[no]"]) else 0.0
    return None


def score_generic(row: pd.Series) -> float | None:
    """Generic scoring: check if ref_answer appears in the model answer."""
    answer = str(row.get("answer", "")).lower()
    ref = str(row.get("ref_answer", "")).lower().strip()
    if not ref or ref == "nan":
        return None
    return 1.0 if ref in answer else 0.0


TASK_SCORERS = {
    "prime": score_prime,
    "composite": score_prime,  # Same format as prime
    "usmle": score_usmle,
    "usmlefullzeroshot": score_usmle,
    "countahappynumber": score_happy_number,
    "happynumber": score_happy_number,
}


def score_row(row: pd.Series) -> float | None:
    """Score a single row based on its dataset/task type."""
    dataset = str(row.get("dataset", "")).lower().strip()
    scorer = TASK_SCORERS.get(dataset, score_generic)
    return scorer(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ChatGPT / LLM Drift Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover and load evaluation files
    # ------------------------------------------------------------------
    if not GENERATION_DIR.exists():
        print(f"[WARN] Generation directory not found: {GENERATION_DIR}")
        print("       Searching for evaluation files elsewhere...")
        # Try alternative locations
        eval_files = sorted(RAW_DIR.rglob("*EVAL*.csv")) + sorted(RAW_DIR.rglob("*eval*.csv"))
        if not eval_files:
            print("[ERROR] No evaluation files found.")
            pd.DataFrame({"status": ["no_eval_data"]}).to_csv(
                PROCESSED_DIR / "summary_statistics.csv", index=False
            )
            return
    else:
        eval_files = sorted(GENERATION_DIR.glob("*_EVAL.csv"))

    print(f"\nFound {len(eval_files)} evaluation files:")
    for f in eval_files:
        print(f"  - {f.name}")

    # ------------------------------------------------------------------
    # 2. Load all evaluation files
    # ------------------------------------------------------------------
    frames = []
    for f in eval_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            task_name = f.stem.replace("_EVAL", "").replace("_eval", "")
            df["_task_file"] = task_name
            frames.append(df)
            print(f"\n  {f.name}: shape={df.shape}")
            print(f"    Columns: {df.columns.tolist()}")
            if "model" in df.columns:
                print(f"    Models: {df['model'].unique().tolist()}")
            if "dataset" in df.columns:
                print(f"    Datasets: {df['dataset'].unique().tolist()}")
            if "date" in df.columns:
                print(f"    Dates: {sorted(df['date'].unique().tolist())}")
        except Exception as e:
            print(f"  [WARN] Failed to load {f.name}: {e}")

    if not frames:
        print("[ERROR] No evaluation data loaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")

    # ------------------------------------------------------------------
    # 3. Explore data structure
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    print("DATA STRUCTURE")
    print("=" * 70)
    print(f"Columns: {combined.columns.tolist()}")

    # Detect key columns
    col_model = None
    for cand in ["model", "Model", "model_name"]:
        if cand in combined.columns:
            col_model = cand
            break

    col_dataset = None
    for cand in ["dataset", "task", "benchmark"]:
        if cand in combined.columns:
            col_dataset = cand
            break

    col_date = None
    for cand in ["date", "timestamp", "eval_date"]:
        if cand in combined.columns:
            col_date = cand
            break

    col_answer = None
    for cand in ["answer", "response", "model_output", "prediction"]:
        if cand in combined.columns:
            col_answer = cand
            break

    col_ref = None
    for cand in ["ref_answer", "reference", "ground_truth", "label", "correct_answer"]:
        if cand in combined.columns:
            col_ref = cand
            break

    print(f"\nDetected columns:")
    print(f"  model:     {col_model}")
    print(f"  dataset:   {col_dataset}")
    print(f"  date:      {col_date}")
    print(f"  answer:    {col_answer}")
    print(f"  reference: {col_ref}")

    if col_model:
        print(f"\nModels ({combined[col_model].nunique()}):")
        print(combined[col_model].value_counts().to_string())

    if col_dataset:
        print(f"\nDatasets/Tasks ({combined[col_dataset].nunique()}):")
        print(combined[col_dataset].value_counts().to_string())

    # ------------------------------------------------------------------
    # 4. Score responses
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCORING RESPONSES")
    print("=" * 70)

    if col_answer and col_ref:
        combined["_score"] = combined.apply(score_row, axis=1)
        scored = combined.dropna(subset=["_score"])
        print(f"\n  Scored {len(scored)} / {len(combined)} responses")
        print(f"  Overall accuracy: {scored['_score'].mean():.4f}")

        # Check for task-specific extra columns (e.g., Directly Usable for leetcode)
        for extra_col in ["Directly Usable", "Response Rate", "Code_Submit"]:
            if extra_col in combined.columns:
                print(f"\n  Extra column '{extra_col}':")
                print(combined[extra_col].describe().to_string())
    else:
        print("\n  [WARN] Cannot score: missing answer or reference columns.")
        scored = combined
        scored["_score"] = np.nan

    # ------------------------------------------------------------------
    # 5. Build model x task accuracy matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING MODEL x TASK ACCURACY MATRIX")
    print("=" * 70)

    if col_model and col_dataset and "_score" in scored.columns:
        # Clean model names for readability
        def clean_model_name(name):
            return str(name).replace("openaichat/", "").strip()

        scored["_model_clean"] = scored[col_model].apply(clean_model_name)

        # Model x Task accuracy
        accuracy_matrix = scored.pivot_table(
            index="_model_clean",
            columns=col_dataset,
            values="_score",
            aggfunc="mean",
        )
        print(f"\n  Model x Task accuracy matrix: {accuracy_matrix.shape}")
        print(accuracy_matrix.round(4).to_string())

        out_path = PROCESSED_DIR / "model_x_task_accuracy.csv"
        accuracy_matrix.to_csv(out_path)
        print(f"  Saved to: {out_path}")

        # Count matrix (number of items per cell)
        count_matrix = scored.pivot_table(
            index="_model_clean",
            columns=col_dataset,
            values="_score",
            aggfunc="count",
        )
        out_path = PROCESSED_DIR / "model_x_task_counts.csv"
        count_matrix.to_csv(out_path)
        print(f"\n  Count matrix: {count_matrix.shape}")
        print(f"  Saved to: {out_path}")

        # Standard response matrix format: model x task_item
        # Create a unique item ID per task+question
        if "id" in scored.columns:
            scored["_item_id"] = scored[col_dataset].astype(str) + "_" + scored["id"].astype(str)
        else:
            scored["_item_id"] = scored[col_dataset].astype(str) + "_" + scored.groupby(col_dataset).cumcount().astype(str)

        response_matrix = scored.pivot_table(
            index="_model_clean",
            columns="_item_id",
            values="_score",
            aggfunc="first",
        )
        out_path = PROCESSED_DIR / "response_matrix.csv"
        response_matrix.to_csv(out_path)
        print(f"\n  Response matrix (model x item): {response_matrix.shape}")
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 6. Model version comparison (drift analysis)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS: API VERSION COMPARISON")
    print("=" * 70)

    if col_model and "_score" in scored.columns:
        # Group by model family (GPT-3.5 vs GPT-4) and version (0301/0314 vs 0613)
        def extract_family_version(model_name: str) -> tuple[str, str]:
            name = clean_model_name(model_name)
            if "gpt-4" in name.lower():
                family = "GPT-4"
            elif "gpt-3.5" in name.lower():
                family = "GPT-3.5"
            else:
                family = name
            # Extract version suffix
            version = name.split("-")[-1] if "-" in name else "unknown"
            return family, version

        scored[["_family", "_version"]] = scored[col_model].apply(
            lambda x: pd.Series(extract_family_version(x))
        )

        # Accuracy by family + version + task
        drift = scored.groupby(["_family", "_version", col_dataset])["_score"].agg(
            ["mean", "count", "std"]
        ).round(4)
        drift.columns = ["accuracy", "n_items", "std"]
        print("\n  Accuracy by model family, version, and task:")
        print(drift.to_string())

        out_path = PROCESSED_DIR / "drift_by_version_task.csv"
        drift.to_csv(out_path)
        print(f"  Saved to: {out_path}")

        # Pivot: rows = (family, version), columns = task
        drift_pivot = scored.pivot_table(
            index=["_family", "_version"],
            columns=col_dataset,
            values="_score",
            aggfunc="mean",
        )
        out_path = PROCESSED_DIR / "drift_pivot.csv"
        drift_pivot.to_csv(out_path)
        print(f"\n  Drift pivot table:")
        print(drift_pivot.round(4).to_string())
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 7. Save clean combined data
    # ------------------------------------------------------------------
    keep_cols = [c for c in [
        col_model, col_dataset, col_date, "id", col_ref, "_score", "_task_file",
        "latency", "cost", "answer_token",
    ] if c is not None and c in combined.columns]

    if keep_cols:
        clean = combined[keep_cols].copy()
        out_path = PROCESSED_DIR / "evaluations_clean.csv"
        clean.to_csv(out_path, index=False)
        print(f"\n  Clean evaluations CSV: {clean.shape}")
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 8. Summary statistics
    # ------------------------------------------------------------------
    stats = {
        "metric": ["total_evaluations", "evaluation_files"],
        "value": [len(combined), len(eval_files)],
    }
    if col_model:
        stats["metric"].append("unique_models")
        stats["value"].append(combined[col_model].nunique())
    if col_dataset:
        stats["metric"].append("unique_tasks")
        stats["value"].append(combined[col_dataset].nunique())
    if "_score" in scored.columns:
        stats["metric"].append("overall_accuracy")
        stats["value"].append(round(scored["_score"].mean(), 4))
        stats["metric"].append("scored_responses")
        stats["value"].append(len(scored))

    stats_df = pd.DataFrame(stats)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics:")
    print(stats_df.to_string(index=False))
    print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("LLM Drift processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
