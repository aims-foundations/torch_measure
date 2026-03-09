"""
Build BigCodeBench response matrices from HuggingFace per-model per-task data.

Data sources:
  - bigcode/bigcodebench-perf: Per-model per-task pass/fail for Full benchmark (1,140 tasks)
    - "complete" split: 153 models evaluated on code completion
    - "instruct" split: 126 models evaluated on instruction following
  - bigcode/bigcodebench-hard-perf: Per-model per-task pass/fail for Hard subset (148 tasks)
    - "complete" split: 199 models
    - "instruct" split: 173 models
  - bigcode/bigcodebench-results: Aggregate scores + model metadata (202 models)
  - results_full.json / results_hard.json: Website leaderboard data with model metadata

Outputs:
  - response_matrix.csv: Binary (models x tasks) matrix for Full-Complete variant
  - response_matrix_instruct.csv: Binary matrix for Full-Instruct variant
  - response_matrix_hard_complete.csv: Binary matrix for Hard-Complete variant
  - response_matrix_hard_instruct.csv: Binary matrix for Hard-Instruct variant
  - model_summary.csv: Per-model statistics across all variants
"""

import os
import json
import pandas as pd
import numpy as np

# Paths
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_perf_matrix(csv_path):
    """Load a per-model per-task performance CSV into a clean response matrix.

    Returns:
        model_names: list of model names
        task_ids: list of task IDs (sorted)
        matrix: numpy array of shape (n_models, n_tasks) with 0/1 values
        df: pandas DataFrame with Model as index and task_ids as columns
    """
    df = pd.read_csv(csv_path)
    model_col = "Model"

    # Separate model names from task columns
    model_names = df[model_col].tolist()
    task_cols = sorted([c for c in df.columns if c != model_col])

    # Build clean matrix
    matrix_df = df.set_index(model_col)[task_cols].copy()
    matrix_df = matrix_df.astype(int)

    return model_names, task_cols, matrix_df.values, matrix_df


def build_response_matrix(csv_path, output_name, variant_label):
    """Build and save a response matrix from raw perf CSV."""
    model_names, task_ids, matrix, matrix_df = load_perf_matrix(csv_path)

    n_models = len(model_names)
    n_tasks = len(task_ids)
    total_cells = n_models * n_tasks
    n_pass = int(matrix.sum())
    n_fail = total_cells - n_pass
    fill_rate = 1.0  # No NaN in this data
    mean_pass_rate = matrix.mean()

    print(f"\n{'='*60}")
    print(f"  {variant_label}")
    print(f"{'='*60}")
    print(f"  Models:          {n_models}")
    print(f"  Tasks:           {n_tasks}")
    print(f"  Matrix dims:     {n_models} x {n_tasks}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Pass cells:      {n_pass:,} ({n_pass/total_cells*100:.1f}%)")
    print(f"  Fail cells:      {n_fail:,} ({n_fail/total_cells*100:.1f}%)")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")
    print(f"  Mean pass rate:  {mean_pass_rate*100:.1f}%")

    # Per-model stats
    per_model_pass = matrix.mean(axis=1)
    print(f"\n  Per-model pass rate:")
    print(f"    Min:    {per_model_pass.min()*100:.1f}% ({model_names[per_model_pass.argmin()]})")
    print(f"    Max:    {per_model_pass.max()*100:.1f}% ({model_names[per_model_pass.argmax()]})")
    print(f"    Median: {np.median(per_model_pass)*100:.1f}%")
    print(f"    Std:    {per_model_pass.std()*100:.1f}%")

    # Per-task stats
    per_task_solve = matrix.mean(axis=0)
    print(f"\n  Per-task solve rate:")
    print(f"    Min:    {per_task_solve.min()*100:.1f}%")
    print(f"    Max:    {per_task_solve.max()*100:.1f}%")
    print(f"    Median: {np.median(per_task_solve)*100:.1f}%")
    print(f"    Std:    {per_task_solve.std()*100:.1f}%")

    # Count tasks by difficulty
    unsolved = (per_task_solve == 0).sum()
    easy = (per_task_solve > 0.9).sum()
    hard = (per_task_solve < 0.1).sum()
    print(f"\n  Task difficulty distribution:")
    print(f"    Unsolved (0%):   {unsolved}")
    print(f"    Hard (<10%):     {hard}")
    print(f"    Easy (>90%):     {easy}")

    # Save response matrix
    output_path = os.path.join(PROCESSED_DIR, output_name)
    matrix_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    return {
        "variant": variant_label,
        "n_models": n_models,
        "n_tasks": n_tasks,
        "mean_pass_rate": mean_pass_rate,
        "model_names": model_names,
        "per_model_pass": per_model_pass,
    }


def build_model_summary(all_stats, metadata_path, results_json_path):
    """Build a comprehensive model summary CSV combining all variants."""

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    metadata_dict = {}
    for _, row in metadata_df.iterrows():
        metadata_dict[row["model"]] = {
            "link": row.get("link", ""),
            "moe": row.get("moe", False),
            "size": row.get("size", None),
            "act_param": row.get("act_param", None),
            "type": row.get("type", ""),
            "date": row.get("date", ""),
            "prefill": row.get("prefill", False),
        }

    # Also load website JSON for additional metadata
    with open(results_json_path, "r") as f:
        website_results = json.load(f)
    for model_name, info in website_results.items():
        if model_name not in metadata_dict:
            metadata_dict[model_name] = {
                "link": info.get("link", ""),
                "moe": info.get("moe", False),
                "size": info.get("size", None),
                "act_param": info.get("act_param", None),
                "type": "instruct" if info.get("prompted", False) else "base",
                "date": info.get("date", ""),
                "prefill": info.get("prefill", False),
            }

    # Collect all unique models across variants
    all_models = set()
    variant_data = {}
    for stats in all_stats:
        variant = stats["variant"]
        for i, model in enumerate(stats["model_names"]):
            all_models.add(model)
            if model not in variant_data:
                variant_data[model] = {}
            variant_data[model][variant] = stats["per_model_pass"][i]

    # Build summary rows
    rows = []
    for model in sorted(all_models):
        row = {"model": model}

        # Add metadata
        meta = metadata_dict.get(model, {})
        row["link"] = meta.get("link", "")
        row["size_B"] = meta.get("size", None)
        row["act_param_B"] = meta.get("act_param", None)
        row["moe"] = meta.get("moe", False)
        row["type"] = meta.get("type", "")
        row["date"] = meta.get("date", "")

        # Add per-variant pass rates
        vd = variant_data.get(model, {})
        row["complete_pass_rate"] = vd.get("Full-Complete", None)
        row["instruct_pass_rate"] = vd.get("Full-Instruct", None)
        row["hard_complete_pass_rate"] = vd.get("Hard-Complete", None)
        row["hard_instruct_pass_rate"] = vd.get("Hard-Instruct", None)

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sort by complete pass rate (descending), with NaN at bottom
    summary_df = summary_df.sort_values("complete_pass_rate", ascending=False, na_position="last")

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total unique models: {len(summary_df)}")
    print(f"  Models with complete scores: {summary_df['complete_pass_rate'].notna().sum()}")
    print(f"  Models with instruct scores: {summary_df['instruct_pass_rate'].notna().sum()}")
    print(f"  Models with hard-complete scores: {summary_df['hard_complete_pass_rate'].notna().sum()}")
    print(f"  Models with hard-instruct scores: {summary_df['hard_instruct_pass_rate'].notna().sum()}")
    print(f"\n  Top 10 models (by complete pass@1):")
    top10 = summary_df.dropna(subset=["complete_pass_rate"]).head(10)
    for _, r in top10.iterrows():
        cpr = r["complete_pass_rate"] * 100
        ipr = r["instruct_pass_rate"] * 100 if pd.notna(r["instruct_pass_rate"]) else None
        ipr_str = f"{ipr:.1f}%" if ipr is not None else "N/A"
        print(f"    {r['model']:40s}  complete={cpr:.1f}%  instruct={ipr_str}")

    print(f"\n  Saved: {output_path}")
    return summary_df


def main():
    print("BigCodeBench Response Matrix Builder")
    print("=" * 60)

    # Define all variants to process
    variants = [
        {
            "csv": os.path.join(RAW_DIR, "bigcodebench_complete_perf.csv"),
            "output": "response_matrix.csv",
            "label": "Full-Complete",
        },
        {
            "csv": os.path.join(RAW_DIR, "bigcodebench_instruct_perf.csv"),
            "output": "response_matrix_instruct.csv",
            "label": "Full-Instruct",
        },
        {
            "csv": os.path.join(RAW_DIR, "bigcodebench_hard_complete_perf.csv"),
            "output": "response_matrix_hard_complete.csv",
            "label": "Hard-Complete",
        },
        {
            "csv": os.path.join(RAW_DIR, "bigcodebench_hard_instruct_perf.csv"),
            "output": "response_matrix_hard_instruct.csv",
            "label": "Hard-Instruct",
        },
    ]

    all_stats = []
    for v in variants:
        if os.path.exists(v["csv"]):
            stats = build_response_matrix(v["csv"], v["output"], v["label"])
            all_stats.append(stats)
        else:
            print(f"\nWARNING: {v['csv']} not found, skipping {v['label']}")

    # Build model summary
    metadata_path = os.path.join(RAW_DIR, "bigcodebench_results_metadata.csv")
    results_json_path = os.path.join(RAW_DIR, "results_full.json")
    summary_df = build_model_summary(all_stats, metadata_path, results_json_path)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  PRIMARY response matrix (Full-Complete):")
    primary = [s for s in all_stats if s["variant"] == "Full-Complete"][0]
    print(f"    Dimensions: {primary['n_models']} models x {primary['n_tasks']} tasks")
    print(f"    Fill rate:  100.0%")
    print(f"    Mean pass:  {primary['mean_pass_rate']*100:.1f}%")
    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
