"""
Build MathArena response matrices from HuggingFace per-model per-problem output data.

Data sources:
  MathArena HuggingFace datasets (https://huggingface.co/MathArena):
  - 27 output datasets covering AIME 2025/2026, HMMT, BRUMO, CMIMC, SMT,
    APEX, ArXivMath, Kangaroo, IMO, IMC, USAMO, Putnam, Miklos competitions
  - Each dataset has ~62 models evaluated 4 times per problem
  - Final-answer competitions have a `correct` boolean field
  - Proof-based competitions have `points_judge_1/2` fields (0-7 scale)

Outputs:
  For each final-answer competition:
    - response_matrix_{comp}.csv:       Average correctness across 4 attempts
    - response_matrix_{comp}_binary.csv: Majority-vote binary (>=2/4 correct)
    - response_matrix_{comp}_raw.csv:    All attempts (model_attempt x problem)
  For proof-based competitions:
    - response_matrix_{comp}_points.csv: Average points (normalized 0-1)
  Combined:
    - response_matrix_aime_combined.csv:     AIME 2025+2026 combined
    - response_matrix_all_final_answer.csv:  All final-answer competitions merged
    - model_summary.csv:                     Per-model statistics across all comps

GitHub: https://github.com/eth-sri/matharena
Website: https://matharena.ai
Paper: "MathArena: Evaluating LLMs on Uncontaminated Math Competitions" (NeurIPS D&B 2025)
"""

import os
import sys
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Final-answer competitions (have `correct` boolean column)
FINAL_ANSWER_DATASETS = {
    "aime_2025": "MathArena/aime_2025_outputs",
    "aime_2025_I": "MathArena/aime_2025_I_outputs",
    "aime_2025_II": "MathArena/aime_2025_II_outputs",
    "aime_2026": "MathArena/aime_2026_outputs",
    "aime_2026_I": "MathArena/aime_2026_I_outputs",
    "hmmt_feb_2025": "MathArena/hmmt_feb_2025_outputs",
    "hmmt_feb_2026": "MathArena/hmmt_feb_2026_outputs",
    "hmmt_nov_2025": "MathArena/hmmt_nov_2025_outputs",
    "brumo_2025": "MathArena/brumo_2025_outputs",
    "cmimc_2025": "MathArena/cmimc_2025_outputs",
    "smt_2025": "MathArena/smt_2025_outputs",
    "apex_2025": "MathArena/apex_2025_outputs",
    "apex_shortlist": "MathArena/apex_shortlist_outputs",
    "arxivmath_1225": "MathArena/arxivmath-1225_outputs",
    "arxivmath_0126": "MathArena/arxivmath-0126_outputs",
    "arxivmath_0226": "MathArena/arxivmath-0226_outputs",
    "kangaroo_2025_1_2": "MathArena/kangaroo_2025_1-2_outputs",
    "kangaroo_2025_3_4": "MathArena/kangaroo_2025_3-4_outputs",
    "kangaroo_2025_5_6": "MathArena/kangaroo_2025_5-6_outputs",
    "kangaroo_2025_7_8": "MathArena/kangaroo_2025_7-8_outputs",
    "kangaroo_2025_9_10": "MathArena/kangaroo_2025_9-10_outputs",
    "kangaroo_2025_11_12": "MathArena/kangaroo_2025_11-12_outputs",
}

# Proof-based competitions (have points_judge_1/2 columns, no `correct`)
PROOF_DATASETS = {
    "usamo_2025": "MathArena/usamo_2025_outputs",
    "imo_2025": "MathArena/imo_2025_outputs",
    "imc_2025": "MathArena/imc_2025_outputs",
    "putnam_2025": "MathArena/putnam_2025_outputs",
    "miklos_2025": "MathArena/miklos_2025_outputs",
}

# Primary datasets for combined matrices (avoid overlap with split versions)
AIME_PRIMARY = ["aime_2025", "aime_2026"]
PRIMARY_FINAL_ANSWER = [
    "aime_2025", "aime_2026",
    "hmmt_feb_2025", "hmmt_feb_2026", "hmmt_nov_2025",
    "brumo_2025", "cmimc_2025", "smt_2025",
    "apex_2025", "apex_shortlist",
]


def download_dataset(dataset_id, comp_name):
    """Download a HuggingFace dataset and cache as parquet in raw/."""
    cache_path = os.path.join(RAW_DIR, f"{comp_name}.parquet")
    if os.path.exists(cache_path):
        print(f"  [cached] {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading {dataset_id} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train")
        # Select only columns we need to keep file sizes manageable
        keep_cols = [
            "problem_idx", "model_name", "idx_answer", "correct",
            "gold_answer", "parsed_answer", "cost",
            "input_tokens", "output_tokens",
            # proof-based columns
            "points_judge_1", "max_points_judge_1",
            "points_judge_2", "max_points_judge_2",
        ]
        available_cols = [c for c in keep_cols if c in ds.column_names]
        ds_small = ds.select_columns(available_cols)
        df = ds_small.to_pandas()
        df.to_parquet(cache_path, index=False)
        print(f"  Saved to {cache_path} ({len(df)} rows, "
              f"{os.path.getsize(cache_path)/1024:.1f} KB)")
        return df
    except Exception as e:
        print(f"  ERROR downloading {dataset_id}: {e}")
        return None


def build_final_answer_matrices(df, comp_name, comp_label):
    """Build response matrices for a final-answer competition.

    Each model is evaluated 4 times per problem (idx_answer 0-3).
    We produce:
      1. Average accuracy matrix (float): mean of correct across attempts
      2. Binary majority-vote matrix: 1 if >=2/4 attempts correct
      3. Raw matrix: all attempts as separate rows
    """
    if df is None or "correct" not in df.columns:
        print(f"  SKIP {comp_label}: no 'correct' column")
        return None

    # Ensure correct is boolean/int
    df["correct"] = df["correct"].astype(int)

    # Basic stats
    models = sorted(df["model_name"].unique())
    problems = sorted(df["problem_idx"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    n_models = len(models)
    n_problems = len(problems)
    attempts_per = df.groupby(["model_name", "problem_idx"])["idx_answer"].nunique()

    print(f"\n{'='*65}")
    print(f"  {comp_label}")
    print(f"{'='*65}")
    print(f"  Models:            {n_models}")
    print(f"  Problems:          {n_problems}")
    print(f"  Total rows:        {len(df):,}")
    print(f"  Attempts/model/prob: {attempts_per.min()}-{attempts_per.max()} "
          f"(median {attempts_per.median():.0f})")

    # --- 1) Average accuracy matrix ---
    avg_df = (
        df.groupby(["model_name", "problem_idx"])["correct"]
        .mean()
        .reset_index()
        .pivot(index="model_name", columns="problem_idx", values="correct")
    )
    avg_df = avg_df[problems]  # ensure column order
    avg_df = avg_df.loc[models]  # ensure row order

    matrix = avg_df.values
    total_cells = n_models * n_problems
    fill_rate = 1.0 - np.isnan(matrix).sum() / total_cells
    mean_acc = np.nanmean(matrix)

    print(f"  Matrix dims:       {n_models} x {n_problems}")
    print(f"  Fill rate:         {fill_rate*100:.1f}%")
    print(f"  Mean accuracy:     {mean_acc*100:.1f}%")

    # Per-model stats
    per_model = np.nanmean(matrix, axis=1)
    best_idx = np.nanargmax(per_model)
    worst_idx = np.nanargmin(per_model)
    print(f"\n  Per-model accuracy:")
    print(f"    Best:   {per_model[best_idx]*100:.1f}% ({models[best_idx]})")
    print(f"    Worst:  {per_model[worst_idx]*100:.1f}% ({models[worst_idx]})")
    print(f"    Median: {np.nanmedian(per_model)*100:.1f}%")
    print(f"    Std:    {np.nanstd(per_model)*100:.1f}%")

    # Per-problem stats
    per_prob = np.nanmean(matrix, axis=0)
    print(f"\n  Per-problem solve rate:")
    print(f"    Min:    {np.nanmin(per_prob)*100:.1f}%")
    print(f"    Max:    {np.nanmax(per_prob)*100:.1f}%")
    print(f"    Median: {np.nanmedian(per_prob)*100:.1f}%")
    print(f"    Std:    {np.nanstd(per_prob)*100:.1f}%")

    # Difficulty distribution
    unsolved = np.sum(per_prob == 0)
    hard = np.sum(per_prob < 0.1)
    easy = np.sum(per_prob > 0.9)
    print(f"\n  Problem difficulty distribution:")
    print(f"    Unsolved (0%):   {unsolved}")
    print(f"    Hard (<10%):     {hard}")
    print(f"    Easy (>90%):     {easy}")

    # Save average accuracy matrix
    out_avg = os.path.join(PROCESSED_DIR, f"response_matrix_{comp_name}.csv")
    avg_df.to_csv(out_avg)
    print(f"\n  Saved: {out_avg}")

    # --- 2) Binary majority-vote matrix ---
    binary_df = (avg_df >= 0.5).astype(int)
    out_bin = os.path.join(PROCESSED_DIR, f"response_matrix_{comp_name}_binary.csv")
    binary_df.to_csv(out_bin)
    print(f"  Saved: {out_bin}")

    binary_matrix = binary_df.values
    print(f"\n  Majority-vote binary matrix:")
    print(f"    Pass cells:  {int(binary_matrix.sum()):,} "
          f"({binary_matrix.sum()/total_cells*100:.1f}%)")
    print(f"    Fail cells:  {total_cells - int(binary_matrix.sum()):,} "
          f"({(1 - binary_matrix.sum()/total_cells)*100:.1f}%)")

    # --- 3) Raw attempts matrix ---
    raw_df = df.copy()
    raw_df["model_attempt"] = (
        raw_df["model_name"] + "_attempt" + raw_df["idx_answer"].astype(str)
    )
    raw_pivot = (
        raw_df.pivot(index="model_attempt", columns="problem_idx", values="correct")
    )
    raw_pivot = raw_pivot[problems]
    out_raw = os.path.join(PROCESSED_DIR, f"response_matrix_{comp_name}_raw.csv")
    raw_pivot.to_csv(out_raw)
    print(f"  Saved: {out_raw}")

    return {
        "comp_name": comp_name,
        "comp_label": comp_label,
        "n_models": n_models,
        "n_problems": n_problems,
        "mean_accuracy": mean_acc,
        "fill_rate": fill_rate,
        "models": models,
        "per_model_acc": per_model,
        "avg_df": avg_df,
        "binary_df": binary_df,
    }


def build_proof_matrices(df, comp_name, comp_label):
    """Build response matrices for proof-based competitions.

    These have points_judge_1/2 instead of correct. We normalize to 0-1.
    """
    if df is None:
        print(f"  SKIP {comp_label}: no data")
        return None

    has_j1 = "points_judge_1" in df.columns
    has_j2 = "points_judge_2" in df.columns

    if not has_j1 and not has_j2:
        print(f"  SKIP {comp_label}: no points columns")
        return None

    # Compute normalized points (average of two judges, normalized by max)
    if has_j1 and has_j2:
        max_pts_1 = df.get("max_points_judge_1", pd.Series([7]*len(df)))
        max_pts_2 = df.get("max_points_judge_2", pd.Series([7]*len(df)))
        # Handle None/NaN in points
        p1 = pd.to_numeric(df["points_judge_1"], errors="coerce").fillna(0)
        p2 = pd.to_numeric(df["points_judge_2"], errors="coerce").fillna(0)
        m1 = pd.to_numeric(max_pts_1, errors="coerce").fillna(7)
        m2 = pd.to_numeric(max_pts_2, errors="coerce").fillna(7)
        df["norm_points"] = ((p1 / m1) + (p2 / m2)) / 2.0
    elif has_j1:
        p1 = pd.to_numeric(df["points_judge_1"], errors="coerce").fillna(0)
        m1 = pd.to_numeric(df.get("max_points_judge_1",
                                   pd.Series([7]*len(df))), errors="coerce").fillna(7)
        df["norm_points"] = p1 / m1
    else:
        p2 = pd.to_numeric(df["points_judge_2"], errors="coerce").fillna(0)
        m2 = pd.to_numeric(df.get("max_points_judge_2",
                                   pd.Series([7]*len(df))), errors="coerce").fillna(7)
        df["norm_points"] = p2 / m2

    df["norm_points"] = df["norm_points"].clip(0, 1)

    models = sorted(df["model_name"].unique())
    problems = sorted(df["problem_idx"].unique(),
                      key=lambda x: int(x) if str(x).isdigit() else x)
    n_models = len(models)
    n_problems = len(problems)

    print(f"\n{'='*65}")
    print(f"  {comp_label} (proof-based)")
    print(f"{'='*65}")
    print(f"  Models:       {n_models}")
    print(f"  Problems:     {n_problems}")
    print(f"  Total rows:   {len(df):,}")

    pts_df = (
        df.groupby(["model_name", "problem_idx"])["norm_points"]
        .mean()
        .reset_index()
        .pivot(index="model_name", columns="problem_idx", values="norm_points")
    )
    pts_df = pts_df.reindex(columns=problems, index=models)

    matrix = pts_df.values
    fill_rate = 1.0 - np.isnan(matrix).sum() / (n_models * n_problems)
    mean_pts = np.nanmean(matrix)

    print(f"  Matrix dims:  {n_models} x {n_problems}")
    print(f"  Fill rate:    {fill_rate*100:.1f}%")
    print(f"  Mean score:   {mean_pts*100:.1f}%")

    per_model = np.nanmean(matrix, axis=1)
    best_idx = np.nanargmax(per_model)
    print(f"\n  Per-model score:")
    print(f"    Best:   {per_model[best_idx]*100:.1f}% ({models[best_idx]})")
    print(f"    Median: {np.nanmedian(per_model)*100:.1f}%")

    out_pts = os.path.join(PROCESSED_DIR, f"response_matrix_{comp_name}_points.csv")
    pts_df.to_csv(out_pts)
    print(f"\n  Saved: {out_pts}")

    return {
        "comp_name": comp_name,
        "comp_label": comp_label,
        "n_models": n_models,
        "n_problems": n_problems,
        "mean_accuracy": mean_pts,
        "fill_rate": fill_rate,
        "models": models,
        "per_model_acc": per_model,
        "avg_df": pts_df,
    }


def build_combined_matrices(all_stats):
    """Build combined response matrices from multiple competitions."""

    # --- AIME combined (2025 + 2026) ---
    aime_stats = [s for s in all_stats if s and s["comp_name"] in AIME_PRIMARY]
    if len(aime_stats) >= 2:
        print(f"\n{'='*65}")
        print(f"  COMBINED: AIME (2025 + 2026)")
        print(f"{'='*65}")
        frames = []
        for s in aime_stats:
            df = s["avg_df"].copy()
            df.columns = [f"{s['comp_name']}_p{c}" for c in df.columns]
            frames.append(df)
        combined = pd.concat(frames, axis=1)
        # Only keep models present in all
        combined = combined.dropna(how="any")
        n_models = len(combined)
        n_problems = len(combined.columns)
        mean_acc = combined.values.mean()
        print(f"  Models (intersection): {n_models}")
        print(f"  Total problems:        {n_problems}")
        print(f"  Mean accuracy:         {mean_acc*100:.1f}%")
        out = os.path.join(PROCESSED_DIR, "response_matrix_aime_combined.csv")
        combined.to_csv(out)
        print(f"  Saved: {out}")

    # --- All primary final-answer competitions combined ---
    primary_stats = [s for s in all_stats
                     if s and s["comp_name"] in PRIMARY_FINAL_ANSWER]
    if len(primary_stats) >= 2:
        print(f"\n{'='*65}")
        print(f"  COMBINED: All Primary Final-Answer Competitions")
        print(f"{'='*65}")
        frames = []
        for s in primary_stats:
            df = s["avg_df"].copy()
            df.columns = [f"{s['comp_name']}_p{c}" for c in df.columns]
            frames.append(df)
        combined = pd.concat(frames, axis=1)
        # Keep all models (allow NaN for competitions they weren't evaluated on)
        n_models = len(combined)
        n_problems = len(combined.columns)
        fill = 1.0 - combined.isna().sum().sum() / (n_models * n_problems)
        mean_acc = combined.values[~np.isnan(combined.values)].mean()
        print(f"  Models (union):  {n_models}")
        print(f"  Total problems:  {n_problems}")
        print(f"  Fill rate:       {fill*100:.1f}%")
        print(f"  Mean accuracy:   {mean_acc*100:.1f}%")
        out = os.path.join(PROCESSED_DIR, "response_matrix_all_final_answer.csv")
        combined.to_csv(out)
        print(f"  Saved: {out}")


def build_model_summary(all_stats):
    """Build a comprehensive model summary CSV."""
    # Collect all unique models
    model_data = defaultdict(dict)
    for s in all_stats:
        if s is None:
            continue
        for i, model in enumerate(s["models"]):
            model_data[model][s["comp_name"]] = s["per_model_acc"][i]

    rows = []
    for model in sorted(model_data.keys()):
        row = {"model": model}
        comps = model_data[model]
        row["n_competitions"] = len(comps)

        # Individual competition scores
        for comp_name in comps:
            row[f"acc_{comp_name}"] = comps[comp_name]

        # Overall mean across competitions
        scores = list(comps.values())
        row["mean_acc_all"] = np.mean(scores) if scores else None

        # AIME-specific scores
        aime_scores = [v for k, v in comps.items() if k.startswith("aime")]
        if aime_scores:
            row["mean_acc_aime"] = np.mean(aime_scores)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("mean_acc_all", ascending=False, na_position="last")

    out = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(out, index=False)

    print(f"\n{'='*65}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*65}")
    print(f"  Total unique models: {len(summary_df)}")

    # Competition coverage
    acc_cols = [c for c in summary_df.columns if c.startswith("acc_")]
    for col in sorted(acc_cols):
        n = summary_df[col].notna().sum()
        comp = col.replace("acc_", "")
        print(f"    {comp:30s}: {n} models")

    # Top 15 models overall
    print(f"\n  Top 15 models (by mean accuracy across all competitions):")
    top = summary_df.dropna(subset=["mean_acc_all"]).head(15)
    for _, r in top.iterrows():
        acc = r["mean_acc_all"] * 100
        nc = r["n_competitions"]
        aime = r.get("mean_acc_aime")
        aime_str = f"{aime*100:.1f}%" if pd.notna(aime) else "N/A"
        print(f"    {r['model']:45s}  overall={acc:.1f}%  "
              f"aime={aime_str}  comps={int(nc)}")

    print(f"\n  Saved: {out}")
    return summary_df


def main():
    print("=" * 65)
    print("  MathArena Response Matrix Builder")
    print("  Data: https://huggingface.co/MathArena")
    print("=" * 65)

    all_stats = []

    # -----------------------------------------------------------------------
    # Process final-answer competitions
    # -----------------------------------------------------------------------
    print(f"\n  Processing {len(FINAL_ANSWER_DATASETS)} final-answer competitions...")
    for comp_name, dataset_id in sorted(FINAL_ANSWER_DATASETS.items()):
        df = download_dataset(dataset_id, comp_name)
        if df is not None:
            label = comp_name.replace("_", " ").title()
            stats = build_final_answer_matrices(df, comp_name, label)
            if stats:
                all_stats.append(stats)

    # -----------------------------------------------------------------------
    # Process proof-based competitions
    # -----------------------------------------------------------------------
    print(f"\n  Processing {len(PROOF_DATASETS)} proof-based competitions...")
    for comp_name, dataset_id in sorted(PROOF_DATASETS.items()):
        df = download_dataset(dataset_id, comp_name)
        if df is not None:
            label = comp_name.replace("_", " ").title()
            stats = build_proof_matrices(df, comp_name, label)
            if stats:
                all_stats.append(stats)

    # -----------------------------------------------------------------------
    # Combined matrices
    # -----------------------------------------------------------------------
    build_combined_matrices(all_stats)

    # -----------------------------------------------------------------------
    # Model summary
    # -----------------------------------------------------------------------
    build_model_summary(all_stats)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"\n  Competitions processed: {len(all_stats)}")
    total_models = len(set(m for s in all_stats if s for m in s["models"]))
    total_problems = sum(s["n_problems"] for s in all_stats if s)
    print(f"  Total unique models:   {total_models}")
    print(f"  Total problems:        {total_problems}")

    # AIME 2025 as primary
    aime25 = [s for s in all_stats if s and s["comp_name"] == "aime_2025"]
    if aime25:
        s = aime25[0]
        print(f"\n  PRIMARY response matrix (AIME 2025):")
        print(f"    Dimensions: {s['n_models']} models x {s['n_problems']} problems")
        print(f"    Fill rate:  {s['fill_rate']*100:.1f}%")
        print(f"    Mean acc:   {s['mean_accuracy']*100:.1f}%")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:55s}  {size_kb:.1f} KB")

    print(f"\n  Raw cache files:")
    for f in sorted(os.listdir(RAW_DIR)):
        fpath = os.path.join(RAW_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:55s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
