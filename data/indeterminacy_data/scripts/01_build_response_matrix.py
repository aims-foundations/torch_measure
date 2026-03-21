"""
Build Indeterminacy Experiments response matrices from multi-judge LLM evaluation data.

Data source:
  - lguerdan/indeterminacy-experiments on HuggingFace
  - "Validating LLM-as-a-Judge under Rating Indeterminacy" (NeurIPS 2025)

Dataset structure:
  - 36 rows = 9 LLM judges x 4 task groups (rating configurations)
  - resp_table: [200 items, 3 scales, 4 categories, R repetitions]
    Binary indicators for forced-choice / multi-label ratings.
    R = 10 for most judges, 8 for o3-mini.
  - 7 meaningful (group, scale) combinations with >= 2 response categories

Judges (9):
  claude-3-5-sonnet-20241022, claude-3-haiku-20240307, deepseek-chat,
  Llama-3.3-70B-Instruct, mistral-large-latest, mistral-small-latest,
  gpt-3.5-turbo, gpt-4o-mini-2024-07-18, o3-mini

Response encoding:
  P(category 0) averaged across repetitions, giving continuous [0, 1] values.
  NaN for items without responses in a given task group.

Outputs:
  - response_matrix_all.csv: Combined 9 judges x 800 items (4 groups x 200)
  - response_matrix_group_{0,1,2,3}.csv: Per-group 9 judges x 200 items
  - judge_summary.csv: Per-judge aggregate statistics
"""

import os
import sys

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

N_JUDGES = 9
N_ITEMS = 200
N_GROUPS = 4


def download_dataset():
    """Download the indeterminacy-experiments dataset from HuggingFace."""
    print("Downloading lguerdan/indeterminacy-experiments from HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("lguerdan/indeterminacy-experiments", split="test")
        print(f"  Loaded {len(ds)} rows")
        return ds
    except ImportError:
        print("  ERROR: 'datasets' library not available. Install with:")
        print("    pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        sys.exit(1)


def build_group_matrix(ds, group_idx):
    """Build a (9 judges x 200 items) response matrix for one task group.

    For each (judge, item), finds the first active scale and computes
    P(category 0) = proportion of reps selecting category 0 among reps
    that have any selection.  Items with no active scale get NaN.
    """
    matrix = np.full((N_JUDGES, N_ITEMS), np.nan)

    for m in range(N_JUDGES):
        row_idx = group_idx * N_JUDGES + m
        rt = np.array(ds[row_idx]["resp_table"])  # [200, 3, 4, reps]

        for i in range(N_ITEMS):
            for s in range(3):
                item_scale = rt[i, s, :, :]  # [4, reps]
                if item_scale.sum() > 0:
                    reps_with_response = item_scale.sum(axis=0) > 0
                    if reps_with_response.sum() > 0:
                        cat0_selected = item_scale[0, reps_with_response]
                        matrix[m, i] = cat0_selected.mean()
                    break
    return matrix


def print_group_stats(matrix, group_idx, model_names):
    """Print detailed statistics for a group's response matrix."""
    n_judges, n_items = matrix.shape
    total = n_judges * n_items
    n_valid = np.sum(~np.isnan(matrix))
    n_nan = total - n_valid

    print(f"\n{'='*60}")
    print(f"  Group {group_idx} Response Matrix")
    print(f"{'='*60}")
    print(f"  Judges:        {n_judges}")
    print(f"  Items:         {n_items}")
    print(f"  Total cells:   {total:,}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid / total * 100:.1f}%)")
    print(f"  Missing cells: {n_nan:,} ({n_nan / total * 100:.1f}%)")

    if n_valid > 0:
        valid_vals = matrix[~np.isnan(matrix)]
        print(f"\n  Value distribution:")
        print(f"    Mean:   {np.mean(valid_vals):.3f}")
        print(f"    Median: {np.median(valid_vals):.3f}")
        print(f"    Std:    {np.std(valid_vals):.3f}")
        print(f"    Min:    {np.min(valid_vals):.3f}")
        print(f"    Max:    {np.max(valid_vals):.3f}")

        # Binary-like distribution
        n_zero = np.sum(valid_vals == 0.0)
        n_one = np.sum(valid_vals == 1.0)
        n_frac = n_valid - n_zero - n_one
        print(f"\n  Response breakdown:")
        print(f"    Exactly 0.0 (all reps chose other): {n_zero:,} ({n_zero / n_valid * 100:.1f}%)")
        print(f"    Exactly 1.0 (all reps chose cat 0): {n_one:,} ({n_one / n_valid * 100:.1f}%)")
        print(f"    Fractional  (mixed across reps):    {n_frac:,} ({n_frac / n_valid * 100:.1f}%)")

        # Per-judge stats
        print(f"\n  Per-judge mean P(cat 0):")
        for j, name in enumerate(model_names):
            row = matrix[j, :]
            valid_row = row[~np.isnan(row)]
            if len(valid_row) > 0:
                print(f"    {name:40s}  mean={np.mean(valid_row):.3f}  valid={len(valid_row)}")

        # Per-item stats
        per_item_mean = np.nanmean(matrix, axis=0)
        valid_items = ~np.isnan(per_item_mean)
        if valid_items.sum() > 0:
            item_means = per_item_mean[valid_items]
            print(f"\n  Per-item mean P(cat 0) (across judges):")
            print(f"    Min:    {np.min(item_means):.3f}")
            print(f"    Max:    {np.max(item_means):.3f}")
            print(f"    Median: {np.median(item_means):.3f}")
            print(f"    Std:    {np.std(item_means):.3f}")

            # Agreement: items where all judges agree
            high_agreement = np.sum((item_means > 0.9) | (item_means < 0.1))
            print(f"    High agreement (>0.9 or <0.1): {high_agreement}")


def main():
    print("Indeterminacy Experiments Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download
    print("STEP 1: Downloading data")
    print("-" * 60)
    ds = download_dataset()

    # Extract model info
    model_names = []
    providers = []
    for m in range(N_JUDGES):
        mi = ds[m]["model_info"]
        model_names.append(mi["model"])
        providers.append(mi["provider"])
    print(f"\n  Judges ({N_JUDGES}):")
    for name, prov in zip(model_names, providers):
        print(f"    {prov:12s}  {name}")

    # Step 2: Build per-group matrices
    print(f"\nSTEP 2: Building per-group response matrices")
    print("-" * 60)

    group_matrices = {}
    for g in range(N_GROUPS):
        matrix = build_group_matrix(ds, g)
        group_matrices[g] = matrix
        print_group_stats(matrix, g, model_names)

        # Save per-group CSV
        item_ids = [f"g{g}_item_{i:03d}" for i in range(N_ITEMS)]
        df = pd.DataFrame(matrix, index=model_names, columns=item_ids)
        df.index.name = "judge"
        output_path = os.path.join(PROCESSED_DIR, f"response_matrix_group_{g}.csv")
        df.to_csv(output_path)
        print(f"\n  Saved: {output_path}")

    # Step 3: Build combined matrix
    print(f"\nSTEP 3: Building combined response matrix")
    print("-" * 60)

    combined = np.concatenate(
        [group_matrices[g] for g in range(N_GROUPS)], axis=1
    )
    combined_items = []
    for g in range(N_GROUPS):
        combined_items.extend([f"g{g}_item_{i:03d}" for i in range(N_ITEMS)])

    n_judges, n_total_items = combined.shape
    n_valid = np.sum(~np.isnan(combined))
    total = combined.size

    print(f"\n  Combined matrix: {n_judges} judges x {n_total_items} items")
    print(f"  Valid cells: {n_valid:,} / {total:,} ({n_valid / total * 100:.1f}%)")
    print(f"  Mean: {np.nanmean(combined):.3f}")

    df_combined = pd.DataFrame(combined, index=model_names, columns=combined_items)
    df_combined.index.name = "judge"
    combined_path = os.path.join(PROCESSED_DIR, "response_matrix_all.csv")
    df_combined.to_csv(combined_path)
    print(f"  Saved: {combined_path}")

    # Step 4: Build judge summary
    print(f"\nSTEP 4: Building judge summary")
    print("-" * 60)

    summary_rows = []
    for j, name in enumerate(model_names):
        row_data = {"judge": name, "provider": providers[j]}
        # Overall stats
        row_all = combined[j, :]
        valid = row_all[~np.isnan(row_all)]
        row_data["n_valid"] = len(valid)
        row_data["mean_p_cat0"] = np.mean(valid) if len(valid) > 0 else np.nan
        row_data["std_p_cat0"] = np.std(valid) if len(valid) > 0 else np.nan
        row_data["frac_deterministic"] = (
            np.mean((valid == 0.0) | (valid == 1.0)) if len(valid) > 0 else np.nan
        )

        # Per-group stats
        for g in range(N_GROUPS):
            row_g = group_matrices[g][j, :]
            valid_g = row_g[~np.isnan(row_g)]
            row_data[f"group_{g}_mean"] = (
                np.mean(valid_g) if len(valid_g) > 0 else np.nan
            )

        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(PROCESSED_DIR, "judge_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Judge summary:")
    for _, r in summary_df.iterrows():
        print(
            f"    {r['judge']:40s}  mean={r['mean_p_cat0']:.3f}  "
            f"det={r['frac_deterministic']:.1%}  valid={r['n_valid']}"
        )
    print(f"\n  Saved: {summary_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Source: lguerdan/indeterminacy-experiments (HuggingFace)")
    print(f"  Paper:  Validating LLM-as-a-Judge under Rating Indeterminacy (NeurIPS 2025)")
    print(f"  Judges: {N_JUDGES}")
    print(f"  Items per group: {N_ITEMS}")
    print(f"  Task groups: {N_GROUPS}")
    print(f"  Combined matrix: {n_judges} judges x {n_total_items} items")
    print(f"  Response type: P(category 0), continuous [0, 1]")

    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
