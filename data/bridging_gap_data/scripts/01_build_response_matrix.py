#!/usr/bin/env python3
"""
Build response matrix for the Bridging-the-Gap African Languages evaluation.

Data source:
  GitHub repo: InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages
  - results/gpt_performance/wino_evaluation_results_{0,1,2}.csv
    Three independent evaluation runs of GPT-4o, GPT-4, and GPT-3.5 on 1,767
    Winogrande items across 12 languages (English + 11 African languages).
    Each CSV is already a response matrix: rows = items, columns = model_language
    combinations, values = binary 0/1 correct/incorrect.

Processing:
  1. Clone the repo (without LFS) into raw/
  2. Load all three evaluation-run CSVs
  3. Average across the 3 runs and threshold at majority vote (>=2 of 3 correct)
     to produce a single robust binary response matrix
  4. Also produce a per-run response matrix (items x model_lang_run)
  5. Build task_metadata.csv with per-item metadata
  6. Build model_summary.csv with per-model per-language accuracy

Outputs (in processed/):
  - response_matrix.csv          : Binary matrix (items x model_language), majority-vote
  - response_matrix_per_run.csv  : Binary matrix (items x model_language_run), all 3 runs
  - response_matrix_averaged.csv : Averaged (mean of 3 runs, continuous 0-1)
  - task_metadata.csv            : Item-level metadata
  - model_summary.csv            : Per-model per-language accuracy statistics
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

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
REPO_DIR = RAW_DIR / "Bridging-the-Gap-Low-Resource-African-Languages"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages.git"

# ──────────────────────────────────────────────────────────────────────
# Language metadata
# ──────────────────────────────────────────────────────────────────────
LANG_NAMES = {
    "en": "English",
    "xh": "Xhosa",
    "zu": "Zulu",
    "af": "Afrikaans",
    "ig": "Igbo",
    "sn": "Shona",
    "ts": "Tsonga",
    "st": "Sesotho",
    "nso": "Sepedi",
    "tn": "Setswana",
    "bm": "Bambara",
    "am": "Amharic",
}

LANG_FAMILIES = {
    "en": "Indo-European (Germanic)",
    "af": "Indo-European (Germanic)",
    "xh": "Niger-Congo (Nguni)",
    "zu": "Niger-Congo (Nguni)",
    "sn": "Niger-Congo (Shona)",
    "ts": "Niger-Congo (Tsonga)",
    "st": "Niger-Congo (Sotho-Tswana)",
    "nso": "Niger-Congo (Sotho-Tswana)",
    "tn": "Niger-Congo (Sotho-Tswana)",
    "ig": "Niger-Congo (Igboid)",
    "bm": "Niger-Congo (Manding)",
    "am": "Afro-Asiatic (Semitic)",
}


def clone_repo():
    """Clone the GitHub repo without LFS (skip large files)."""
    if REPO_DIR.exists():
        print(f"  Repository already exists at {REPO_DIR}")
        return

    print(f"  Cloning repo (without LFS) into {RAW_DIR}/ ...")
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    result = subprocess.run(
        ["git", "clone", REPO_URL],
        cwd=str(RAW_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR cloning repo: {result.stderr}")
        sys.exit(1)
    print(f"  Clone complete.")


def load_gpt_performance_csvs():
    """Load the three Winogrande evaluation CSVs.

    Returns a list of DataFrames, one per run.
    """
    gpt_dir = REPO_DIR / "results" / "gpt_performance"
    csv_files = sorted(gpt_dir.glob("wino_evaluation_results_*.csv"))

    if not csv_files:
        print(f"  ERROR: No CSV files found in {gpt_dir}")
        sys.exit(1)

    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        run_idx = csv_path.stem.split("_")[-1]  # "0", "1", "2"
        print(f"  Loaded {csv_path.name}: {df.shape[0]} items x {df.shape[1]} columns (run {run_idx})")
        dfs.append((run_idx, df))

    return dfs


def parse_model_language(col_name):
    """Parse a column name like 'gpt-4o_en' into (model, language).

    Handles model names with hyphens and dots (gpt-4o, gpt-4, gpt-3.5).
    The last underscore-separated token is the language code.
    """
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return col_name, "unknown"


def build_response_matrices(run_dfs):
    """Build the response matrices from the per-run DataFrames.

    Returns:
        majority_df: Binary response matrix (majority vote across 3 runs)
        averaged_df: Averaged response matrix (mean across 3 runs, continuous)
        per_run_df:  Per-run response matrix (items x model_lang_run)
        item_ids: List of item IDs
        answers: Dict of item_id -> correct answer
        model_lang_cols: List of model_language column names
    """
    # Validate that all runs have the same items and columns
    ids_list = [set(df["id"].tolist()) for _, df in run_dfs]
    assert all(ids == ids_list[0] for ids in ids_list), "Item IDs differ across runs!"

    cols_list = [set(df.columns.tolist()) for _, df in run_dfs]
    assert all(cols == cols_list[0] for cols in cols_list), "Columns differ across runs!"

    # Use the first run's ordering
    run0_idx, run0_df = run_dfs[0]
    item_ids = run0_df["id"].tolist()
    answers = dict(zip(run0_df["id"], run0_df["answer"]))
    model_lang_cols = [c for c in run0_df.columns if c not in ("id", "answer")]

    print(f"\n  Item IDs: {len(item_ids)}")
    print(f"  Model-language columns: {len(model_lang_cols)}")
    print(f"  Columns: {model_lang_cols}")

    # ── Per-run matrix: items x model_lang_run ──
    per_run_parts = []
    for run_idx, df in run_dfs:
        df_sorted = df.set_index("id").loc[item_ids]
        run_cols = {c: f"{c}_run{run_idx}" for c in model_lang_cols}
        part = df_sorted[model_lang_cols].rename(columns=run_cols)
        per_run_parts.append(part)

    per_run_df = pd.concat(per_run_parts, axis=1)
    per_run_df.index.name = "item_id"

    # ── Averaged matrix: mean across 3 runs ──
    stacked = []
    for run_idx, df in run_dfs:
        df_sorted = df.set_index("id").loc[item_ids]
        stacked.append(df_sorted[model_lang_cols].values)

    stacked_arr = np.stack(stacked, axis=0)  # (3, n_items, n_cols)
    averaged_arr = np.mean(stacked_arr, axis=0)

    averaged_df = pd.DataFrame(
        averaged_arr,
        index=item_ids,
        columns=model_lang_cols,
    )
    averaged_df.index.name = "item_id"

    # ── Majority-vote matrix: >= 2/3 correct -> 1 ──
    majority_arr = (averaged_arr >= (2.0 / 3.0)).astype(int)
    majority_df = pd.DataFrame(
        majority_arr,
        index=item_ids,
        columns=model_lang_cols,
    )
    majority_df.index.name = "item_id"

    return majority_df, averaged_df, per_run_df, item_ids, answers, model_lang_cols


def build_task_metadata(item_ids, answers):
    """Build task_metadata.csv with per-item information."""
    rows = []
    for item_id in item_ids:
        rows.append({
            "item_id": item_id,
            "benchmark": "winogrande",
            "answer": answers.get(item_id, ""),
            "source": "Bridging-the-Gap",
        })

    meta_df = pd.DataFrame(rows)
    return meta_df


def build_model_summary(majority_df, averaged_df, model_lang_cols):
    """Build model_summary.csv with per-model per-language accuracy.

    Uses both the majority-vote matrix (for binary accuracy) and the
    averaged matrix (for mean accuracy across 3 runs).
    """
    rows = []
    for col in model_lang_cols:
        model, language = parse_model_language(col)

        # Majority-vote accuracy
        majority_col = majority_df[col]
        n_items = len(majority_col)
        n_correct_majority = int(majority_col.sum())
        accuracy_majority = n_correct_majority / n_items * 100

        # Mean accuracy (averaged across 3 runs)
        avg_col = averaged_df[col]
        accuracy_mean = avg_col.mean() * 100

        rows.append({
            "model": model,
            "language": language,
            "language_name": LANG_NAMES.get(language, language),
            "language_family": LANG_FAMILIES.get(language, ""),
            "column_name": col,
            "n_items": n_items,
            "n_correct_majority": n_correct_majority,
            "accuracy_majority_pct": round(accuracy_majority, 2),
            "accuracy_mean_pct": round(accuracy_mean, 2),
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["model", "accuracy_mean_pct"], ascending=[True, False]
    )
    return summary_df


def print_summary(majority_df, averaged_df, model_summary_df, task_metadata_df):
    """Print a comprehensive summary of the built data."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_items, n_model_langs = majority_df.shape
    print(f"\n  Response matrix dimensions: {n_items} items x {n_model_langs} model-language combinations")
    print(f"  Total cells: {n_items * n_model_langs:,}")
    print(f"  Overall accuracy (majority vote): {majority_df.values.mean() * 100:.1f}%")
    print(f"  Overall accuracy (averaged): {averaged_df.values.mean() * 100:.1f}%")

    # Per-model summary
    print("\n  Per-model accuracy (averaged across 3 runs):")
    models = model_summary_df["model"].unique()
    for model in sorted(models):
        subset = model_summary_df[model_summary_df["model"] == model]
        mean_acc = subset["accuracy_mean_pct"].mean()
        en_row = subset[subset["language"] == "en"]
        en_acc = en_row["accuracy_mean_pct"].iloc[0] if len(en_row) > 0 else float("nan")
        afr_subset = subset[subset["language"] != "en"]
        afr_mean = afr_subset["accuracy_mean_pct"].mean() if len(afr_subset) > 0 else float("nan")
        print(f"    {model:12s}  English: {en_acc:5.1f}%  African avg: {afr_mean:5.1f}%  Overall: {mean_acc:5.1f}%")

    # Per-language summary
    print("\n  Per-language accuracy (averaged across all models, 3 runs):")
    for lang in model_summary_df["language"].unique():
        subset = model_summary_df[model_summary_df["language"] == lang]
        mean_acc = subset["accuracy_mean_pct"].mean()
        lang_name = LANG_NAMES.get(lang, lang)
        family = LANG_FAMILIES.get(lang, "")
        print(f"    {lang:4s} ({lang_name:12s}, {family:30s}): {mean_acc:5.1f}%")

    # Item difficulty
    item_pass_rate = majority_df.mean(axis=1)
    print(f"\n  Item difficulty (majority-vote pass rate across {n_model_langs} model-langs):")
    print(f"    Mean pass rate: {item_pass_rate.mean():.3f}")
    print(f"    Median pass rate: {item_pass_rate.median():.3f}")
    print(f"    Items answered correctly by all: {(item_pass_rate == 1.0).sum()}")
    print(f"    Items answered correctly by none: {(item_pass_rate == 0.0).sum()}")
    print(f"    Items with pass rate > 0.5: {(item_pass_rate > 0.5).sum()}")


def main():
    print("=" * 70)
    print("Bridging-the-Gap African Languages — Response Matrix Builder")
    print("=" * 70)

    # ── Step 1: Clone repo ──
    print("\n[1] Cloning repository...")
    clone_repo()

    # ── Step 2: Load CSVs ──
    print("\n[2] Loading GPT performance CSVs...")
    run_dfs = load_gpt_performance_csvs()

    # ── Step 3: Build response matrices ──
    print("\n[3] Building response matrices...")
    majority_df, averaged_df, per_run_df, item_ids, answers, model_lang_cols = \
        build_response_matrices(run_dfs)

    # ── Step 4: Build task metadata ──
    print("\n[4] Building task metadata...")
    task_metadata_df = build_task_metadata(item_ids, answers)
    print(f"  task_metadata: {task_metadata_df.shape[0]} items")

    # ── Step 5: Build model summary ──
    print("\n[5] Building model summary...")
    model_summary_df = build_model_summary(majority_df, averaged_df, model_lang_cols)
    print(f"  model_summary: {model_summary_df.shape[0]} rows "
          f"({model_summary_df['model'].nunique()} models x "
          f"{model_summary_df['language'].nunique()} languages)")

    # ── Step 6: Save outputs ──
    print("\n[6] Saving outputs...")

    out_majority = PROCESSED_DIR / "response_matrix.csv"
    majority_df.to_csv(out_majority)
    print(f"  Saved: {out_majority} ({majority_df.shape[0]} x {majority_df.shape[1]})")

    out_averaged = PROCESSED_DIR / "response_matrix_averaged.csv"
    averaged_df.to_csv(out_averaged)
    print(f"  Saved: {out_averaged}")

    out_per_run = PROCESSED_DIR / "response_matrix_per_run.csv"
    per_run_df.to_csv(out_per_run)
    print(f"  Saved: {out_per_run} ({per_run_df.shape[0]} x {per_run_df.shape[1]})")

    out_meta = PROCESSED_DIR / "task_metadata.csv"
    task_metadata_df.to_csv(out_meta, index=False)
    print(f"  Saved: {out_meta}")

    out_summary = PROCESSED_DIR / "model_summary.csv"
    model_summary_df.to_csv(out_summary, index=False)
    print(f"  Saved: {out_summary}")

    # ── Step 7: Print summary ──
    print_summary(majority_df, averaged_df, model_summary_df, task_metadata_df)

    # ── Final report ──
    print(f"\n{'=' * 70}")
    print(f"FINAL REPORT")
    print(f"{'=' * 70}")
    print(f"  Response matrix (majority vote): {majority_df.shape[0]} items x {majority_df.shape[1]} model-language combos")
    print(f"  Response matrix (per-run):       {per_run_df.shape[0]} items x {per_run_df.shape[1]} model-language-run combos")
    print(f"  Task metadata:                   {task_metadata_df.shape[0]} items")
    print(f"  Model summary:                   {model_summary_df.shape[0]} rows")
    print(f"\n  Output directory: {PROCESSED_DIR}")

    print(f"\n  Output files:")
    for f in sorted(PROCESSED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s}  {size_kb:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
