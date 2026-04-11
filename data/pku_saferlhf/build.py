"""
Build PKU-SafeRLHF pairwise preference data from PKU-Alignment/PKU-SafeRLHF.

Data source:
  - PKU-Alignment/PKU-SafeRLHF on HuggingFace Hub: ~82K expert comparison
    pairs with dual annotations for helpfulness and safety preferences.

  The dataset has four configurations:
    - default: All Alpaca variants combined (73907 train + 8211 test = 82118)
    - alpaca-7b: Alpaca-7B responses (27393 train + 3036 test = 30429)
    - alpaca2-7b: Alpaca2-7B responses (25564 train + 2848 test = 28412)
    - alpaca3-8b: Alpaca3-8B responses (20950 train + 2327 test = 23277)

Format:
  - Each sample is a (prompt, response_0, response_1) triple with:
      - better_response_id: Which response is more helpful (0 or 1)
      - safer_response_id: Which response is safer (0 or 1)
      - Per-response safety flags, severity levels, and 20 harm categories

Outputs:
  - processed/helpfulness_summary.csv: Per-pair metadata for helpfulness prefs
  - processed/safety_summary.csv: Per-pair metadata for safety preferences
  - processed/config_stats.csv: Aggregate statistics per configuration
  - processed/harm_category_stats.csv: Harm category distribution
"""

INFO = {
    'description': 'Build PKU-SafeRLHF pairwise preference data from PKU-Alignment/PKU-SafeRLHF',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2406.15513',
    'data_source_url': 'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-NC-4.0',
    'citation': """@misc{ji2025pkusaferlhfmultilevelsafetyalignment,
      title={PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference}, 
      author={Jiaming Ji and Donghai Hong and Borong Zhang and Boyuan Chen and Juntao Dai and Boren Zheng and Tianyi Qiu and Jiayi Zhou and Kaile Wang and Boxuan Li and Sirui Han and Yike Guo and Yaodong Yang},
      year={2025},
      eprint={2406.15513},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.15513}, 
}""",
    'tags': ['safety'],
}


import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_SOURCE = "PKU-Alignment/PKU-SafeRLHF"

CONFIGS = ["default", "alpaca-7b", "alpaca2-7b", "alpaca3-8b"]

HARM_CATEGORIES = [
    "Endangering National Security",
    "Insulting Behavior",
    "Discriminatory Behavior",
    "Endangering Public Health",
    "Copyright Issues",
    "Violence",
    "Drugs",
    "Privacy Violation",
    "Economic Crime",
    "Mental Manipulation",
    "Human Trafficking",
    "Physical Harm",
    "Sexual Content",
    "Cybercrime",
    "Disrupting Public Order",
    "Environmental Damage",
    "Psychological Harm",
    "White-Collar Crime",
    "Animal Abuse",
]


def download_config(config_name: str) -> pd.DataFrame:
    """Download and combine train+test splits for a configuration."""
    from datasets import load_dataset

    ds = load_dataset(
        HF_SOURCE,
        name=config_name,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    rows = []
    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        split_ds = ds[split_name]
        print(f"    {split_name}: {len(split_ds)} rows")
        for i, item in enumerate(split_ds):
            row = {
                "config": config_name,
                "split": split_name,
                "pair_idx": i,
                "prompt": item["prompt"],
                "response_0_len": len(item["response_0"]),
                "response_1_len": len(item["response_1"]),
                "prompt_source": item.get("prompt_source", ""),
                "response_0_source": item.get("response_0_source", ""),
                "response_1_source": item.get("response_1_source", ""),
                "is_response_0_safe": item["is_response_0_safe"],
                "is_response_1_safe": item["is_response_1_safe"],
                "response_0_severity_level": item["response_0_severity_level"],
                "response_1_severity_level": item["response_1_severity_level"],
                "better_response_id": item["better_response_id"],
                "safer_response_id": item["safer_response_id"],
            }

            # Extract harm categories as individual columns
            harm_cat_0 = item.get("response_0_harm_category", {})
            harm_cat_1 = item.get("response_1_harm_category", {})
            for cat in HARM_CATEGORIES:
                row[f"resp0_harm_{cat}"] = harm_cat_0.get(cat, False)
                row[f"resp1_harm_{cat}"] = harm_cat_1.get(cat, False)

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Total {config_name}: {len(df)} pairs")
    return df


def print_config_stats(df: pd.DataFrame, config_name: str) -> dict:
    """Print and return aggregate statistics for a configuration."""
    print(f"\n{'=' * 60}")
    print(f"  {config_name.upper()} STATISTICS")
    print(f"{'=' * 60}")

    n_pairs = len(df)
    print(f"  Total pairs: {n_pairs:,}")

    # Split breakdown
    print(f"\n  Split breakdown:")
    for split, count in df["split"].value_counts().items():
        print(f"    {split}: {count:,}")

    # Helpfulness preference distribution
    better_counts = df["better_response_id"].value_counts()
    print(f"\n  Helpfulness preference (better_response_id):")
    for val, count in sorted(better_counts.items()):
        print(f"    response_{val}: {count:,} ({count / n_pairs * 100:.1f}%)")

    # Safety preference distribution
    safer_counts = df["safer_response_id"].value_counts()
    print(f"\n  Safety preference (safer_response_id):")
    for val, count in sorted(safer_counts.items()):
        print(f"    response_{val}: {count:,} ({count / n_pairs * 100:.1f}%)")

    # Safety flags
    n_resp0_safe = df["is_response_0_safe"].sum()
    n_resp1_safe = df["is_response_1_safe"].sum()
    both_safe = ((df["is_response_0_safe"]) & (df["is_response_1_safe"])).sum()
    neither_safe = ((~df["is_response_0_safe"]) & (~df["is_response_1_safe"])).sum()
    print(f"\n  Safety flags:")
    print(f"    response_0 safe: {n_resp0_safe:,} ({n_resp0_safe / n_pairs * 100:.1f}%)")
    print(f"    response_1 safe: {n_resp1_safe:,} ({n_resp1_safe / n_pairs * 100:.1f}%)")
    print(f"    Both safe:       {both_safe:,} ({both_safe / n_pairs * 100:.1f}%)")
    print(f"    Neither safe:    {neither_safe:,} ({neither_safe / n_pairs * 100:.1f}%)")

    # Severity levels
    print(f"\n  Severity levels (response_0):")
    for level, count in sorted(df["response_0_severity_level"].value_counts().items()):
        print(f"    Level {level}: {count:,} ({count / n_pairs * 100:.1f}%)")
    print(f"\n  Severity levels (response_1):")
    for level, count in sorted(df["response_1_severity_level"].value_counts().items()):
        print(f"    Level {level}: {count:,} ({count / n_pairs * 100:.1f}%)")

    # Agreement between helpfulness and safety
    agree = (df["better_response_id"] == df["safer_response_id"]).sum()
    print(f"\n  Helpfulness-Safety agreement:")
    print(f"    Agree: {agree:,} ({agree / n_pairs * 100:.1f}%)")
    print(f"    Disagree: {n_pairs - agree:,} ({(n_pairs - agree) / n_pairs * 100:.1f}%)")

    # Response length stats
    print(f"\n  Response lengths (chars):")
    print(f"    response_0 mean: {df['response_0_len'].mean():.0f}")
    print(f"    response_1 mean: {df['response_1_len'].mean():.0f}")

    return {
        "config": config_name,
        "n_pairs": n_pairs,
        "n_train": (df["split"] == "train").sum(),
        "n_test": (df["split"] == "test").sum(),
        "pct_better_0": (df["better_response_id"] == 0).mean() * 100,
        "pct_safer_0": (df["safer_response_id"] == 0).mean() * 100,
        "pct_both_safe": both_safe / n_pairs * 100,
        "pct_neither_safe": neither_safe / n_pairs * 100,
        "pct_agree": agree / n_pairs * 100,
        "resp0_len_mean": df["response_0_len"].mean(),
        "resp1_len_mean": df["response_1_len"].mean(),
    }


def build_harm_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute harm category prevalence across all responses."""
    rows = []
    for cat in HARM_CATEGORIES:
        col0 = f"resp0_harm_{cat}"
        col1 = f"resp1_harm_{cat}"
        if col0 in df.columns and col1 in df.columns:
            n0 = df[col0].sum()
            n1 = df[col1].sum()
            total = n0 + n1
            rows.append({
                "harm_category": cat,
                "response_0_count": int(n0),
                "response_1_count": int(n1),
                "total_count": int(total),
                "pct_of_pairs": total / (2 * len(df)) * 100,
            })
    return pd.DataFrame(rows).sort_values("total_count", ascending=False)


def build_response_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a (harm-category x response) binary response matrix.

    Each pair contributes two response items (one per side). Rows are the
    19 harm categories plus an overall `is_unsafe` row. Values are binary
    {0, 1} where 1 indicates that category (or overall unsafe) is flagged
    for that response.
    """
    n_pairs = len(df)
    n_items = 2 * n_pairs
    subject_ids = ["is_unsafe"] + HARM_CATEGORIES
    n_subjects = len(subject_ids)

    matrix = np.zeros((n_subjects, n_items), dtype=np.float64)
    item_ids = [None] * n_items

    for i, (_, row) in enumerate(df.iterrows()):
        base = f"{row['config']}_{row['split']}_{int(row['pair_idx'])}"
        # response 0
        col0 = 2 * i
        item_ids[col0] = f"{base}_resp0"
        matrix[0, col0] = 0.0 if bool(row["is_response_0_safe"]) else 1.0
        for j, cat in enumerate(HARM_CATEGORIES):
            key = f"resp0_harm_{cat}"
            matrix[j + 1, col0] = 1.0 if bool(row.get(key, False)) else 0.0
        # response 1
        col1 = 2 * i + 1
        item_ids[col1] = f"{base}_resp1"
        matrix[0, col1] = 0.0 if bool(row["is_response_1_safe"]) else 1.0
        for j, cat in enumerate(HARM_CATEGORIES):
            key = f"resp1_harm_{cat}"
            matrix[j + 1, col1] = 1.0 if bool(row.get(key, False)) else 0.0

    matrix_df = pd.DataFrame(matrix, index=subject_ids, columns=item_ids)
    matrix_df.index.name = "category"
    return matrix_df


def main():
    print("PKU-SafeRLHF Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    all_stats_rows = []
    all_dfs = {}

    # Step 1: Download and process each configuration
    print("STEP 1: Downloading PKU-SafeRLHF from HuggingFace")
    print("-" * 60)

    for config_name in CONFIGS:
        print(f"\n--- {config_name} ---")
        df = download_config(config_name)

        if df.empty:
            print(f"  WARNING: No data for {config_name}, skipping")
            continue

        stats = print_config_stats(df, config_name)
        all_stats_rows.append(stats)
        all_dfs[config_name] = df

    # Step 2: Save summaries
    print(f"\n\n{'=' * 60}")
    print("STEP 2: Saving processed data")
    print("-" * 60)

    # Save per-config summary CSVs (without full text for size)
    summary_cols = [
        "config", "split", "pair_idx",
        "response_0_len", "response_1_len",
        "prompt_source", "response_0_source", "response_1_source",
        "is_response_0_safe", "is_response_1_safe",
        "response_0_severity_level", "response_1_severity_level",
        "better_response_id", "safer_response_id",
    ]

    if "default" in all_dfs:
        default_df = all_dfs["default"]

        # Helpfulness summary
        help_path = os.path.join(PROCESSED_DIR, "helpfulness_summary.csv")
        default_df[summary_cols].to_csv(help_path, index=False)
        print(f"  Saved: {help_path}")

        # Safety summary
        safe_path = os.path.join(PROCESSED_DIR, "safety_summary.csv")
        default_df[summary_cols].to_csv(safe_path, index=False)
        print(f"  Saved: {safe_path}")

        # Harm category stats
        harm_stats = build_harm_category_stats(default_df)
        harm_path = os.path.join(PROCESSED_DIR, "harm_category_stats.csv")
        harm_stats.to_csv(harm_path, index=False)
        print(f"  Saved: {harm_path}")

        # Build and save the category x response binary matrix.
        print("\n  Building response matrix (categories x responses) ...")
        matrix_df = build_response_matrix(default_df)
        rm_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
        matrix_df.to_csv(rm_path)
        n_subs, n_items = matrix_df.shape
        print(f"  Saved: {rm_path}")
        print(f"  Shape: {n_subs} categories x {n_items:,} response items")

        print(f"\n  Harm category distribution (top 10):")
        for _, row in harm_stats.head(10).iterrows():
            print(
                f"    {row['harm_category']:35s}  "
                f"total={row['total_count']:6d}  "
                f"({row['pct_of_pairs']:.2f}%)"
            )

    # Save aggregate config stats
    stats_df = pd.DataFrame(all_stats_rows)
    stats_path = os.path.join(PROCESSED_DIR, "config_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Config stats saved: {stats_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    total = sum(r["n_pairs"] for r in all_stats_rows)
    print(f"  Total preference pairs (across configs): {total:,}")
    for row in all_stats_rows:
        print(f"    {row['config']:15s}: {row['n_pairs']:,} pairs")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


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
