"""
Build PersonalLLM response matrices from reward-model preference scores.

Data source:
  - namkoong-lab/PersonalLLM on HuggingFace Hub
  - 10,402 prompts (9,402 train + 1,000 test), each with 8 LLM responses
  - 10 reward models score every response, acting as proxies for users
    with heterogeneous preferences

Structure:
  - Each prompt has 8 responses from different LLMs:
    cohere/command-r-plus, openai/gpt-4-turbo, openai/gpt-4o,
    anthropic/claude-3-opus, anthropic/claude-3-sonnet,
    meta-llama/llama-3-70b-instruct:nitro, google/gemini-pro-1.5,
    mistralai/mixtral-8x22b-instruct
  - Each response is scored by 10 reward models with different preference
    profiles (continuous scores on model-specific scales)

Score format:
  - Continuous reward scores (unbounded, model-specific scale)
  - Higher scores = better response according to that reward model
  - Different reward models have systematically different preference patterns

Outputs:
  - response_matrix.csv: Reward scores (reward_models x prompt-response pairs)
    for the full dataset (train + test)
  - response_matrix_train.csv: Train split only
  - response_matrix_test.csv: Test split only
  - item_metadata.csv: Per-item metadata (prompt_id, response_model, subset)
  - subject_summary.csv: Per-reward-model aggregate statistics
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SRC_REPO = "namkoong-lab/PersonalLLM"

# Reward model short names (column suffixes) -> full model names.
REWARD_MODEL_MAP = {
    "gemma_2b": "weqweasdas/RM-Gemma-2B",
    "gemma_7b": "weqweasdas/RM-Gemma-7B",
    "mistral_raft": "hendrydong/Mistral-RM-for-RAFT-GSHF-v0",
    "mistral_ray": "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback",
    "mistral_weqweasdas": "weqweasdas/RM-Mistral-7B",
    "llama3_sfairx": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
    "oasst_deberta_v3": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "beaver_7b": "PKU-Alignment/beaver-7b-v1.0-cost",
    "oasst_pythia_7b": "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
    "oasst_pythia_1b": "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
}

REWARD_MODEL_KEYS = sorted(REWARD_MODEL_MAP.keys())
N_RESPONSES = 8


def download_dataset():
    """Download PersonalLLM dataset from HuggingFace Hub."""
    print("Downloading PersonalLLM dataset from HuggingFace Hub ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SRC_REPO, token=HF_TOKEN if HF_TOKEN else None)
        print(f"  Train: {len(ds['train'])} rows")
        print(f"  Test:  {len(ds['test'])} rows")
        return ds
    except Exception as e:
        print(f"  Failed to download: {e}")
        sys.exit(1)


def build_response_matrix(dataset, split_name="all"):
    """Build reward_models x prompt-response-pairs matrix from a dataset split.

    Parameters
    ----------
    dataset : HuggingFace Dataset
        A single split (or concatenation) of the PersonalLLM dataset.
    split_name : str
        Name for logging and output files.

    Returns
    -------
    tuple of (matrix_df, item_meta_df)
    """
    n_prompts = len(dataset)
    n_items = n_prompts * N_RESPONSES
    n_subjects = len(REWARD_MODEL_KEYS)

    print(f"\n{'='*60}")
    print(f"  Building response matrix: {split_name}")
    print(f"  {n_subjects} reward models x {n_items} items "
          f"({n_prompts} prompts x {N_RESPONSES} responses)")
    print(f"{'='*60}")

    # Initialize matrix
    matrix = np.full((n_subjects, n_items), np.nan)

    # Collect item metadata
    item_ids = []
    item_meta_rows = []

    for row_idx, row in enumerate(dataset):
        prompt_id = row["prompt_id"]
        subset = row.get("subset", "")

        for resp_idx in range(1, N_RESPONSES + 1):
            item_col_idx = row_idx * N_RESPONSES + (resp_idx - 1)
            item_id = f"{prompt_id}_{resp_idx}"
            item_ids.append(item_id)

            # Response model
            model_col = f"response_{resp_idx}_model"
            response_model = row.get(model_col, "")

            item_meta_rows.append({
                "item_id": item_id,
                "prompt_id": prompt_id,
                "response_idx": resp_idx,
                "response_model": response_model,
                "subset": subset,
            })

            # Fill reward scores
            for rm_idx, rm_key in enumerate(REWARD_MODEL_KEYS):
                score_col = f"response_{resp_idx}_{rm_key}"
                score = row.get(score_col)
                if score is not None:
                    matrix[rm_idx, item_col_idx] = float(score)

        if (row_idx + 1) % 2000 == 0:
            print(f"    Processed {row_idx + 1}/{n_prompts} prompts")

    # Subject IDs (full reward model names)
    subject_ids = [REWARD_MODEL_MAP[k] for k in REWARD_MODEL_KEYS]

    # Create DataFrames
    matrix_df = pd.DataFrame(matrix, index=subject_ids, columns=item_ids)
    matrix_df.index.name = "reward_model"

    item_meta_df = pd.DataFrame(item_meta_rows)

    # Statistics
    total_cells = n_subjects * n_items
    n_valid = np.sum(~np.isnan(matrix))
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells

    print(f"\n  Reward models: {n_subjects}")
    print(f"  Items:         {n_items}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:     {fill_rate*100:.1f}%")

    # Per-subject stats
    per_subject_mean = np.nanmean(matrix, axis=1)
    per_subject_std = np.nanstd(matrix, axis=1)
    print(f"\n  Per-reward-model mean score:")
    for i, rm_key in enumerate(REWARD_MODEL_KEYS):
        print(f"    {REWARD_MODEL_MAP[rm_key]:55s}  "
              f"mean={per_subject_mean[i]:+.4f}  std={per_subject_std[i]:.4f}")

    # Per-item stats
    per_item_mean = np.nanmean(matrix, axis=0)
    per_item_std = np.nanstd(matrix, axis=0)
    print(f"\n  Item score distribution (across all reward models):")
    print(f"    Mean of means: {np.mean(per_item_mean):.4f}")
    print(f"    Std of means:  {np.std(per_item_mean):.4f}")
    print(f"    Min mean:      {np.min(per_item_mean):.4f}")
    print(f"    Max mean:      {np.max(per_item_mean):.4f}")

    # Subset breakdown
    subsets = item_meta_df["subset"].unique()
    print(f"\n  Subset distribution:")
    for s in sorted(subsets):
        count = (item_meta_df["subset"] == s).sum()
        print(f"    {s:40s}  n={count:6d}")

    return matrix_df, item_meta_df


def build_subject_summary(matrix_df):
    """Build per-reward-model summary statistics."""
    rows = []
    for rm in matrix_df.index:
        rm_row = matrix_df.loc[rm]
        rows.append({
            "reward_model": rm,
            "mean_score": rm_row.mean(),
            "std_score": rm_row.std(),
            "min_score": rm_row.min(),
            "max_score": rm_row.max(),
            "n_items": rm_row.notna().sum(),
            "n_missing": rm_row.isna().sum(),
        })

    subject_df = pd.DataFrame(rows)
    subject_df = subject_df.sort_values("mean_score", ascending=False)
    output_path = os.path.join(PROCESSED_DIR, "subject_summary.csv")
    subject_df.to_csv(output_path, index=False)
    print(f"\n  Subject summary saved: {output_path}")
    return subject_df


def main():
    print("PersonalLLM Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading PersonalLLM dataset")
    print("-" * 60)
    ds = download_dataset()

    # Step 2: Build response matrices per split
    print("\nSTEP 2: Building response matrices")
    print("-" * 60)

    # Train split
    matrix_train, meta_train = build_response_matrix(ds["train"], "train")
    train_path = os.path.join(PROCESSED_DIR, "response_matrix_train.csv")
    matrix_train.to_csv(train_path)
    print(f"\n  Saved: {train_path}")

    # Test split
    matrix_test, meta_test = build_response_matrix(ds["test"], "test")
    test_path = os.path.join(PROCESSED_DIR, "response_matrix_test.csv")
    matrix_test.to_csv(test_path)
    print(f"\n  Saved: {test_path}")

    # Combined (all)
    from datasets import concatenate_datasets

    combined = concatenate_datasets([ds["train"], ds["test"]])
    matrix_all, meta_all = build_response_matrix(combined, "all")
    all_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_all.to_csv(all_path)
    print(f"\n  Saved: {all_path}")

    # Step 3: Save item metadata
    print("\nSTEP 3: Saving item metadata")
    print("-" * 60)
    meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    meta_all.to_csv(meta_path, index=False)
    print(f"  Item metadata saved: {meta_path}")

    # Step 4: Build subject summary
    print("\nSTEP 4: Building subject summary")
    print("-" * 60)
    subject_df = build_subject_summary(matrix_all)

    # Step 5: Preference correlation analysis
    print("\nSTEP 5: Preference correlation analysis")
    print("-" * 60)
    corr_matrix = matrix_all.T.corr()
    corr_path = os.path.join(PROCESSED_DIR, "reward_model_correlation.csv")
    corr_matrix.to_csv(corr_path)
    print(f"  Correlation matrix saved: {corr_path}")
    print(f"\n  Mean pairwise correlation: {corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean():.4f}")
    print(f"  Min pairwise correlation:  {corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].min():.4f}")
    print(f"  Max pairwise correlation:  {corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].max():.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Response matrix (all):   {matrix_all.shape[0]} reward models x {matrix_all.shape[1]} items")
    print(f"  Response matrix (train): {matrix_train.shape[0]} reward models x {matrix_train.shape[1]} items")
    print(f"  Response matrix (test):  {matrix_test.shape[0]} reward models x {matrix_test.shape[1]} items")
    print(f"  Unique subsets: {len(meta_all['subset'].unique())}")
    print(f"  Response models: {len(meta_all['response_model'].unique())}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
