"""
Build HelpSteer2 response matrices from NVIDIA HelpSteer2 human preference data.

Data source:
  - nvidia/HelpSteer2 on HuggingFace: ~21K samples (~10K prompts x 2 responses),
    rated by human annotators on 5 attributes (helpfulness, correctness, coherence,
    complexity, verbosity) on a 0-4 integer scale.

Structure:
  HelpSteer2 does NOT include model identities for responses. Each prompt has
  exactly 2 anonymous responses rated on the same 5 attributes. We build the
  response matrix as:
    - Rows (subjects): response_0, response_1 (the two responses per prompt)
    - Columns (items): prompts (identified by integer index)
    - Values: human ratings normalized to [0, 1] (from original 0-4 scale)

Outputs (in processed/):
  - response_matrix_helpfulness.csv: Helpfulness scores (2 x n_prompts)
  - response_matrix_correctness.csv: Correctness scores (2 x n_prompts)
  - response_matrix_coherence.csv:   Coherence scores (2 x n_prompts)
  - response_matrix_complexity.csv:  Complexity scores (2 x n_prompts)
  - response_matrix_verbosity.csv:   Verbosity scores (2 x n_prompts)
  - response_matrix_overall.csv:     Mean across all 5 attributes (2 x n_prompts)
  - prompt_metadata.csv:             Per-prompt metadata (prompt text, etc.)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
MAX_SCORE = 4  # Original scores are 0-4


def download_helpsteer2() -> pd.DataFrame:
    """Download HelpSteer2 from HuggingFace and cache locally."""
    cache_path = RAW_DIR / "helpsteer2_all.csv"

    if cache_path.exists():
        print(f"  Loading cached data from {cache_path}")
        return pd.read_csv(cache_path)

    print("  Downloading nvidia/HelpSteer2 from HuggingFace ...")
    from datasets import load_dataset

    ds_train = load_dataset("nvidia/HelpSteer2", split="train")
    ds_val = load_dataset("nvidia/HelpSteer2", split="validation")

    df_train = pd.DataFrame(ds_train)
    df_val = pd.DataFrame(ds_val)
    df = pd.concat([df_train, df_val], ignore_index=True)

    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} rows to {cache_path}")
    return df


def build_response_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build per-attribute response matrices from paired responses.

    Returns dict mapping attribute name -> DataFrame (2 x n_prompts).
    """
    df = df.copy()
    df["response_idx"] = df.groupby("prompt").cumcount()

    # Keep only prompts with exactly 2 responses
    prompt_counts = df.groupby("prompt").size()
    valid_prompts = prompt_counts[prompt_counts == 2].index
    df = df[df["prompt"].isin(valid_prompts)].copy()

    # Create stable prompt IDs
    unique_prompts = sorted(df["prompt"].unique())
    prompt_to_id = {p: str(i) for i, p in enumerate(unique_prompts)}
    df["prompt_id"] = df["prompt"].map(prompt_to_id)

    n_prompts = len(unique_prompts)
    print(f"\n  Paired prompts (exactly 2 responses): {n_prompts}")

    matrices = {}

    for attr in ATTRIBUTES:
        pivot = df.pivot_table(
            values=attr,
            index="response_idx",
            columns="prompt_id",
            aggfunc="first",
        )
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        # Normalize to [0, 1]
        pivot = pivot / MAX_SCORE

        # Rename index
        pivot.index = [f"response_{i}" for i in pivot.index]
        pivot.index.name = "subject"

        matrices[attr] = pivot

    # Overall: mean across all attributes
    overall = sum(matrices[attr] for attr in ATTRIBUTES) / len(ATTRIBUTES)
    overall.index.name = "subject"
    matrices["overall"] = overall

    return matrices, df, unique_prompts


def save_prompt_metadata(df: pd.DataFrame, unique_prompts: list[str]) -> None:
    """Save per-prompt metadata."""
    rows = []
    for i, prompt in enumerate(unique_prompts):
        prompt_rows = df[df["prompt"] == prompt]
        row = {
            "prompt_id": str(i),
            "prompt_text": prompt[:500],  # Truncate long prompts
            "prompt_length": len(prompt),
        }
        # Add mean scores across the two responses
        for attr in ATTRIBUTES:
            row[f"mean_{attr}"] = prompt_rows[attr].mean()
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    meta_path = PROCESSED_DIR / "prompt_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n  Saved prompt metadata: {meta_path} ({len(meta_df)} prompts)")


def main():
    print("HelpSteer2 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")

    # Step 1: Download
    print("\nSTEP 1: Downloading HelpSteer2 data")
    print("-" * 60)
    df = download_helpsteer2()

    print(f"\n  Total rows:      {len(df)}")
    print(f"  Unique prompts:  {df['prompt'].nunique()}")
    print(f"  Columns:         {df.columns.tolist()}")

    # Score distributions
    print("\n  Score distributions (raw 0-4):")
    for attr in ATTRIBUTES:
        print(f"    {attr:15s}: mean={df[attr].mean():.2f}, "
              f"std={df[attr].std():.2f}, "
              f"min={df[attr].min()}, max={df[attr].max()}")

    # Step 2: Build response matrices
    print("\nSTEP 2: Building response matrices")
    print("-" * 60)
    matrices, df_processed, unique_prompts = build_response_matrices(df)

    for name, mat in sorted(matrices.items()):
        n_subjects, n_items = mat.shape
        total_cells = n_subjects * n_items
        n_valid = mat.notna().sum().sum()
        fill_rate = n_valid / total_cells if total_cells > 0 else 0

        print(f"\n  {name}:")
        print(f"    Dimensions:  {n_subjects} subjects x {n_items} items")
        print(f"    Fill rate:   {fill_rate*100:.1f}%")

        all_vals = mat.values.flatten()
        valid_vals = all_vals[~np.isnan(all_vals)]
        if len(valid_vals) > 0:
            print(f"    Mean:        {np.mean(valid_vals):.3f}")
            print(f"    Std:         {np.std(valid_vals):.3f}")
            print(f"    Min:         {np.min(valid_vals):.3f}")
            print(f"    Max:         {np.max(valid_vals):.3f}")

        # Save
        output_path = PROCESSED_DIR / f"response_matrix_{name}.csv"
        mat.to_csv(output_path)
        print(f"    Saved: {output_path}")

    # Step 3: Save prompt metadata
    print("\nSTEP 3: Saving prompt metadata")
    print("-" * 60)
    save_prompt_metadata(df_processed, unique_prompts)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Attributes: {ATTRIBUTES}")
    print(f"  Score range: 0-4 (normalized to [0, 1])")
    print(f"  Subjects:    response_0, response_1 (anonymous paired responses)")

    for name, mat in sorted(matrices.items()):
        n_s, n_i = mat.shape
        print(f"  helpsteer2/{name}: {n_s} x {n_i}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = PROCESSED_DIR / f
        size_kb = fpath.stat().st_size / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
