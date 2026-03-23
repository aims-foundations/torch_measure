#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process LMSYS ToxicChat dataset.

Loads the HuggingFace datasets-format data from raw/dataset/.
The data has conversation text + toxicity labels + OpenAI moderation scores.
Builds:
  1. Summary of toxicity rates by split (train/test)
  2. Labeled data as clean CSV with parsed moderation scores
  3. Jailbreaking vs toxicity cross-tabulation

Saves outputs to processed/.
"""

import ast
import json
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LMSYS ToxicChat Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    dataset_path = RAW_DIR / "dataset"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_path}")
        return

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    print(f"\nDataset structure: {ds}")
    print(f"Splits: {list(ds.keys())}")

    # Explore first split
    first_split = list(ds.keys())[0]
    print(f"\nColumns: {ds[first_split].column_names}")
    print(f"First example:")
    for k, v in ds[first_split][0].items():
        print(f"  {k}: {str(v)[:120]}")

    # ------------------------------------------------------------------
    # 2. Convert to DataFrames
    # ------------------------------------------------------------------
    dfs = {}
    for split_name in ds:
        df = ds[split_name].to_pandas()
        df["_split"] = split_name
        dfs[split_name] = df
        print(f"\n--- Split: {split_name} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")

    combined = pd.concat(dfs.values(), ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")

    # ------------------------------------------------------------------
    # 3. Detect and normalize key columns
    # ------------------------------------------------------------------
    # Expected: conv_id, user_input, model_output, human_annotation, toxicity, jailbreaking, openai_moderation
    col_names = combined.columns.tolist()
    print(f"\nAll columns: {col_names}")

    # Detect annotation column (human_annotation is the toxicity label)
    annotation_col = None
    for cand in ["human_annotation", "annotation", "toxic", "label"]:
        if cand in col_names:
            annotation_col = cand
            break

    toxicity_col = None
    for cand in ["toxicity", "toxic_score", "toxicity_score"]:
        if cand in col_names:
            toxicity_col = cand
            break

    jailbreak_col = None
    for cand in ["jailbreaking", "jailbreak", "is_jailbreak"]:
        if cand in col_names:
            jailbreak_col = cand
            break

    moderation_col = None
    for cand in ["openai_moderation", "moderation", "moderation_scores"]:
        if cand in col_names:
            moderation_col = cand
            break

    print(f"\nDetected columns:")
    print(f"  annotation: {annotation_col}")
    print(f"  toxicity:   {toxicity_col}")
    print(f"  jailbreak:  {jailbreak_col}")
    print(f"  moderation: {moderation_col}")

    # ------------------------------------------------------------------
    # 4. Parse and normalize fields
    # ------------------------------------------------------------------
    # human_annotation may be string "True"/"False" or bool
    if annotation_col:
        combined["is_toxic"] = combined[annotation_col].apply(
            lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
        )

    if toxicity_col:
        combined["toxicity_numeric"] = pd.to_numeric(combined[toxicity_col], errors="coerce")

    if jailbreak_col:
        combined["is_jailbreak"] = combined[jailbreak_col].apply(
            lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
        )

    # Parse OpenAI moderation scores (stored as string repr of list of [category, score] pairs)
    if moderation_col:
        def parse_moderation(val):
            """Parse moderation field: list of [category, score] pairs."""
            if pd.isna(val):
                return {}
            try:
                parsed = ast.literal_eval(str(val))
                if isinstance(parsed, list):
                    return {cat: float(score) for cat, score in parsed}
                elif isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            try:
                parsed = json.loads(str(val))
                if isinstance(parsed, list):
                    return {cat: float(score) for cat, score in parsed}
                elif isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {}

        print("\nParsing OpenAI moderation scores...")
        moderation_parsed = combined[moderation_col].apply(parse_moderation)

        # Extract moderation categories
        all_categories = set()
        for d in moderation_parsed:
            all_categories.update(d.keys())
        all_categories = sorted(all_categories)
        print(f"  Found {len(all_categories)} moderation categories: {all_categories}")

        for cat in all_categories:
            col_name = f"mod_{cat.replace('/', '_').replace(' ', '_')}"
            combined[col_name] = moderation_parsed.apply(lambda d, c=cat: d.get(c, np.nan))

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for split_name, df in dfs.items():
        n = len(df)
        print(f"\n--- {split_name} (n={n}) ---")

        if annotation_col:
            # Merge is_toxic back
            mask = combined["_split"] == split_name
            toxic_count = combined.loc[mask, "is_toxic"].sum()
            print(f"  Toxic conversations: {toxic_count} ({100*toxic_count/n:.1f}%)")

        if jailbreak_col:
            jb_count = combined.loc[mask, "is_jailbreak"].sum()
            print(f"  Jailbreaking attempts: {jb_count} ({100*jb_count/n:.1f}%)")

        if toxicity_col:
            tox_vals = combined.loc[mask, "toxicity_numeric"]
            print(f"  Toxicity score: mean={tox_vals.mean():.4f}, std={tox_vals.std():.4f}")

    # Overall
    print(f"\n--- Overall (n={len(combined)}) ---")
    if annotation_col:
        total_toxic = combined["is_toxic"].sum()
        print(f"  Toxic: {total_toxic} ({100*total_toxic/len(combined):.1f}%)")
    if jailbreak_col:
        total_jb = combined["is_jailbreak"].sum()
        print(f"  Jailbreaking: {total_jb} ({100*total_jb/len(combined):.1f}%)")

    # ------------------------------------------------------------------
    # 6. Cross-tabulations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-TABULATIONS")
    print("=" * 70)

    if annotation_col and jailbreak_col:
        ct = pd.crosstab(
            combined["is_toxic"].map({0: "Not Toxic", 1: "Toxic"}),
            combined["is_jailbreak"].map({0: "Not Jailbreak", 1: "Jailbreak"}),
            margins=True,
            margins_name="TOTAL",
        )
        out_path = PROCESSED_DIR / "toxic_x_jailbreak.csv"
        ct.to_csv(out_path)
        print(f"\n  Toxicity x Jailbreaking:")
        print(ct.to_string())
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # 7a. Save full labeled data as CSV
    out_path = PROCESSED_DIR / "toxicchat_labeled.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Full labeled data: {combined.shape}")
    print(f"  Saved to: {out_path}")

    # 7b. Summary statistics
    stats_rows = []
    for split_name in ds:
        mask = combined["_split"] == split_name
        n = mask.sum()
        row = {"split": split_name, "n_conversations": n}
        if annotation_col:
            row["n_toxic"] = int(combined.loc[mask, "is_toxic"].sum())
            row["pct_toxic"] = row["n_toxic"] / n * 100
        if jailbreak_col:
            row["n_jailbreak"] = int(combined.loc[mask, "is_jailbreak"].sum())
            row["pct_jailbreak"] = row["n_jailbreak"] / n * 100
        if toxicity_col:
            row["mean_toxicity_score"] = combined.loc[mask, "toxicity_numeric"].mean()
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics: {stats_df.shape}")
    print(f"  Saved to: {out_path}")
    print(stats_df.to_string(index=False))

    # 7c. Moderation category means
    mod_cols = [c for c in combined.columns if c.startswith("mod_")]
    if mod_cols:
        mod_means = combined[mod_cols].mean().sort_values(ascending=False)
        mod_df = mod_means.reset_index()
        mod_df.columns = ["category", "mean_score"]
        out_path = PROCESSED_DIR / "moderation_category_means.csv"
        mod_df.to_csv(out_path, index=False)
        print(f"\n  Moderation category means:")
        print(mod_df.to_string(index=False))
        print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("ToxicChat processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
