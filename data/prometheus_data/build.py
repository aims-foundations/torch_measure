"""
Build Prometheus response matrices from GPT-4 evaluation scores.

Data sources:
  - prometheus-eval/Feedback-Collection on HuggingFace (100K instances)
    GPT-4 evaluates model responses on custom rubrics (996 criteria), scored 1-5.
  - prometheus-eval/Preference-Collection on HuggingFace (200K instances)
    GPT-4 pairwise preferences between two responses on custom rubrics.

Response matrix structure:
  Feedback-Collection:
    - Rows (subjects): 996 unique evaluation rubric criteria.
    - Columns (items): Instance indices within each criterion (~100 per criterion).
    - Values: GPT-4 scores normalized to [0, 1] from original 1-5 scale via (s-1)/4.

  Preference-Collection:
    - Rows (subjects): 996 unique evaluation rubric criteria.
    - Columns (items): Instance indices within each criterion (~200 per criterion).
    - Values: Binary {0, 1} where 1 = response B preferred, 0 = response A preferred.

Outputs:
  - processed/feedback_response_matrix.csv
  - processed/preference_response_matrix.csv
  - processed/criteria_metadata.csv
"""

import os
from collections import defaultdict

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

SRC_FEEDBACK_REPO = "prometheus-eval/Feedback-Collection"
SRC_PREFERENCE_REPO = "prometheus-eval/Preference-Collection"


def load_feedback_collection():
    """Load Feedback-Collection dataset from HuggingFace."""
    cache_path = os.path.join(RAW_DIR, "feedback_scores.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached extraction: {cache_path}")
        return pd.read_csv(cache_path)

    print("  Loading prometheus-eval/Feedback-Collection from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        SRC_FEEDBACK_REPO,
        split="train",
        token=HF_TOKEN,
    )

    records = []
    criteria_instance_count = defaultdict(int)

    for item in ds:
        criteria = item.get("orig_criteria", "")
        score_str = item.get("orig_score", "")
        instruction = item.get("orig_instruction", "")

        if not criteria or not score_str:
            continue

        try:
            score = int(score_str)
        except (ValueError, TypeError):
            continue

        if score < 1 or score > 5:
            continue

        instance_idx = criteria_instance_count[criteria]
        criteria_instance_count[criteria] += 1

        records.append({
            "criteria": criteria,
            "instance_idx": instance_idx,
            "score": score,
            "instruction_preview": instruction[:200] if instruction else "",
        })

    print(f"  Extracted {len(records):,} evaluation records")
    print(f"  Unique criteria: {len(criteria_instance_count):,}")

    scores_df = pd.DataFrame(records)
    scores_df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    return scores_df


def load_preference_collection():
    """Load Preference-Collection dataset from HuggingFace."""
    cache_path = os.path.join(RAW_DIR, "preference_scores.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached extraction: {cache_path}")
        return pd.read_csv(cache_path)

    print("  Loading prometheus-eval/Preference-Collection from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        SRC_PREFERENCE_REPO,
        split="train",
        token=HF_TOKEN,
    )

    records = []
    criteria_instance_count = defaultdict(int)

    for item in ds:
        criteria = item.get("orig_criteria", "")
        preference = item.get("orig_preference", "")
        score_a_str = item.get("orig_score_A", "")
        score_b_str = item.get("orig_score_B", "")
        instruction = item.get("orig_instruction", "")

        if not criteria or not preference:
            continue

        # Binary: 1 if B preferred, 0 if A preferred
        if preference == "B":
            binary_pref = 1
        elif preference == "A":
            binary_pref = 0
        else:
            continue

        # Parse scores for metadata
        try:
            score_a = int(score_a_str) if score_a_str else None
            score_b = int(score_b_str) if score_b_str else None
        except (ValueError, TypeError):
            score_a = None
            score_b = None

        instance_idx = criteria_instance_count[criteria]
        criteria_instance_count[criteria] += 1

        records.append({
            "criteria": criteria,
            "instance_idx": instance_idx,
            "preference": binary_pref,
            "score_a": score_a,
            "score_b": score_b,
            "instruction_preview": instruction[:200] if instruction else "",
        })

    print(f"  Extracted {len(records):,} preference records")
    print(f"  Unique criteria: {len(criteria_instance_count):,}")

    pref_df = pd.DataFrame(records)
    pref_df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    return pref_df


def build_feedback_matrix(scores_df):
    """Build response matrix from Feedback-Collection scores.

    Rows = criteria (996), Columns = instance index within criterion (~100 each).
    Values = normalized scores [0, 1] from (score - 1) / 4.
    """
    print("\nBuilding Feedback-Collection response matrix...")

    # Normalize scores from 1-5 to [0, 1]
    scores_df = scores_df.copy()
    scores_df["norm_score"] = (scores_df["score"] - 1) / 4.0

    # Pivot: criteria x instance_idx
    matrix_df = scores_df.pivot_table(
        index="criteria",
        columns="instance_idx",
        values="norm_score",
        aggfunc="mean",
    )
    matrix_df.index.name = "criteria"

    n_criteria, n_instances = matrix_df.shape
    print(f"  Matrix: {n_criteria} criteria x {n_instances} instances")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "feedback_response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def build_preference_matrix(pref_df):
    """Build response matrix from Preference-Collection preferences.

    Rows = criteria (996), Columns = instance index within criterion (~200 each).
    Values = binary {0, 1} where 1 = B preferred.
    """
    print("\nBuilding Preference-Collection response matrix...")

    # Pivot: criteria x instance_idx
    matrix_df = pref_df.pivot_table(
        index="criteria",
        columns="instance_idx",
        values="preference",
        aggfunc="mean",
    )
    matrix_df.index.name = "criteria"

    n_criteria, n_instances = matrix_df.shape
    print(f"  Matrix: {n_criteria} criteria x {n_instances} instances")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "preference_response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def build_criteria_metadata(scores_df, pref_df):
    """Build metadata for each unique criterion."""
    print("\nBuilding criteria metadata...")

    # From feedback collection
    feedback_stats = scores_df.groupby("criteria").agg(
        n_feedback=("score", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
    ).reset_index()

    # From preference collection
    pref_stats = pref_df.groupby("criteria").agg(
        n_preference=("preference", "count"),
        pref_b_rate=("preference", "mean"),
    ).reset_index()

    # Merge
    meta = feedback_stats.merge(pref_stats, on="criteria", how="outer")

    output_path = os.path.join(PROCESSED_DIR, "criteria_metadata.csv")
    meta.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Total unique criteria: {len(meta)}")

    return meta


def print_statistics(feedback_matrix, pref_matrix, scores_df, pref_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  PROMETHEUS STATISTICS")
    print(f"{'='*60}")

    # Feedback-Collection stats
    n_crit, n_inst = feedback_matrix.shape
    total_cells = n_crit * n_inst
    n_valid = feedback_matrix.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Feedback-Collection matrix:")
    print(f"    Criteria:      {n_crit}")
    print(f"    Instances:     {n_inst}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Score distribution
    all_scores = feedback_matrix.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Normalized score distribution [0, 1]:")
        print(f"    Mean:   {np.mean(valid_scores):.3f}")
        print(f"    Median: {np.median(valid_scores):.3f}")
        print(f"    Std:    {np.std(valid_scores):.3f}")

    # Raw score histogram
    print(f"\n  Raw score distribution (1-5):")
    for s in range(1, 6):
        count = (scores_df["score"] == s).sum()
        pct = count / len(scores_df) * 100
        bar = "#" * int(pct)
        print(f"    {s}: {count:8,} ({pct:5.1f}%) {bar}")

    # Preference-Collection stats
    n_crit_p, n_inst_p = pref_matrix.shape
    total_cells_p = n_crit_p * n_inst_p
    n_valid_p = pref_matrix.notna().sum().sum()

    print(f"\n  Preference-Collection matrix:")
    print(f"    Criteria:  {n_crit_p}")
    print(f"    Instances: {n_inst_p}")
    print(f"    Valid:     {n_valid_p:,}")

    # Preference distribution
    pref_b_rate = pref_df["preference"].mean()
    print(f"\n  Preference distribution:")
    print(f"    A preferred: {(1 - pref_b_rate)*100:.1f}%")
    print(f"    B preferred: {pref_b_rate*100:.1f}%")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("Prometheus Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Load Feedback-Collection
    print("STEP 1: Loading Feedback-Collection")
    print("-" * 60)
    scores_df = load_feedback_collection()

    # Step 2: Load Preference-Collection
    print("\nSTEP 2: Loading Preference-Collection")
    print("-" * 60)
    pref_df = load_preference_collection()

    # Step 3: Build feedback response matrix
    print("\nSTEP 3: Building Feedback-Collection response matrix")
    print("-" * 60)
    feedback_matrix = build_feedback_matrix(scores_df)

    # Step 4: Build preference response matrix
    print("\nSTEP 4: Building Preference-Collection response matrix")
    print("-" * 60)
    pref_matrix = build_preference_matrix(pref_df)

    # Step 5: Build criteria metadata
    print("\nSTEP 5: Building criteria metadata")
    print("-" * 60)
    build_criteria_metadata(scores_df, pref_df)

    # Step 6: Statistics
    print("\nSTEP 6: Detailed statistics")
    print("-" * 60)
    print_statistics(feedback_matrix, pref_matrix, scores_df, pref_df)


if __name__ == "__main__":
    main()
