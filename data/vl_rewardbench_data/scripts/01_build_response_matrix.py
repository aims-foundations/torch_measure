"""
Build VL-RewardBench response matrices from per-judge evaluation results.

Data source:
  - MMInstruction/VL-RewardBench: 1,247 (image, query, response_pair) evaluation
    pairs across 3 categories: General, Hallucination, Reasoning.
  - 16 VLM judges evaluated on preference ranking accuracy.

VL-RewardBench evaluates vision-language generative reward models (VL-GenRMs)
on their ability to correctly identify the human-preferred response in
image-text preference pairs.

Categories (derived from item ID prefixes):
  - General: WildVision, VLFeedback items (general multimodal instructions).
  - Hallucination: RLAIF-V, RLHF-V, POVID, LRVInstruction items
    (visual hallucination detection).
  - Reasoning: MathVerse, MMMU-Pro items (multimodal reasoning tasks).

Score format:
  - Binary result: 1.0 (correctly identified preferred response) or
    0.0 (failed to identify preferred response).

Outputs:
  - response_matrix.csv: Binary results (judges x items) for all judges.
  - item_metadata.csv: Per-item metadata (category, query_source, models).
  - judge_summary.csv: Per-judge aggregate statistics.
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

SRC_REPO = "MMInstruction/VL-RewardBench"

CATEGORIES = ["General", "Hallucination", "Reasoning"]

# The 16 judges evaluated in the paper (Table 2, arXiv:2411.17451).
JUDGES = [
    "LLaVA-OneVision-7B-ov",
    "InternVL2-8B",
    "Phi-3.5-Vision",
    "Qwen2-VL-7B",
    "Qwen2-VL-72B",
    "Llama-3.2-11B",
    "Llama-3.2-90B",
    "Molmo-7B",
    "Molmo-72B",
    "Pixtral-12B",
    "NVLM-D-72B",
    "Gemini-1.5-Flash",
    "Gemini-1.5-Pro",
    "Claude-3.5-Sonnet",
    "GPT-4o-mini",
    "GPT-4o",
]

# ---------------------------------------------------------------------------
# Category assignment from item ID prefix
# ---------------------------------------------------------------------------

_HALLUCINATION_PREFIXES = ("RLAIF-V-", "RLHF-V-", "hallucination_pair-", "LRVInstruction-")
_REASONING_PREFIXES = ("mathverse_", "mmmu_pro_")
_GENERAL_PREFIXES = ("wildvision-battle-", "VLFeedback-")


def _infer_category(item_id: str) -> str:
    """Infer the category (General, Hallucination, Reasoning) from item ID prefix."""
    lid = item_id.lower()
    for prefix in _HALLUCINATION_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "Hallucination"
    for prefix in _REASONING_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "Reasoning"
    for prefix in _GENERAL_PREFIXES:
        if lid.startswith(prefix.lower()):
            return "General"
    return "Unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_dataset():
    """Download VL-RewardBench dataset from HuggingFace Hub."""
    print("\nDownloading VL-RewardBench dataset from HuggingFace ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SRC_REPO, split="test", token=HF_TOKEN)
        rows = []
        for item in ds:
            item_id = str(item["id"])
            category = _infer_category(item_id)
            rows.append({
                "id": item_id,
                "query_source": item.get("query_source", ""),
                "models": str(item.get("models", [])),
                "human_ranking": str(item.get("human_ranking", [])),
                "judge": item.get("judge", ""),
                "category": category,
            })
        meta_df = pd.DataFrame(rows)
        meta_path = os.path.join(RAW_DIR, "item_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  Saved {len(meta_df)} items to {meta_path}")

        # Category breakdown
        for cat in CATEGORIES:
            n = len(meta_df[meta_df["category"] == cat])
            print(f"    {cat}: {n} items")
        n_unknown = len(meta_df[meta_df["category"] == "Unknown"])
        if n_unknown > 0:
            print(f"    Unknown: {n_unknown} items")

        return meta_df, ds
    except Exception as e:
        print(f"  Failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Response matrix building
# ---------------------------------------------------------------------------


def build_response_matrix(item_ids, judge_results):
    """Build binary response matrix from per-judge evaluation results.

    Parameters
    ----------
    item_ids : list[str]
        Ordered item IDs.
    judge_results : dict[str, dict[str, float]]
        {judge_name: {item_id: binary_result}}.

    Returns
    -------
    pd.DataFrame
        Response matrix (judges x items).
    """
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    judge_names = sorted(judge_results.keys())
    n_judges = len(judge_names)

    print(f"\n{'='*60}")
    print(f"  Building response matrix: {n_judges} judges x {n_items} items")
    print(f"{'='*60}")

    # Build matrix
    matrix = np.full((n_judges, n_items), np.nan)
    for j, judge_name in enumerate(judge_names):
        results = judge_results[judge_name]
        for iid, val in results.items():
            if iid in item_id_to_idx and val is not None:
                matrix[j, item_id_to_idx[iid]] = float(val)

    # Create DataFrame
    matrix_df = pd.DataFrame(matrix, index=judge_names, columns=item_ids)
    matrix_df.index.name = "judge"

    # Statistics
    total_cells = n_judges * n_items
    n_valid = np.sum(~np.isnan(matrix))
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"  Judges:        {n_judges}")
    print(f"  Items:         {n_items}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:     {fill_rate*100:.1f}%")

    if n_valid > 0:
        valid_vals = matrix[~np.isnan(matrix)]
        mean_acc = np.mean(valid_vals)
        print(f"\n  Overall accuracy: {mean_acc:.4f}")

        # Per-judge stats
        per_judge_acc = np.nanmean(matrix, axis=1)
        if not np.all(np.isnan(per_judge_acc)):
            best_idx = np.nanargmax(per_judge_acc)
            worst_idx = np.nanargmin(per_judge_acc)
            print(f"\n  Per-judge accuracy:")
            print(f"    Best:   {per_judge_acc[best_idx]:.4f} ({judge_names[best_idx]})")
            print(f"    Worst:  {per_judge_acc[worst_idx]:.4f} ({judge_names[worst_idx]})")
            print(f"    Median: {np.nanmedian(per_judge_acc):.4f}")
            print(f"    Std:    {np.nanstd(per_judge_acc):.4f}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    return matrix_df, item_ids, judge_names


def build_category_summary(matrix_df, item_meta_df):
    """Build per-category accuracy summary for all judges."""
    rows = []
    for judge in matrix_df.index:
        row = {"judge": judge}
        judge_row = matrix_df.loc[judge]
        for cat in CATEGORIES:
            cat_items = item_meta_df[item_meta_df["category"] == cat]["id"].tolist()
            cat_vals = judge_row[judge_row.index.isin(cat_items)]
            row[cat] = cat_vals.mean() if len(cat_vals) > 0 else np.nan
        row["overall"] = judge_row.mean()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("overall", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "response_matrix_by_category.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Category summary saved: {output_path}")

    print(f"\n  Top judges (by overall accuracy):")
    for _, r in summary_df.head(10).iterrows():
        print(f"    {r['judge']:30s}  overall={r['overall']:.4f}")

    return summary_df


def build_judge_summary(matrix_df):
    """Build per-judge summary statistics."""
    rows = []
    for judge in matrix_df.index:
        judge_row = matrix_df.loc[judge]
        rows.append({
            "judge": judge,
            "accuracy": judge_row.mean(),
            "n_items": judge_row.notna().sum(),
            "n_correct": (judge_row == 1.0).sum(),
        })

    judge_df = pd.DataFrame(rows)
    judge_df = judge_df.sort_values("accuracy", ascending=False)
    output_path = os.path.join(PROCESSED_DIR, "judge_summary.csv")
    judge_df.to_csv(output_path, index=False)
    print(f"\n  Judge summary saved: {output_path}")
    return judge_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("VL-RewardBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download dataset
    print("STEP 1: Downloading VL-RewardBench dataset")
    print("-" * 60)
    item_meta_df, ds = download_dataset()
    item_ids = item_meta_df["id"].tolist()

    # Step 2: Load or compute judge results
    # NOTE: Replace this section with actual result loading when
    # pre-computed evaluation results are available.
    print("\nSTEP 2: Loading judge evaluation results")
    print("-" * 60)
    print("  NOTE: Judge evaluation results should be loaded from")
    print("  pre-computed outputs. Initializing empty results for now.")

    judge_results: dict[str, dict[str, float]] = {}
    for judge in JUDGES:
        judge_results[judge] = {}
    print(f"  Initialized results for {len(judge_results)} judges")

    # Step 3: Build response matrix
    print("\nSTEP 3: Building response matrix")
    print("-" * 60)
    matrix_df, item_ids, judge_names = build_response_matrix(
        item_ids, judge_results
    )

    # Step 4: Build category summary
    print("\nSTEP 4: Building category summary")
    print("-" * 60)
    cat_df = build_category_summary(matrix_df, item_meta_df)

    # Step 5: Build judge summary
    print("\nSTEP 5: Building judge summary")
    print("-" * 60)
    judge_df = build_judge_summary(matrix_df)

    # Step 6: Save enriched item metadata
    print("\nSTEP 6: Saving enriched item metadata")
    print("-" * 60)
    per_item_acc = matrix_df.mean(axis=0)
    item_stats = pd.DataFrame({
        "id": item_ids,
        "mean_accuracy": [per_item_acc.get(iid, np.nan) for iid in item_ids],
        "n_judges": [matrix_df[iid].notna().sum() for iid in item_ids],
    })
    merged = item_stats.merge(item_meta_df, on="id", how="left")
    merged_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    merged.to_csv(merged_path, index=False)
    print(f"  Item metadata saved: {merged_path}")

    # Category breakdown
    print(f"\n  Per-category accuracy (across all judges):")
    for cat in CATEGORIES:
        cat_items = merged[merged["category"] == cat]
        mean_acc = cat_items["mean_accuracy"].mean()
        print(f"    {cat:20s}  n={len(cat_items):4d}  mean_acc={mean_acc:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Response matrix: {len(judge_names)} judges x {len(item_ids)} items")
    print(f"  Categories: {', '.join(CATEGORIES)}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
