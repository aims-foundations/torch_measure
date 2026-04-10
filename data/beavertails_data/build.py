"""
Build BeaverTails binary safety classification response matrices.

Data source:
  - PKU-Alignment/BeaverTails on HuggingFace Hub: 330K+ QA pairs with
    human-annotated safety labels across 14 harm categories.

  The dataset has 4 splits:
      330k_train (301K rows), 330k_test (33.4K rows),
      30k_train (27.2K rows), 30k_test (3.02K rows).

  We use the 330k_test split (33,432 QA pairs) for the response matrix.

Fields per example:
  - prompt (str): The input question/query.
  - response (str): The generated response.
  - is_safe (bool): Overall binary safety label.
  - category (dict): 14 harm category booleans (True = unsafe).

The 14 harm categories:
  animal_abuse, child_abuse, controversial_topics/politics,
  discrimination/stereotype/injustice, drug_abuse/weapons/banned_substance,
  financial_crime/property_crime/theft, hate_speech/offensive_language,
  misinformation_regarding_ethics/laws_and_safety,
  non_violent_unethical_behavior, privacy_violation, self_harm,
  sexually_explicit/adult_content, terrorism/organized_crime,
  violence/aiding_and_abetting/incitement.

Outputs:
  - processed/response_matrix.csv: Binary matrix (15 categories x N items),
    rows = harm categories + overall, columns = QA pair IDs.
  - processed/item_metadata.csv: Per-item metadata (prompt, response, is_safe).
  - processed/category_stats.csv: Per-category aggregate statistics.
"""

import os

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

SRC_REPO = "PKU-Alignment/BeaverTails"

HARM_CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def download_data():
    """Download BeaverTails 330k_test split from HuggingFace Hub."""
    from datasets import load_dataset

    print("Downloading PKU-Alignment/BeaverTails (330k_test split) ...")
    ds = load_dataset(SRC_REPO, split="330k_test", token=HF_TOKEN)
    print(f"  Loaded {len(ds):,} QA pairs")
    return ds


def build_response_matrix(ds):
    """Build binary safety classification response matrix from BeaverTails data.

    Returns the response matrix DataFrame, item metadata DataFrame, and
    category statistics.
    """
    n_items = len(ds)
    subject_ids = ["overall"] + HARM_CATEGORIES
    n_subjects = len(subject_ids)

    print(f"\n{'='*60}")
    print(f"  Building response matrix: {n_subjects} classifiers x {n_items:,} items")
    print(f"{'='*60}")

    # Build matrix
    matrix = np.zeros((n_subjects, n_items), dtype=np.float64)
    item_rows = []

    for i, example in enumerate(ds):
        # Overall: 1 = unsafe (is_safe=False), 0 = safe (is_safe=True)
        matrix[0, i] = 0.0 if example["is_safe"] else 1.0

        # Per-category labels
        category_dict = example["category"]
        for j, cat in enumerate(HARM_CATEGORIES):
            matrix[j + 1, i] = 1.0 if category_dict.get(cat, False) else 0.0

        # Item metadata
        item_rows.append({
            "item_id": f"bt_{i:06d}",
            "prompt": example["prompt"],
            "response": example["response"],
            "is_safe": example["is_safe"],
            "n_categories_flagged": sum(
                1 for cat in HARM_CATEGORIES if category_dict.get(cat, False)
            ),
        })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,}/{n_items:,} ...")

    item_ids = [f"bt_{i:06d}" for i in range(n_items)]

    # Create DataFrames
    matrix_df = pd.DataFrame(matrix, index=subject_ids, columns=item_ids)
    matrix_df.index.name = "category"
    item_meta_df = pd.DataFrame(item_rows)

    # Statistics
    total_cells = n_subjects * n_items
    n_flagged = int(np.sum(matrix))
    print(f"\n  Total cells:    {total_cells:,}")
    print(f"  Flagged cells:  {n_flagged:,} ({n_flagged / total_cells * 100:.1f}%)")

    # Overall safety stats
    n_unsafe = int(matrix[0].sum())
    n_safe = n_items - n_unsafe
    print(f"\n  Overall safety:")
    print(f"    Safe:   {n_safe:,} ({n_safe / n_items * 100:.1f}%)")
    print(f"    Unsafe: {n_unsafe:,} ({n_unsafe / n_items * 100:.1f}%)")

    # Per-category stats
    cat_stats = []
    print(f"\n  Per-category breakdown:")
    for j, cat in enumerate(HARM_CATEGORIES):
        n_cat_flagged = int(matrix[j + 1].sum())
        pct = n_cat_flagged / n_items * 100
        print(f"    {cat:55s}  {n_cat_flagged:6,} ({pct:5.1f}%)")
        cat_stats.append({
            "category": cat,
            "n_flagged": n_cat_flagged,
            "n_total": n_items,
            "pct_flagged": pct,
        })

    cat_stats_df = pd.DataFrame(cat_stats)

    # Multi-label statistics
    n_multi = int((item_meta_df["n_categories_flagged"] > 1).sum())
    n_single = int(
        ((item_meta_df["n_categories_flagged"] == 1)
         & (~item_meta_df["is_safe"])).sum()
    )
    n_no_cat = int(
        ((item_meta_df["n_categories_flagged"] == 0)
         & (~item_meta_df["is_safe"])).sum()
    )
    print(f"\n  Multi-label stats (among unsafe items):")
    print(f"    Single category:     {n_single:,}")
    print(f"    Multiple categories: {n_multi:,}")
    print(f"    No category flagged: {n_no_cat:,}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    return matrix_df, item_meta_df, cat_stats_df


def main():
    print("BeaverTails Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading BeaverTails from HuggingFace")
    print("-" * 60)
    ds = download_data()

    # Step 2: Build response matrix
    print("\nSTEP 2: Building response matrix")
    print("-" * 60)
    matrix_df, item_meta_df, cat_stats_df = build_response_matrix(ds)

    # Step 3: Save item metadata
    print("\nSTEP 3: Saving item metadata")
    print("-" * 60)
    item_meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    item_meta_df.to_csv(item_meta_path, index=False)
    print(f"  Saved: {item_meta_path}")

    # Step 4: Save category statistics
    print("\nSTEP 4: Saving category statistics")
    print("-" * 60)
    cat_stats_path = os.path.join(PROCESSED_DIR, "category_stats.csv")
    cat_stats_df.to_csv(cat_stats_path, index=False)
    print(f"  Saved: {cat_stats_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    n_subjects, n_items = matrix_df.shape
    print(f"  Response matrix: {n_subjects} classifiers x {n_items:,} QA pairs")
    print(f"  Harm categories: {len(HARM_CATEGORIES)}")
    print(f"  Items: {n_items:,}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
