#!/usr/bin/env python3
"""01_build_response_matrix.py — Download and process NVIDIA AEGIS AI Safety dataset.

Downloads the NVIDIA Aegis 2.0 AI Content Safety Dataset from HuggingFace.
Source: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
Paper: "Aegis: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts"
       (Ghosh et al., NVIDIA, 2024)

Loads the HuggingFace datasets-format data from raw/dataset/.
Each row has a prompt + response with safety labels and violated_categories.
Builds:
  1. Prompt x hazard-category binary matrix
  2. Summary statistics on safety label distributions
  3. Category co-occurrence matrix

Saves outputs to processed/.
"""

INFO = {
    'description': 'Download and process NVIDIA AEGIS AI Safety dataset',
    'testing_condition': """Subjects are human-written prompts (not models) and items are the 13 NVIDIA AEGIS hazard categories. A cell is 1 if human annotators labeled the prompt as belonging to that hazard category. The `_unsafe_only` variant restricts to prompts labeled unsafe. Use this dataset for item analysis of the hazard taxonomy itself, not to benchmark model safety.""",
    'paper_url': 'https://arxiv.org/abs/2501.09004',
    'data_source_url': 'https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0',
    'subject_type': 'prompt',
    'item_type': 'hazard_category',
    'license': 'CC-BY-4.0',
    'citation': """@misc{ghosh2025aegis20diverseaisafety,
      title={Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails}, 
      author={Shaona Ghosh and Prasoon Varshney and Makesh Narsimhan Sreedhar and Aishwarya Padmakumar and Traian Rebedea and Jibin Rajan Varghese and Christopher Parisien},
      year={2025},
      eprint={2501.09004},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.09004}, 
}""",
    'tags': ['safety'],
}


import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Known AEGIS hazard categories (13 categories from NVIDIA's taxonomy)
# ---------------------------------------------------------------------------
AEGIS_CATEGORIES = [
    "Criminal Planning/Confessions",
    "Controlled/Regulated Substances",
    "Guns/Illegal Weapons",
    "Harassment",
    "Hate/Identity Hate",
    "Need for Caution",
    "PII/Privacy",
    "Self-Harm",
    "Sexual",
    "Sexual (minor)",
    "Suicide",
    "Threat",
    "Violence",
]


def download():
    """Download NVIDIA Aegis 2.0 AI Content Safety Dataset from HuggingFace."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "dataset"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading nvidia/Aegis-AI-Content-Safety-Dataset-2.0...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download()

    print("=" * 70)
    print("AEGIS AI Safety Dataset Processing")
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

    first_split = list(ds.keys())[0]
    print(f"\nColumns: {ds[first_split].column_names}")
    print(f"\nFirst 3 examples:")
    for i in range(min(3, len(ds[first_split]))):
        print(f"  [{i}]:")
        for k, v in ds[first_split][i].items():
            print(f"    {k}: {str(v)[:120]}")

    # ------------------------------------------------------------------
    # 2. Convert to DataFrames and combine
    # ------------------------------------------------------------------
    dfs = {}
    for split_name in ds:
        df = ds[split_name].to_pandas()
        df["_split"] = split_name
        dfs[split_name] = df
        print(f"\n--- Split: {split_name} ---")
        print(f"  Shape: {df.shape}")

    combined = pd.concat(dfs.values(), ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")

    # ------------------------------------------------------------------
    # 3. Detect key columns
    # ------------------------------------------------------------------
    col_names = combined.columns.tolist()

    prompt_label_col = None
    for cand in ["prompt_label", "prompt_safety", "prompt_class"]:
        if cand in col_names:
            prompt_label_col = cand
            break

    response_label_col = None
    for cand in ["response_label", "response_safety", "response_class"]:
        if cand in col_names:
            response_label_col = cand
            break

    categories_col = None
    for cand in ["violated_categories", "categories", "hazard_categories", "violated_category"]:
        if cand in col_names:
            categories_col = cand
            break

    id_col = None
    for cand in ["id", "idx", "index", "sample_id"]:
        if cand in col_names:
            id_col = cand
            break

    print(f"\nDetected columns:")
    print(f"  prompt_label:       {prompt_label_col}")
    print(f"  response_label:     {response_label_col}")
    print(f"  violated_categories: {categories_col}")
    print(f"  id:                 {id_col}")

    # ------------------------------------------------------------------
    # 4. Parse violated categories
    # ------------------------------------------------------------------
    if categories_col:
        def parse_categories(val):
            """Parse violated_categories field into a list of category strings."""
            if pd.isna(val) or str(val).strip().lower() in ("none", "nan", "safe", ""):
                return []
            val_str = str(val).strip()
            # Could be comma-separated
            cats = [c.strip() for c in val_str.split(",") if c.strip()]
            return cats

        combined["_parsed_categories"] = combined[categories_col].apply(parse_categories)

        # Discover all unique categories
        all_cats = set()
        for cat_list in combined["_parsed_categories"]:
            all_cats.update(cat_list)
        all_cats = sorted(all_cats)
        print(f"\nDiscovered {len(all_cats)} unique hazard categories:")
        for cat in all_cats:
            count = sum(1 for cl in combined["_parsed_categories"] if cat in cl)
            print(f"  {cat}: {count}")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for split_name, df in dfs.items():
        mask = combined["_split"] == split_name
        sub = combined[mask]
        n = len(sub)
        print(f"\n--- {split_name} (n={n}) ---")

        if prompt_label_col:
            print(f"  Prompt labels:")
            print(sub[prompt_label_col].value_counts().to_string())

        if response_label_col:
            print(f"  Response labels:")
            print(sub[response_label_col].value_counts().to_string())

        if categories_col:
            has_violation = sub["_parsed_categories"].apply(len).gt(0).sum()
            print(f"  Samples with violations: {has_violation} ({100*has_violation/n:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Build prompt x hazard-category binary matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING RESPONSE MATRICES")
    print("=" * 70)

    if categories_col:
        # Use either discovered categories or the known AEGIS list
        # Prefer discovered so we capture what is actually in the data
        categories_to_use = all_cats if all_cats else AEGIS_CATEGORIES
        print(f"\nUsing {len(categories_to_use)} categories for binary matrix")

        # Create an ID for each sample
        if id_col:
            combined["_sample_id"] = combined[id_col].astype(str)
        else:
            combined["_sample_id"] = [f"sample_{i}" for i in range(len(combined))]

        # Build binary matrix: rows = samples, columns = categories
        binary_data = {}
        for cat in categories_to_use:
            binary_data[cat] = combined["_parsed_categories"].apply(lambda cl, c=cat: 1 if c in cl else 0).values

        binary_matrix = pd.DataFrame(binary_data, index=combined["_sample_id"].values)

        # Remove rows that have no violations at all (all zeros) -- keep for completeness
        print(f"\n  Full binary matrix (sample x category): {binary_matrix.shape}")
        print(f"  Samples with at least 1 violation: {(binary_matrix.sum(axis=1) > 0).sum()}")
        print(f"  Category totals:")
        print(binary_matrix.sum().sort_values(ascending=False).to_string())

        out_path = PROCESSED_DIR / "response_matrix.csv"
        binary_matrix.to_csv(out_path)
        print(f"  Saved to: {out_path}")

        # Save item content
        items = pd.DataFrame({
            "item_id": binary_matrix.columns,
            "content": binary_matrix.columns,
        })
        items.to_csv(PROCESSED_DIR / "item_content.csv", index=False)
        print(f"Saved item_content.csv ({len(items)} items)")

        # Also save a smaller version: only samples with violations
        unsafe_matrix = binary_matrix[binary_matrix.sum(axis=1) > 0]
        out_path = PROCESSED_DIR / "response_matrix_unsafe_only.csv"
        unsafe_matrix.to_csv(out_path)
        print(f"\n  Unsafe-only matrix: {unsafe_matrix.shape}")
        print(f"  Saved to: {out_path}")

        # Category co-occurrence matrix
        cooccurrence = binary_matrix.T.dot(binary_matrix)
        out_path = PROCESSED_DIR / "category_cooccurrence.csv"
        cooccurrence.to_csv(out_path)
        print(f"\n  Category co-occurrence matrix: {cooccurrence.shape}")
        print(f"  Saved to: {out_path}")
        print(cooccurrence.to_string())

    # ------------------------------------------------------------------
    # 7. Prompt label x Response label cross-tab
    # ------------------------------------------------------------------
    if prompt_label_col and response_label_col:
        ct = pd.crosstab(
            combined[prompt_label_col],
            combined[response_label_col],
            margins=True,
            margins_name="TOTAL",
        )
        out_path = PROCESSED_DIR / "prompt_x_response_label.csv"
        ct.to_csv(out_path)
        print(f"\n  Prompt label x Response label:")
        print(ct.to_string())
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 8. Save clean labeled CSV
    # ------------------------------------------------------------------
    # Drop the heavy text columns for summary, keep labels
    label_cols = [c for c in [
        id_col, prompt_label_col, response_label_col, categories_col, "_split",
    ] if c is not None and c in combined.columns]
    label_df = combined[label_cols].copy()
    out_path = PROCESSED_DIR / "aegis_labels.csv"
    label_df.to_csv(out_path, index=False)
    print(f"\n  Labels CSV: {label_df.shape}")
    print(f"  Saved to: {out_path}")

    # Save full data too
    out_path = PROCESSED_DIR / "aegis_full.csv"
    # Drop the parsed categories helper column
    save_cols = [c for c in combined.columns if not c.startswith("_")]
    combined[save_cols].to_csv(out_path, index=False)
    print(f"\n  Full data CSV: {combined[save_cols].shape}")
    print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 9. Summary statistics CSV
    # ------------------------------------------------------------------
    stats_rows = []
    for split_name in ds:
        mask = combined["_split"] == split_name
        sub = combined[mask]
        n = len(sub)
        row = {"split": split_name, "n_samples": n}
        if prompt_label_col:
            row["n_unsafe_prompts"] = int((sub[prompt_label_col] == "unsafe").sum())
            row["pct_unsafe_prompts"] = row["n_unsafe_prompts"] / n * 100
        if response_label_col:
            row["n_unsafe_responses"] = int((sub[response_label_col] == "unsafe").sum())
            row["pct_unsafe_responses"] = row["n_unsafe_responses"] / n * 100
        if categories_col:
            row["n_with_violations"] = int(sub["_parsed_categories"].apply(len).gt(0).sum())
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics:")
    print(stats_df.to_string(index=False))
    print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("AEGIS processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


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
