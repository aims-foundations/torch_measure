"""
Build RewardBench response matrices from per-model per-item evaluation scores.

Data sources:
  - allenai/reward-bench-results HuggingFace dataset repo:
    eval-set-scores/{org}/{model}.json files contain per-item binary (0/1)
    results for each reward model on the 2,985-item RewardBench eval set.
  - allenai/reward-bench HuggingFace dataset: Core eval set with
    (prompt, chosen, rejected) trios and subset labels.

Score format:
  - Binary 0/1: whether the reward model correctly ranked chosen > rejected.

RewardBench subsets (23 total, grouped into 4 categories):
  - Chat: alpacaeval-easy, alpacaeval-length, alpacaeval-hard,
          mt-bench-easy, mt-bench-med
  - Chat Hard: mt-bench-hard, llmbar-natural, llmbar-adver-neighbor,
               llmbar-adver-GPTInst, llmbar-adver-GPTOut, llmbar-adver-manual
  - Safety: refusals-dangerous, refusals-offensive, xstest-should-refuse,
            xstest-should-respond, donotanswer
  - Reasoning: math-prm, hep-cpp, hep-go, hep-java, hep-js, hep-python,
               hep-rust

Outputs:
  - raw/{org}/{model}.json: Raw JSON per model (cached)
  - processed/response_matrix.csv: Binary response matrix (models x items)
  - processed/item_metadata.csv: Per-item metadata (subset, category, id)
  - processed/model_summary.csv: Per-model aggregate statistics
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# HuggingFace endpoints
HF_RESULTS_REPO = "allenai/reward-bench-results"
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"

# Reference model (has 'id' field) for constructing canonical item IDs
REFERENCE_MODEL_PATH = "eval-set-scores/openai/gpt-4o-2024-05-13.json"

# Subset -> category mapping
SUBSET_CATEGORIES = {
    "alpacaeval-easy": "chat",
    "alpacaeval-length": "chat",
    "alpacaeval-hard": "chat",
    "mt-bench-easy": "chat",
    "mt-bench-med": "chat",
    "mt-bench-hard": "chat_hard",
    "llmbar-natural": "chat_hard",
    "llmbar-adver-neighbor": "chat_hard",
    "llmbar-adver-GPTInst": "chat_hard",
    "llmbar-adver-GPTOut": "chat_hard",
    "llmbar-adver-manual": "chat_hard",
    "refusals-dangerous": "safety",
    "refusals-offensive": "safety",
    "xstest-should-refuse": "safety",
    "xstest-should-respond": "safety",
    "donotanswer": "safety",
    "math-prm": "reasoning",
    "hep-cpp": "reasoning",
    "hep-go": "reasoning",
    "hep-java": "reasoning",
    "hep-js": "reasoning",
    "hep-python": "reasoning",
    "hep-rust": "reasoning",
}


def download_file(url, dest_path, retries=3, delay=1.0):
    """Download a file from URL with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    FAILED after {retries} attempts: {e}")
                return False
    return False


def enumerate_model_files():
    """List all JSON files under eval-set-scores/ via HF API."""
    print("Enumerating model files from HuggingFace API ...")
    tree_url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/eval-set-scores"

    req = urllib.request.Request(tree_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        orgs_data = json.loads(resp.read())
    orgs = [d["path"] for d in orgs_data if d["type"] == "directory"]
    print(f"  Found {len(orgs)} organizations")

    all_files = []
    for org_path in orgs:
        url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/{org_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files_data = json.loads(resp.read())
        files = [
            f["path"]
            for f in files_data
            if f["type"] == "file" and f["path"].endswith(".json")
        ]
        all_files.extend(files)

    print(f"  Found {len(all_files)} total JSON files")
    return all_files


def download_all_models(all_files):
    """Download all model JSON files to raw/ directory (with caching)."""
    print(f"\nDownloading {len(all_files)} model files ...")
    downloaded = 0
    skipped = 0
    failed = 0

    for i, fpath in enumerate(all_files):
        # Mirror the HF directory structure in raw/
        # eval-set-scores/org/model.json -> raw/org/model.json
        rel_path = fpath.replace("eval-set-scores/", "")
        dest_path = os.path.join(RAW_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Skip if already cached
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            skipped += 1
            continue

        url = f"{HF_RESOLVE_BASE}/{HF_RESULTS_REPO}/resolve/main/{fpath}"
        success = download_file(url, dest_path)
        if success:
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{len(all_files)}")
            time.sleep(0.3)

    print(f"  Downloaded: {downloaded}, Cached: {skipped}, Failed: {failed}")


def get_reference_item_data():
    """Load reference model to extract canonical item IDs and subsets."""
    # Check if reference model is cached locally
    ref_rel = REFERENCE_MODEL_PATH.replace("eval-set-scores/", "")
    ref_local = os.path.join(RAW_DIR, ref_rel)

    if os.path.exists(ref_local) and os.path.getsize(ref_local) > 100:
        with open(ref_local) as f:
            data = json.load(f)
    else:
        url = f"{HF_RESOLVE_BASE}/{HF_RESULTS_REPO}/resolve/main/{REFERENCE_MODEL_PATH}"
        print(f"  Downloading reference model from {url}")
        dest_dir = os.path.dirname(ref_local)
        os.makedirs(dest_dir, exist_ok=True)
        download_file(url, ref_local)
        with open(ref_local) as f:
            data = json.load(f)

    ids = data["id"]
    subsets = data["subset"]
    item_ids = [f"{subset}:{item_id}" for subset, item_id in zip(subsets, ids)]
    return item_ids, ids, subsets


def build_response_matrix(all_files, ref_subsets):
    """Build the response matrix from downloaded JSON files."""
    n_items = len(ref_subsets)
    model_names = []
    model_types = []
    rows = []
    skipped = 0

    for fpath in all_files:
        rel_path = fpath.replace("eval-set-scores/", "")
        local_path = os.path.join(RAW_DIR, rel_path)

        if not os.path.exists(local_path):
            skipped += 1
            continue

        try:
            with open(local_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"    WARNING: Failed to parse {rel_path}: {e}")
            skipped += 1
            continue

        results = data.get("results")
        if results is None or len(results) != n_items:
            print(
                f"    WARNING: {rel_path} has {len(results) if results else 0} "
                f"results (expected {n_items}), skipping"
            )
            skipped += 1
            continue

        # Verify subset ordering matches reference
        file_subsets = data.get("subset", [])
        if file_subsets and file_subsets != ref_subsets:
            print(f"    WARNING: {rel_path} has different subset ordering, skipping")
            skipped += 1
            continue

        model_name = data.get("model", rel_path.replace(".json", ""))
        model_type = data.get("model_type", "Unknown")

        model_names.append(model_name)
        model_types.append(model_type)
        rows.append([int(r) for r in results])

    print(f"  Loaded: {len(model_names)}, Skipped: {skipped}")
    return model_names, model_types, rows


def main():
    print("RewardBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Enumerate files
    print("STEP 1: Enumerating model files from HuggingFace")
    print("-" * 60)
    all_files = enumerate_model_files()

    # Step 2: Download all model files
    print("\nSTEP 2: Downloading model JSON files")
    print("-" * 60)
    download_all_models(all_files)

    # Step 3: Get reference item IDs
    print("\nSTEP 3: Extracting reference item IDs")
    print("-" * 60)
    item_ids, raw_ids, ref_subsets = get_reference_item_data()
    print(f"  {len(item_ids)} items across {len(set(ref_subsets))} subsets")

    # Step 4: Build response matrix
    print("\nSTEP 4: Building response matrix")
    print("-" * 60)
    model_names, model_types, rows = build_response_matrix(all_files, ref_subsets)

    if not rows:
        print("ERROR: No valid model data found!")
        sys.exit(1)

    # Convert to DataFrame (models as rows, items as columns)
    matrix_df = pd.DataFrame(rows, index=model_names, columns=item_ids)
    matrix_df.index.name = "Model"

    n_models = len(model_names)
    n_items = len(item_ids)
    total_cells = n_models * n_items

    print(f"\n{'='*60}")
    print(f"  RESPONSE MATRIX STATISTICS")
    print(f"{'='*60}")
    print(f"  Models (subjects):   {n_models}")
    print(f"  Items:               {n_items}")
    print(f"  Matrix dimensions:   {n_models} x {n_items}")
    print(f"  Total cells:         {total_cells:,}")
    print(f"  All values binary:   {set(matrix_df.values.flatten()) == {0, 1}}")

    # Overall accuracy
    overall_acc = matrix_df.values.mean()
    print(f"\n  Overall accuracy:    {overall_acc:.4f} ({overall_acc*100:.1f}%)")

    # Per-model statistics
    per_model_acc = matrix_df.mean(axis=1)
    print(f"\n  Per-model accuracy:")
    print(f"    Mean:   {per_model_acc.mean():.4f}")
    print(f"    Median: {per_model_acc.median():.4f}")
    print(f"    Std:    {per_model_acc.std():.4f}")
    print(f"    Min:    {per_model_acc.min():.4f} ({per_model_acc.idxmin()})")
    print(f"    Max:    {per_model_acc.max():.4f} ({per_model_acc.idxmax()})")

    # Per-item statistics
    per_item_acc = matrix_df.mean(axis=0)
    print(f"\n  Per-item accuracy (difficulty):")
    print(f"    Mean:   {per_item_acc.mean():.4f}")
    print(f"    Median: {per_item_acc.median():.4f}")
    print(f"    Std:    {per_item_acc.std():.4f}")
    print(f"    Min:    {per_item_acc.min():.4f}")
    print(f"    Max:    {per_item_acc.max():.4f}")

    # Difficulty distribution
    easy = (per_item_acc >= 0.9).sum()
    medium = ((per_item_acc >= 0.5) & (per_item_acc < 0.9)).sum()
    hard = (per_item_acc < 0.5).sum()
    print(f"\n  Item difficulty distribution:")
    print(f"    Easy (>=90%):     {easy}")
    print(f"    Medium (50-90%):  {medium}")
    print(f"    Hard (<50%):      {hard}")

    # Per-subset statistics
    print(f"\n  Per-subset accuracy:")
    subset_stats = []
    for subset in sorted(set(ref_subsets)):
        mask = [s == subset for s in ref_subsets]
        sub_cols = [item_ids[i] for i, m in enumerate(mask) if m]
        sub_acc = matrix_df[sub_cols].values.mean()
        n_sub_items = sum(mask)
        cat = SUBSET_CATEGORIES.get(subset, "?")
        subset_stats.append({
            "subset": subset,
            "category": cat,
            "n_items": n_sub_items,
            "accuracy": sub_acc,
        })
        print(f"    {subset:30s}  n={n_sub_items:4d}  acc={sub_acc:.3f}  ({cat})")

    # Per-category statistics
    print(f"\n  Per-category accuracy:")
    for cat in ["chat", "chat_hard", "safety", "reasoning"]:
        cat_subsets = [s for s, c in SUBSET_CATEGORIES.items() if c == cat]
        mask = [s in cat_subsets for s in ref_subsets]
        cat_cols = [item_ids[i] for i, m in enumerate(mask) if m]
        cat_acc = matrix_df[cat_cols].values.mean()
        n_cat_items = sum(mask)
        print(f"    {cat:15s}  n={n_cat_items:4d}  acc={cat_acc:.3f}")

    # Model type distribution
    from collections import Counter
    type_counts = Counter(model_types)
    print(f"\n  Model type distribution:")
    for mtype, count in type_counts.most_common():
        print(f"    {mtype:25s}  {count}")

    # Top and bottom models
    sorted_models = per_model_acc.sort_values(ascending=False)
    print(f"\n  Top 10 models (by accuracy):")
    for model, acc in sorted_models.head(10).items():
        print(f"    {model:55s}  {acc:.4f}")

    print(f"\n  Bottom 10 models (by accuracy):")
    for model, acc in sorted_models.tail(10).items():
        print(f"    {model:55s}  {acc:.4f}")

    # Step 5: Save outputs
    print(f"\n{'='*60}")
    print(f"  SAVING OUTPUTS")
    print(f"{'='*60}")

    # Save response matrix
    matrix_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(matrix_path)
    print(f"  Saved: {matrix_path}")
    print(f"    Shape: {matrix_df.shape}")

    # Save item metadata
    item_meta = pd.DataFrame({
        "item_id": item_ids,
        "raw_id": raw_ids,
        "subset": ref_subsets,
        "category": [SUBSET_CATEGORIES.get(s, "?") for s in ref_subsets],
        "mean_accuracy": [per_item_acc[iid] for iid in item_ids],
    })
    item_meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    item_meta.to_csv(item_meta_path, index=False)
    print(f"  Saved: {item_meta_path}")

    # Save model summary
    model_summary = pd.DataFrame({
        "model": model_names,
        "model_type": model_types,
        "accuracy": [per_model_acc[m] for m in model_names],
    })
    model_summary = model_summary.sort_values("accuracy", ascending=False)
    model_summary_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    model_summary.to_csv(model_summary_path, index=False)
    print(f"  Saved: {model_summary_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Response matrix: {n_models} models x {n_items} items (binary 0/1)")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
