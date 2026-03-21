"""
Build RewardBench 2 response matrices from per-judge per-item evaluation results.

Data sources:
  - allenai/reward-bench-2: 1,865 (prompt, chosen, rejected) evaluation triples
    across 6 domains: Factuality, Focus, Math, Precise IF, Safety, Ties.
  - allenai/reward-bench-2-results: Per-judge binary outcomes (correct/incorrect
    preference ranking) and raw reward scores for each item.

Structure:
  - eval-set/: Summary JSON files with per-domain accuracy per judge (~197 judges).
  - eval-set-scores/: Per-item score JSON files with binary results and raw reward
    scores (~188 judges with per-item data).

Score format:
  - Binary result: 1.0 (correctly ranked chosen > rejected) or 0.0 (failed)
  - Raw scores: List of reward model scores per completion (model-specific scale)

Outputs:
  - response_matrix.csv: Binary results (judges x items) for all judges with
    per-item data
  - response_matrix_by_domain.csv: Per-domain summary accuracy for all judges
  - item_metadata.csv: Per-item metadata (domain, num_correct, num_incorrect)
  - judge_summary.csv: Per-judge aggregate statistics and metadata
"""

import json
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

SRC_RESULTS_REPO = "allenai/reward-bench-2-results"
SRC_ITEMS_REPO = "allenai/reward-bench-2"

DOMAINS = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties"]


def download_result_files():
    """Download all per-judge result files from HuggingFace Hub."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)

    # List all files
    files = list(api.list_repo_tree(SRC_RESULTS_REPO, repo_type="dataset", recursive=True))
    summary_files = [
        f.path for f in files
        if hasattr(f, "size") and f.path.startswith("eval-set/") and f.path.endswith(".json")
    ]
    score_files = [
        f.path for f in files
        if hasattr(f, "size") and f.path.startswith("eval-set-scores/") and f.path.endswith(".json")
    ]

    print(f"Found {len(summary_files)} summary files, {len(score_files)} score files")

    # Download summary files
    summary_dir = os.path.join(RAW_DIR, "eval-set")
    os.makedirs(summary_dir, exist_ok=True)
    print(f"\nDownloading summary files to {summary_dir} ...")

    summaries = {}
    for i, fpath in enumerate(summary_files):
        try:
            local = hf_hub_download(
                SRC_RESULTS_REPO, fpath, repo_type="dataset", token=HF_TOKEN
            )
            with open(local) as f:
                data = json.load(f)
            model_name = _model_name_from_path(fpath)
            summaries[model_name] = data

            # Save locally
            parts = fpath.split("/")
            local_dir = os.path.join(summary_dir, parts[1])
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, parts[2])
            with open(local_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: failed {fpath}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(summary_files)}")

    # Download score files
    scores_dir = os.path.join(RAW_DIR, "eval-set-scores")
    os.makedirs(scores_dir, exist_ok=True)
    print(f"\nDownloading score files to {scores_dir} ...")

    scores = {}
    for i, fpath in enumerate(score_files):
        try:
            local = hf_hub_download(
                SRC_RESULTS_REPO, fpath, repo_type="dataset", token=HF_TOKEN
            )
            with open(local) as f:
                data = json.load(f)
            model_name = _model_name_from_path(
                fpath.replace("eval-set-scores/", "eval-set/")
            )
            scores[model_name] = data
        except Exception as e:
            print(f"  Warning: failed {fpath}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(score_files)}")

    print(f"\nDownloaded {len(summaries)} summaries, {len(scores)} score files")
    return summaries, scores


def download_item_metadata():
    """Download item metadata from the main RewardBench 2 dataset."""
    print("\nDownloading item metadata from HuggingFace ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SRC_ITEMS_REPO, split="test", token=HF_TOKEN)
        rows = []
        for item in ds:
            rows.append({
                "id": str(item["id"]),
                "subset": item["subset"],
                "num_correct": item["num_correct"],
                "num_incorrect": item["num_incorrect"],
                "total_completions": item["total_completions"],
            })
        meta_df = pd.DataFrame(rows)
        meta_path = os.path.join(RAW_DIR, "item_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  Saved {len(meta_df)} items to {meta_path}")
        return meta_df
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def _model_name_from_path(path: str) -> str:
    """Extract model name from path like 'eval-set/openai/gpt-4o.json'."""
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[1] + "/" + parts[2].replace(".json", "")
    return path


def build_response_matrix(scores):
    """Build binary response matrix from per-item score data."""
    # Determine item ordering from first judge
    first_judge = next(iter(scores.values()))
    item_ids = [str(x) for x in first_judge["id"]]
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Sort judges
    judge_names = sorted(scores.keys())
    n_judges = len(judge_names)

    print(f"\n{'='*60}")
    print(f"  Building response matrix: {n_judges} judges x {n_items} items")
    print(f"{'='*60}")

    # Build matrix
    matrix = np.full((n_judges, n_items), np.nan)
    for j, judge_name in enumerate(judge_names):
        judge_data = scores[judge_name]
        judge_ids = [str(x) for x in judge_data["id"]]
        judge_results = judge_data["results"]
        for k, iid in enumerate(judge_ids):
            if iid in item_id_to_idx:
                matrix[j, item_id_to_idx[iid]] = float(judge_results[k])

    # Create DataFrame
    matrix_df = pd.DataFrame(matrix, index=judge_names, columns=item_ids)
    matrix_df.index.name = "judge"

    # Statistics
    total_cells = n_judges * n_items
    n_valid = np.sum(~np.isnan(matrix))
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells

    print(f"  Judges:        {n_judges}")
    print(f"  Items:         {n_items}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:     {fill_rate*100:.1f}%")

    # Accuracy stats
    valid_vals = matrix[~np.isnan(matrix)]
    mean_acc = np.mean(valid_vals)
    print(f"\n  Overall accuracy: {mean_acc:.4f}")

    # Per-judge stats
    per_judge_acc = np.nanmean(matrix, axis=1)
    print(f"\n  Per-judge accuracy:")
    print(f"    Best:   {per_judge_acc.max():.4f} ({judge_names[np.argmax(per_judge_acc)]})")
    print(f"    Worst:  {per_judge_acc.min():.4f} ({judge_names[np.argmin(per_judge_acc)]})")
    print(f"    Median: {np.median(per_judge_acc):.4f}")
    print(f"    Std:    {np.std(per_judge_acc):.4f}")

    # Per-item stats
    per_item_acc = np.nanmean(matrix, axis=0)
    easy = np.sum(per_item_acc >= 0.8)
    medium = np.sum((per_item_acc >= 0.4) & (per_item_acc < 0.8))
    hard = np.sum(per_item_acc < 0.4)
    print(f"\n  Item difficulty distribution:")
    print(f"    Easy (>=80%):   {easy}")
    print(f"    Medium (40-80%): {medium}")
    print(f"    Hard (<40%):    {hard}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    return matrix_df, item_ids, judge_names


def build_domain_summary(summaries, scores):
    """Build per-domain accuracy summary for all judges."""
    rows = []
    for model_name in sorted(summaries.keys()):
        data = summaries[model_name]
        row = {"judge": model_name}
        row["model_type"] = data.get("model_type", "")
        for domain in DOMAINS:
            row[domain] = data.get(domain, np.nan)
        # Overall accuracy
        domain_vals = [data.get(d, np.nan) for d in DOMAINS if d in data]
        row["overall"] = np.nanmean(domain_vals) if domain_vals else np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("overall", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "response_matrix_by_domain.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Domain summary saved: {output_path}")

    print(f"\n  Top 10 judges (by overall accuracy):")
    for _, r in summary_df.head(10).iterrows():
        print(f"    {r['judge']:55s}  overall={r['overall']:.4f}  type={r['model_type']}")

    return summary_df


def build_judge_summary(matrix_df, summaries):
    """Build per-judge summary statistics."""
    rows = []
    for judge in matrix_df.index:
        judge_row = matrix_df.loc[judge]
        summary = summaries.get(judge, {})
        rows.append({
            "judge": judge,
            "model_type": summary.get("model_type", ""),
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


def main():
    print("RewardBench 2 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading evaluation results")
    print("-" * 60)
    summaries, scores = download_result_files()

    # Step 2: Download item metadata
    print("\nSTEP 2: Downloading item metadata")
    print("-" * 60)
    item_meta_df = download_item_metadata()

    # Step 3: Build response matrix
    print("\nSTEP 3: Building response matrix")
    print("-" * 60)
    matrix_df, item_ids, judge_names = build_response_matrix(scores)

    # Step 4: Build domain summary
    print("\nSTEP 4: Building domain summary")
    print("-" * 60)
    domain_df = build_domain_summary(summaries, scores)

    # Step 5: Build judge summary
    print("\nSTEP 5: Building judge summary")
    print("-" * 60)
    judge_df = build_judge_summary(matrix_df, summaries)

    # Step 6: Save enriched item metadata
    print("\nSTEP 6: Saving enriched item metadata")
    print("-" * 60)
    if item_meta_df is not None:
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

        # Domain breakdown
        print(f"\n  Per-domain accuracy (across all judges):")
        for domain in DOMAINS:
            domain_items = merged[merged["subset"] == domain]
            mean_acc = domain_items["mean_accuracy"].mean()
            print(f"    {domain:15s}  n={len(domain_items):4d}  mean_acc={mean_acc:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Response matrix: {len(judge_names)} judges x {len(item_ids)} items")
    print(f"  Domains: {', '.join(DOMAINS)}")
    print(f"  Domain summary: {len(domain_df)} judges with domain-level scores")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
