"""
Build JudgeBench response matrices from per-judge evaluation results.

Data sources:
  - ScalerLab/JudgeBench on HuggingFace: 350 GPT pairs + 270 Claude pairs
    across 4 categories: knowledge, reasoning, math, coding.
  - ScalerLab/JudgeBench on GitHub: Per-judge evaluation outputs for 33 judges
    (prompted judges, fine-tuned judges, multi-agent judges, reward models).

Categories:
  - knowledge: 13 MMLU-Pro subjects (143 pairs in GPT split)
  - reasoning: LiveBench reasoning (98 pairs in GPT split)
  - math: LiveBench math + MMLU-Pro math (67 pairs in GPT split)
  - coding: LiveCodeBench (42 pairs in GPT split)

Judge types:
  - arena_hard: Arena-Hard style prompting (11 models)
  - vanilla: Direct prompting (GPT-4o)
  - chat_eval: ChatEval prompting (GPT-4o)
  - auto_j: GAIR AutoJ-13B
  - compass_judger: OpenCompass CompassJudger (4 sizes)
  - judge_lm: BAAI JudgeLM (3 sizes)
  - panda_lm: WeOpenML PandaLM-7B
  - prometheus_2: Prometheus-2 (3 variants)
  - reward_model: Reward models (5 models)
  - skywork_critic: Skywork Critic (2 sizes)
  - vertext_ai: Vertex AI GenAI Evaluation

Scoring:
  Each pair is judged twice (original order + swapped order) to test
  position-bias robustness.  A judge is correct (1.0) if its net vote
  matches the ground truth, incorrect (0.0) if it opposes, and NaN on ties.

Outputs:
  - response_matrix.csv: Binary results (judges x pairs) for all judges
  - response_matrix_by_category.csv: Per-category accuracy summary
  - item_metadata.csv: Per-item metadata (source, category)
  - judge_summary.csv: Per-judge aggregate statistics
"""

import json
import os
import re
from urllib.parse import quote as urlquote

import numpy as np
import pandas as pd
import requests

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SRC_HF_REPO = "ScalerLab/JudgeBench"
SRC_GH_REPO = "ScalerLab/JudgeBench"

# We use the GPT split (350 pairs, 33 judges) for the most judge coverage.
RESPONSE_MODEL = "gpt-4o-2024-05-13"

# Category mapping
_KNOWLEDGE_SOURCES = [
    "mmlu-pro-biology",
    "mmlu-pro-business",
    "mmlu-pro-chemistry",
    "mmlu-pro-computer science",
    "mmlu-pro-economics",
    "mmlu-pro-engineering",
    "mmlu-pro-health",
    "mmlu-pro-history",
    "mmlu-pro-law",
    "mmlu-pro-other",
    "mmlu-pro-philosophy",
    "mmlu-pro-physics",
    "mmlu-pro-psychology",
]

SOURCE_TO_CATEGORY: dict[str, str] = {}
for _s in _KNOWLEDGE_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "knowledge"
for _s in ["livebench-math", "mmlu-pro-math"]:
    SOURCE_TO_CATEGORY[_s] = "math"
SOURCE_TO_CATEGORY["livebench-reasoning"] = "reasoning"
SOURCE_TO_CATEGORY["livecodebench"] = "coding"

CATEGORIES = ["knowledge", "reasoning", "math", "coding"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gh_raw_url(path: str) -> str:
    """Build a raw GitHub URL, encoding path components."""
    encoded = urlquote(path, safe="/")
    return f"https://raw.githubusercontent.com/{SRC_GH_REPO}/main/{encoded}"


def parse_filename(filename: str) -> dict[str, str]:
    """Parse a JudgeBench output filename into its components."""
    name = filename.replace(".jsonl", "")
    parts = name.split(",")
    result = {}
    for part in parts:
        key, value = part.split("=", 1)
        result[key] = value
    return result


def flip_decision(decision: str) -> str:
    """Flip a judgment decision: A>B <-> B>A."""
    if decision == "A>B":
        return "B>A"
    elif decision == "B>A":
        return "A>B"
    return decision


def extract_decision(response_text: str) -> str | None:
    """Extract a decision (A>B or B>A) from a judge's text response."""
    if not response_text:
        return None

    text = response_text.strip().lower()

    if "a>b" in text or "a > b" in text:
        return "A>B"
    if "b>a" in text or "b > a" in text:
        return "B>A"
    if "output (a)" in text:
        return "A>B"
    if "output (b)" in text:
        return "B>A"
    if re.search(r"\bresponse\s*a\b", text):
        return "A>B"
    if re.search(r"\bresponse\s*b\b", text):
        return "B>A"

    return None


def score_judge_on_pair(pair: dict) -> float | None:
    """Score a judge's performance on a single pair using dual-judgment logic.

    Returns 1.0 (correct), 0.0 (incorrect), or None (inconclusive/null).
    """
    label = pair["label"]
    judgments = pair.get("judgments", [])

    if not judgments:
        return None

    if len(judgments) >= 2:
        decision_1 = extract_decision(
            judgments[0].get("judgment", {}).get("response", "")
        )
        decision_2_raw = extract_decision(
            judgments[1].get("judgment", {}).get("response", "")
        )
        decision_2 = flip_decision(decision_2_raw) if decision_2_raw else None

        counter = 0
        if decision_1 is not None:
            if decision_1 == label:
                counter += 1
            elif decision_1 == flip_decision(label):
                counter -= 1
        if decision_2 is not None:
            if decision_2 == label:
                counter += 1
            elif decision_2 == flip_decision(label):
                counter -= 1

        if counter > 0:
            return 1.0
        elif counter < 0:
            return 0.0
        else:
            return None
    else:
        decision = extract_decision(
            judgments[0].get("judgment", {}).get("response", "")
        )
        if decision is None:
            return None
        return 1.0 if decision == label else 0.0


# ---------------------------------------------------------------------------
# Data downloading
# ---------------------------------------------------------------------------


def download_output_files():
    """Download all judge output files from GitHub."""
    print("Listing output files from GitHub ...")
    api_url = f"https://api.github.com/repos/{SRC_GH_REPO}/contents/outputs"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    target_prefix = f"dataset=judgebench,response_model={RESPONSE_MODEL},"
    output_files = sorted(
        f["name"]
        for f in files
        if f["name"].startswith(target_prefix) and f["name"].endswith(".jsonl")
    )
    print(f"  Found {len(output_files)} output files for {RESPONSE_MODEL}")

    judge_results = {}
    for i, filename in enumerate(output_files):
        meta = parse_filename(filename)
        judge_key = f"{meta['judge_name']}/{meta['judge_model']}"
        try:
            url = _gh_raw_url(f"outputs/{filename}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            pairs = [json.loads(line) for line in lines]
            judge_results[judge_key] = pairs

            # Save locally
            local_path = os.path.join(RAW_DIR, filename)
            with open(local_path, "w") as f:
                f.write(r.text)

            print(f"  [{i + 1}/{len(output_files)}] {judge_key}: {len(pairs)} pairs")
        except Exception as e:
            print(f"  [{i + 1}/{len(output_files)}] Warning: failed {judge_key}: {e}")

    print(f"\n  Downloaded results for {len(judge_results)} judges")
    return judge_results


def download_item_metadata():
    """Download item metadata from the HuggingFace dataset."""
    print("\nDownloading item metadata from HuggingFace ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SRC_HF_REPO, split="gpt", token=HF_TOKEN)
        rows = []
        for item in ds:
            source = item["source"]
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "original_id": item["original_id"],
                    "source": source,
                    "category": SOURCE_TO_CATEGORY.get(source, "unknown"),
                    "label": item["label"],
                    "response_model": item["response_model"],
                }
            )
        meta_df = pd.DataFrame(rows)
        meta_path = os.path.join(RAW_DIR, "item_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  Saved {len(meta_df)} items to {meta_path}")
        return meta_df
    except Exception as e:
        print(f"  Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_response_matrix(judge_results):
    """Build binary response matrix from per-judge evaluation data."""
    # Determine item ordering from first judge
    first_judge_data = next(iter(judge_results.values()))
    item_ids = [p["pair_id"] for p in first_judge_data]
    item_id_to_idx = {pid: idx for idx, pid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Build source mapping
    id_to_source = {p["pair_id"]: p["source"] for p in first_judge_data}

    # Sort judges
    judge_keys = sorted(judge_results.keys())
    n_judges = len(judge_keys)

    print(f"\n{'=' * 60}")
    print(f"  Building response matrix: {n_judges} judges x {n_items} pairs")
    print(f"{'=' * 60}")

    # Build matrix
    matrix = np.full((n_judges, n_items), np.nan)
    for j, judge_key in enumerate(judge_keys):
        pairs = judge_results[judge_key]
        pair_map = {p["pair_id"]: p for p in pairs}
        for pid, idx in item_id_to_idx.items():
            if pid in pair_map:
                score = score_judge_on_pair(pair_map[pid])
                if score is not None:
                    matrix[j, idx] = score

    # Create DataFrame
    matrix_df = pd.DataFrame(matrix, index=judge_keys, columns=item_ids)
    matrix_df.index.name = "judge"

    # Statistics
    total_cells = n_judges * n_items
    n_valid = np.sum(~np.isnan(matrix))
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells

    print(f"  Judges:        {n_judges}")
    print(f"  Pairs:         {n_items}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid / total_cells * 100:.1f}%)")
    print(f"  Missing cells: {n_missing:,} ({n_missing / total_cells * 100:.1f}%)")
    print(f"  Fill rate:     {fill_rate * 100:.1f}%")

    # Accuracy stats
    valid_vals = matrix[~np.isnan(matrix)]
    mean_acc = np.mean(valid_vals)
    print(f"\n  Overall accuracy: {mean_acc:.4f}")

    # Per-judge stats
    per_judge_acc = np.nanmean(matrix, axis=1)
    best_idx = np.argmax(per_judge_acc)
    worst_idx = np.argmin(per_judge_acc)
    print(f"\n  Per-judge accuracy:")
    print(f"    Best:   {per_judge_acc[best_idx]:.4f} ({judge_keys[best_idx]})")
    print(f"    Worst:  {per_judge_acc[worst_idx]:.4f} ({judge_keys[worst_idx]})")
    print(f"    Median: {np.median(per_judge_acc):.4f}")
    print(f"    Std:    {np.std(per_judge_acc):.4f}")

    # Per-item stats
    per_item_acc = np.nanmean(matrix, axis=0)
    easy = np.sum(per_item_acc >= 0.8)
    medium = np.sum((per_item_acc >= 0.4) & (per_item_acc < 0.8))
    hard = np.sum(per_item_acc < 0.4)
    print(f"\n  Pair difficulty distribution:")
    print(f"    Easy (>=80%):   {easy}")
    print(f"    Medium (40-80%): {medium}")
    print(f"    Hard (<40%):    {hard}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    return matrix_df, item_ids, judge_keys, id_to_source


def build_category_summary(matrix_df, item_ids, id_to_source, judge_keys):
    """Build per-category accuracy summary."""
    id_to_category = {
        pid: SOURCE_TO_CATEGORY.get(src, "unknown")
        for pid, src in id_to_source.items()
    }

    rows = []
    for judge_key in judge_keys:
        row = {"judge": judge_key}
        judge_row = matrix_df.loc[judge_key]
        for cat in CATEGORIES:
            cat_items = [pid for pid in item_ids if id_to_category.get(pid) == cat]
            cat_vals = judge_row[cat_items]
            row[cat] = cat_vals.mean()
        row["overall"] = judge_row.mean()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("overall", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "response_matrix_by_category.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Category summary saved: {output_path}")

    print(f"\n  Top 10 judges (by overall accuracy):")
    for _, r in summary_df.head(10).iterrows():
        print(f"    {r['judge']:55s}  overall={r['overall']:.4f}")

    return summary_df


def build_judge_summary(matrix_df):
    """Build per-judge summary statistics."""
    rows = []
    for judge in matrix_df.index:
        judge_row = matrix_df.loc[judge]
        rows.append(
            {
                "judge": judge,
                "accuracy": judge_row.mean(),
                "n_items": judge_row.notna().sum(),
                "n_correct": (judge_row == 1.0).sum(),
            }
        )

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
    print("JudgeBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download judge output files
    print("STEP 1: Downloading judge evaluation results")
    print("-" * 60)
    judge_results = download_output_files()

    # Step 2: Download item metadata
    print("\nSTEP 2: Downloading item metadata")
    print("-" * 60)
    item_meta_df = download_item_metadata()

    # Step 3: Build response matrix
    print("\nSTEP 3: Building response matrix")
    print("-" * 60)
    matrix_df, item_ids, judge_keys, id_to_source = build_response_matrix(
        judge_results
    )

    # Step 4: Build category summary
    print("\nSTEP 4: Building category summary")
    print("-" * 60)
    category_df = build_category_summary(
        matrix_df, item_ids, id_to_source, judge_keys
    )

    # Step 5: Build judge summary
    print("\nSTEP 5: Building judge summary")
    print("-" * 60)
    judge_df = build_judge_summary(matrix_df)

    # Step 6: Save enriched item metadata
    print("\nSTEP 6: Saving enriched item metadata")
    print("-" * 60)
    id_to_source = {
        p["pair_id"]: p["source"]
        for p in next(iter(judge_results.values()))
    }
    per_item_acc = matrix_df.mean(axis=0)
    item_stats = pd.DataFrame(
        {
            "pair_id": item_ids,
            "source": [id_to_source.get(pid, "") for pid in item_ids],
            "category": [
                SOURCE_TO_CATEGORY.get(id_to_source.get(pid, ""), "unknown")
                for pid in item_ids
            ],
            "mean_accuracy": [per_item_acc.get(pid, np.nan) for pid in item_ids],
            "n_judges": [matrix_df[pid].notna().sum() for pid in item_ids],
        }
    )

    if item_meta_df is not None:
        merged = item_stats.merge(
            item_meta_df[["pair_id", "original_id", "label"]],
            on="pair_id",
            how="left",
        )
    else:
        merged = item_stats

    merged_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    merged.to_csv(merged_path, index=False)
    print(f"  Item metadata saved: {merged_path}")

    # Category breakdown
    print(f"\n  Per-category accuracy (across all judges):")
    for cat in CATEGORIES:
        cat_items = merged[merged["category"] == cat]
        mean_acc = cat_items["mean_accuracy"].mean()
        print(f"    {cat:15s}  n={len(cat_items):4d}  mean_acc={mean_acc:.4f}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Response matrix: {len(judge_keys)} judges x {len(item_ids)} pairs")
    print(f"  Categories: {', '.join(CATEGORIES)}")
    print(f"  Category summary: {len(category_df)} judges with category-level scores")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
