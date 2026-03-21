"""
Build FLASK response matrices from GPT-4 evaluation review files.

Data source:
  - kaistAI/FLASK on GitHub
  - evaluation_set/flask_evaluation.jsonl: 1,740 instructions with skill/domain metadata
  - gpt_review/outputs/<model>.jsonl: GPT-4 evaluation scores for 15 models on
    1,700 instructions across 12 fine-grained skills

Structure:
  - 15 models: alpaca-13b, bard, chatgpt, claude-v1, davinci-003, gpt-4,
    llama-2-chat-13b, llama-2-chat-70b, tulu-7b, tulu-13b, tulu-30b, tulu-65b,
    vicuna-13b, vicuna-33b, wizardlm-13b
  - 12 skills organized in 4 categories:
    * Logical Thinking: logical_correctness, logical_robustness, logical_efficiency
    * Background Knowledge: factuality, commonsense_understanding
    * Problem Handling: comprehension, insightfulness, completeness, metacognition
    * User Alignment: conciseness, readability, harmlessness
  - Each instruction is evaluated on 2-3 relevant skills (1-5 scale)
  - Scores can be "N/A" (treated as missing)

Outputs:
  - response_matrix_overall.csv: Mean score across applicable skills (models x items)
  - response_matrix_<skill>.csv: Per-skill score matrices (models x items with that skill)
  - item_metadata.csv: Per-item metadata (domain, difficulty, assigned skills)
  - model_summary.csv: Per-model aggregate statistics
"""

import json
import os
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SRC_BASE_URL = "https://raw.githubusercontent.com/kaistAI/FLASK/main"

# Model review files -> display names.
MODEL_FILES = {
    "alpaca_13b.jsonl": "alpaca-13b",
    "bard_review.jsonl": "bard",
    "chatgpt_review.jsonl": "chatgpt",
    "claude_v1_review.jsonl": "claude-v1",
    "davinci_003_review.jsonl": "davinci-003",
    "gpt4_review.jsonl": "gpt-4",
    "llama2_chat_13b.jsonl": "llama-2-chat-13b",
    "llama2_chat_70b.jsonl": "llama-2-chat-70b",
    "tulu_7b_review.jsonl": "tulu-7b",
    "tulu_13b_review.jsonl": "tulu-13b",
    "tulu_30b_review.jsonl": "tulu-30b",
    "tulu_65b_review.jsonl": "tulu-65b",
    "vicuna_13b.jsonl": "vicuna-13b",
    "vicuna_33b.jsonl": "vicuna-33b",
    "wizardlm_13b.jsonl": "wizardlm-13b",
}

# Normalize variant score key names to canonical skill names.
SKILL_NORMALIZE = {
    "commonsense understanding": "commonsense_understanding",
    "commonsense": "commonsense_understanding",
    "commonsense reasoning": "commonsense_understanding",
    "completeness": "completeness",
    "comprehension": "comprehension",
    "comprehension score": "comprehension",
    "conciseness": "conciseness",
    "conciseness score": "conciseness",
    "factuality": "factuality",
    "harmlessness": "harmlessness",
    "insightfulness": "insightfulness",
    "logical correctness": "logical_correctness",
    "logical efficiency": "logical_efficiency",
    "logical robustness": "logical_robustness",
    "metacognition": "metacognition",
    "readability": "readability",
    "readability score": "readability",
}

# Canonical skill ordering.
SKILLS = [
    "logical_correctness",
    "logical_robustness",
    "logical_efficiency",
    "factuality",
    "commonsense_understanding",
    "comprehension",
    "insightfulness",
    "completeness",
    "metacognition",
    "conciseness",
    "readability",
    "harmlessness",
]


def download_jsonl(url: str, local_path: str) -> list[dict]:
    """Download a JSONL file and save locally."""
    if os.path.exists(local_path):
        print(f"    Using cached: {local_path}")
        with open(local_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    print(f"    Downloading: {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as f:
        f.write(text)

    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def download_all_data():
    """Download evaluation set and all model review files."""
    # Evaluation set
    print("\nDownloading evaluation set ...")
    eval_url = f"{SRC_BASE_URL}/evaluation_set/flask_evaluation.jsonl"
    eval_path = os.path.join(RAW_DIR, "flask_evaluation.jsonl")
    eval_set = download_jsonl(eval_url, eval_path)
    print(f"  {len(eval_set)} instructions")

    # Model reviews
    print("\nDownloading model review files ...")
    reviews = {}
    for filename, display_name in sorted(MODEL_FILES.items()):
        url = f"{SRC_BASE_URL}/gpt_review/outputs/{filename}"
        local_path = os.path.join(RAW_DIR, "gpt_review", filename)
        entries = download_jsonl(url, local_path)
        reviews[display_name] = entries
        print(f"  {display_name}: {len(entries)} entries")

    return eval_set, reviews


def parse_scores(entry: dict) -> dict[str, float]:
    """Extract normalized skill -> score mapping from a review entry."""
    scores = {}
    raw_scores = entry.get("score", {})
    for key, val in raw_scores.items():
        norm_skill = SKILL_NORMALIZE.get(key.lower())
        if norm_skill is None:
            continue
        if isinstance(val, (int, float)):
            scores[norm_skill] = float(val)
    return scores


def build_response_matrices(eval_set, reviews):
    """Build overall and per-skill response matrices."""
    # Item ordering from first model
    first_model = next(iter(reviews.values()))
    item_ids = [str(entry["question_id"]) for entry in first_model]
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Sort models
    model_names = sorted(reviews.keys())
    n_models = len(model_names)

    print(f"\n{'='*60}")
    print(f"  Building response matrices: {n_models} models x {n_items} items")
    print(f"{'='*60}")

    # Build per-skill matrices
    skill_matrices = {}
    for skill in SKILLS:
        skill_matrices[skill] = np.full((n_models, n_items), np.nan)

    overall_sum = np.zeros((n_models, n_items))
    overall_count = np.zeros((n_models, n_items))

    for m, model_name in enumerate(model_names):
        entries = reviews[model_name]
        for entry in entries:
            qid = str(entry["question_id"])
            if qid not in item_id_to_idx:
                continue
            i = item_id_to_idx[qid]
            scores = parse_scores(entry)
            for skill, score in scores.items():
                if skill in skill_matrices:
                    skill_matrices[skill][m, i] = score
                    overall_sum[m, i] += score
                    overall_count[m, i] += 1.0

    # Overall mean
    with np.errstate(divide="ignore", invalid="ignore"):
        overall_data = np.where(overall_count > 0, overall_sum / overall_count, np.nan)

    # Save overall matrix
    overall_df = pd.DataFrame(overall_data, index=model_names, columns=item_ids)
    overall_df.index.name = "model"
    overall_path = os.path.join(PROCESSED_DIR, "response_matrix_overall.csv")
    overall_df.to_csv(overall_path)
    print(f"\n  Overall matrix saved: {overall_path}")

    total_cells = n_models * n_items
    n_valid = np.sum(~np.isnan(overall_data))
    print(f"  Overall: {n_models} x {n_items}, {n_valid}/{total_cells} valid "
          f"({n_valid/total_cells*100:.1f}%)")
    print(f"  Overall mean score: {np.nanmean(overall_data):.3f}")

    # Per-model stats
    per_model_mean = np.nanmean(overall_data, axis=1)
    best_idx = np.argmax(per_model_mean)
    worst_idx = np.argmin(per_model_mean)
    print(f"\n  Per-model overall mean:")
    print(f"    Best:   {per_model_mean[best_idx]:.3f} ({model_names[best_idx]})")
    print(f"    Worst:  {per_model_mean[worst_idx]:.3f} ({model_names[worst_idx]})")
    print(f"    Median: {np.median(per_model_mean):.3f}")

    # Save per-skill matrices
    print(f"\n  Per-skill matrices:")
    for skill in SKILLS:
        mat = skill_matrices[skill]

        # Filter to items with at least one non-NaN score
        has_any = ~np.all(np.isnan(mat), axis=0)
        valid_cols = np.where(has_any)[0]
        filtered_mat = mat[:, valid_cols]
        filtered_ids = [item_ids[c] for c in valid_cols]

        skill_df = pd.DataFrame(filtered_mat, index=model_names, columns=filtered_ids)
        skill_df.index.name = "model"
        skill_path = os.path.join(PROCESSED_DIR, f"response_matrix_{skill}.csv")
        skill_df.to_csv(skill_path)

        n_valid_skill = np.sum(~np.isnan(filtered_mat))
        total_skill = filtered_mat.size
        mean_score = np.nanmean(filtered_mat)
        print(f"    {skill:30s}  {n_models} x {len(valid_cols):4d} items, "
              f"mean={mean_score:.3f}, "
              f"{n_valid_skill/total_skill*100:.1f}% valid")

    return overall_df, skill_matrices, item_ids, model_names


def build_item_metadata(eval_set, reviews, item_ids):
    """Build per-item metadata from evaluation set and review data."""
    # Build eval set lookup
    eval_meta = {}
    for entry in eval_set:
        eval_meta[str(entry["idx"])] = {
            "domain": entry.get("domain", ""),
            "difficulty": entry.get("difficulty", ""),
            "skills": ", ".join(entry.get("skill", [])),
            "task": entry.get("task", ""),
        }

    # Build item metadata
    rows = []
    for iid in item_ids:
        meta = eval_meta.get(iid, {})
        rows.append({
            "item_id": iid,
            "domain": meta.get("domain", ""),
            "difficulty": meta.get("difficulty", ""),
            "skills": meta.get("skills", ""),
            "task": meta.get("task", ""),
        })

    meta_df = pd.DataFrame(rows)
    meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"\n  Item metadata saved: {meta_path}")
    print(f"  {len(meta_df)} items")

    # Domain breakdown
    print(f"\n  Domain distribution:")
    for domain, count in meta_df["domain"].value_counts().items():
        print(f"    {domain}: {count}")

    # Difficulty breakdown
    print(f"\n  Difficulty distribution:")
    for diff, count in meta_df["difficulty"].value_counts().items():
        print(f"    {diff}: {count}")

    return meta_df


def build_model_summary(overall_df, model_names):
    """Build per-model summary statistics."""
    rows = []
    for model in model_names:
        model_row = overall_df.loc[model]
        rows.append({
            "model": model,
            "mean_score": model_row.mean(),
            "median_score": model_row.median(),
            "std_score": model_row.std(),
            "n_items": model_row.notna().sum(),
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("mean_score", ascending=False)
    summary_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Model summary saved: {summary_path}")

    print(f"\n  Model rankings (by mean score):")
    for _, r in summary_df.iterrows():
        print(f"    {r['model']:25s}  mean={r['mean_score']:.3f}  "
              f"median={r['median_score']:.3f}  n={int(r['n_items'])}")

    return summary_df


def main():
    print("FLASK Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading data from kaistAI/FLASK")
    print("-" * 60)
    eval_set, reviews = download_all_data()

    # Step 2: Build response matrices
    print("\nSTEP 2: Building response matrices")
    print("-" * 60)
    overall_df, skill_matrices, item_ids, model_names = build_response_matrices(
        eval_set, reviews
    )

    # Step 3: Build item metadata
    print("\nSTEP 3: Building item metadata")
    print("-" * 60)
    item_meta_df = build_item_metadata(eval_set, reviews, item_ids)

    # Step 4: Build model summary
    print("\nSTEP 4: Building model summary")
    print("-" * 60)
    model_summary_df = build_model_summary(overall_df, model_names)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Models: {len(model_names)}")
    print(f"  Items (instructions): {len(item_ids)}")
    print(f"  Skills: {len(SKILLS)}")
    print(f"  Overall matrix: {len(model_names)} x {len(item_ids)}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:50s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
