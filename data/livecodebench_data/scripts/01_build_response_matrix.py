"""
Build a per-problem response matrix from LiveCodeBench submissions.

Data sources:
1. Primary: submissions repo eval_all.json files (72 models, variable problem counts)
2. Supplementary: leaderboard performances_generation.json (fills gaps)

Problem sets are nested: 713 ⊂ 880 ⊂ 1055 (release versions v3, v5, v6).
All 1055 problems form the column space. Models with fewer problems get NaN
for the extra problems.

pass@1 values are continuous [0,1]. N=1 models are binary (0 or 1).
N=10 models have 11 discrete values (0, 0.1, ..., 1.0).

Output: processed/response_matrix.csv (models × problems, pass@1 scores)
"""

import json
import os

import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
SUBMISSIONS_DIR = f"{RAW_DIR}/submissions"
LEADERBOARD_DIR = f"{RAW_DIR}/livecodebench.github.io/build"
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to skip (duplicates or problematic)
SKIP_MODELS = {
    "DeepSeek-V3 copy",           # Duplicate of DeepSeek-V3 (880 vs 1055)
    "O1-2024-12-17 (Low)_prompt_old",  # Old prompt variant duplicate
}


def load_submissions():
    """Load per-problem pass@1 from submissions repo eval_all.json files."""
    models = {}
    dirs = sorted([
        d for d in os.listdir(SUBMISSIONS_DIR)
        if os.path.isdir(os.path.join(SUBMISSIONS_DIR, d)) and d != ".git"
    ])

    for d in dirs:
        if d in SKIP_MODELS:
            print(f"  Skipping {d} (duplicate)")
            continue

        model_dir = os.path.join(SUBMISSIONS_DIR, d)
        json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
        if not json_files:
            print(f"  Skipping {d} (no data files)")
            continue

        # If multiple files, prefer the one with more samples (N=10 > N=1)
        # or more problems
        best_file = None
        best_n_problems = 0
        for f in json_files:
            fpath = os.path.join(model_dir, f)
            with open(fpath) as fp:
                data = json.load(fp)
            if len(data) > best_n_problems:
                best_n_problems = len(data)
                best_file = (f, data)

        fname, data = best_file
        # Extract pass@1 per problem
        results = {}
        for entry in data:
            qid = entry["question_id"]
            p1_key = "pass@1" if "pass@1" in entry else "pass1"
            p1 = entry[p1_key]
            results[qid] = p1

        # Extract N from filename (e.g., codegeneration_10_0.2)
        n_samples = "?"
        parts = fname.replace("Scenario.codegeneration_", "").split("_")
        if parts:
            n_samples = parts[0]

        models[d] = {
            "results": results,
            "n_problems": len(results),
            "n_samples": n_samples,
            "source": "submissions",
        }

    return models


def load_leaderboard_json(path, label):
    """Load per-problem pass@1 from a leaderboard JSON file."""
    with open(path) as f:
        data = json.load(f)

    models = {}
    for perf in data["performances"]:
        model = perf["model"]
        qid = perf["question_id"]
        p1 = perf["pass@1"]
        # Leaderboard stores pass@1 as percentage (0-100), convert to 0-1
        if p1 > 1.0:
            p1 = p1 / 100.0

        if model not in models:
            models[model] = {"results": {}, "source": label}
        models[model]["results"][qid] = p1

    for m in models:
        models[m]["n_problems"] = len(models[m]["results"])

    return models


def main():
    print("=== Loading submissions repo data ===")
    sub_models = load_submissions()
    print(f"Loaded {len(sub_models)} models from submissions repo")

    print("\n=== Loading leaderboard data (v6) ===")
    lb_v6 = load_leaderboard_json(
        f"{LEADERBOARD_DIR}/performances_generation.json", "leaderboard_v6"
    )
    print(f"Loaded {len(lb_v6)} models from leaderboard v6")

    print("\n=== Loading leaderboard data (v5) ===")
    lb_v5 = load_leaderboard_json(
        f"{LEADERBOARD_DIR}/v5.json", "leaderboard_v5"
    )
    print(f"Loaded {len(lb_v5)} models from leaderboard v5")

    # Merge: submissions repo is primary, fill gaps from leaderboard
    all_models = dict(sub_models)  # start with submissions
    filled_from_lb = []
    for lb, lb_label in [(lb_v6, "v6"), (lb_v5, "v5")]:
        for model, data in lb.items():
            if model not in all_models:
                all_models[model] = data
                filled_from_lb.append((model, lb_label, data["n_problems"]))
            elif data["n_problems"] > all_models[model]["n_problems"]:
                # Leaderboard has more problems for this model
                old_n = all_models[model]["n_problems"]
                all_models[model] = data
                filled_from_lb.append(
                    (model, f"{lb_label} (upgraded from {old_n})",
                     data["n_problems"])
                )

    if filled_from_lb:
        print(f"\nFilled {len(filled_from_lb)} models from leaderboard:")
        for model, src, n in filled_from_lb:
            print(f"  {model}: {src} ({n} problems)")

    # Collect all unique question IDs
    all_qids = set()
    for data in all_models.values():
        all_qids.update(data["results"].keys())
    all_qids = sorted(all_qids, key=lambda x: str(x))
    print(f"\nTotal unique problems: {len(all_qids)}")

    # Build the response matrix
    model_names = sorted(all_models.keys())
    print(f"Total models: {len(model_names)}")

    matrix = np.full((len(model_names), len(all_qids)), np.nan)
    qid_to_col = {qid: i for i, qid in enumerate(all_qids)}

    for row_idx, model in enumerate(model_names):
        results = all_models[model]["results"]
        for qid, p1 in results.items():
            col = qid_to_col[qid]
            matrix[row_idx, col] = p1

    # Convert question IDs to strings for CSV
    col_labels = [str(qid) for qid in all_qids]

    df = pd.DataFrame(matrix, index=model_names, columns=col_labels)
    df.index.name = "model"

    # Save full matrix
    csv_path = os.path.join(OUTPUT_DIR, "response_matrix.csv")
    df.to_csv(csv_path)
    print(f"\nSaved {csv_path}")
    print(f"  Shape: {df.shape}")

    # Report coverage
    n_filled = np.count_nonzero(~np.isnan(matrix))
    n_total = matrix.size
    print(f"  Filled cells: {n_filled}/{n_total} ({100*n_filled/n_total:.1f}%)")
    print(f"  Mean pass@1: {np.nanmean(matrix):.3f}")

    # Per-model summary
    print("\n=== Per-Model Summary ===")
    model_stats = []
    for i, model in enumerate(model_names):
        row = matrix[i]
        n_filled_row = np.count_nonzero(~np.isnan(row))
        mean_p1 = np.nanmean(row)
        n_samples = all_models[model].get("n_samples", "?")
        source = all_models[model]["source"]
        model_stats.append({
            "model": model,
            "n_problems": n_filled_row,
            "mean_pass1": mean_p1,
            "n_samples": n_samples,
            "source": source,
        })

    model_stats_df = pd.DataFrame(model_stats).sort_values(
        "mean_pass1", ascending=False)
    model_stats_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)

    print("  Top 10:")
    for _, row in model_stats_df.head(10).iterrows():
        print(f"    {row['model']:50s}  pass@1={row['mean_pass1']:.3f}  "
              f"n={row['n_problems']}  N={row['n_samples']}  "
              f"src={row['source']}")
    print("  Bottom 10:")
    for _, row in model_stats_df.tail(10).iterrows():
        print(f"    {row['model']:50s}  pass@1={row['mean_pass1']:.3f}  "
              f"n={row['n_problems']}  N={row['n_samples']}  "
              f"src={row['source']}")

    # Problem difficulty summary
    task_difficulty = np.nanmean(matrix, axis=0)
    print(f"\n=== Problem Difficulty Distribution ===")
    print(f"  Always pass (pass@1=1.0): "
          f"{np.sum(task_difficulty == 1.0)} / {len(all_qids)}")
    print(f"  Always fail (pass@1=0.0): "
          f"{np.sum(task_difficulty == 0.0)} / {len(all_qids)}")
    print(f"  Mixed (0<pass@1<1): "
          f"{np.sum((task_difficulty > 0) & (task_difficulty < 1))} "
          f"/ {len(all_qids)}")
    print(f"  Mean difficulty: {np.nanmean(task_difficulty):.3f}")
    print(f"  Median difficulty: {np.nanmedian(task_difficulty):.3f}")

    # Problem metadata
    print("\n=== Problem Metadata ===")
    # Load from any model that has 1055 problems
    ref_model = None
    for model in model_names:
        if all_models[model]["n_problems"] == 1055:
            # Prefer submissions repo
            if all_models[model]["source"] == "submissions":
                ref_model = model
                break
    if ref_model is None:
        for model in model_names:
            if all_models[model]["n_problems"] == 1055:
                ref_model = model
                break

    if ref_model:
        ref_dir = os.path.join(SUBMISSIONS_DIR, ref_model)
        json_files = [f for f in os.listdir(ref_dir) if f.endswith(".json")]
        if json_files:
            with open(os.path.join(ref_dir, json_files[0])) as f:
                ref_data = json.load(f)

            problems = []
            for entry in ref_data:
                problems.append({
                    "question_id": entry["question_id"],
                    "question_title": entry.get("question_title", ""),
                    "platform": entry.get("platform", ""),
                    "difficulty": entry.get("difficulty", ""),
                    "contest_id": entry.get("contest_id", ""),
                    "contest_date": entry.get("contest_date", ""),
                })

            prob_df = pd.DataFrame(problems)
            prob_df.to_csv(
                os.path.join(OUTPUT_DIR, "problem_metadata.csv"), index=False)
            print(f"  Saved problem_metadata.csv ({len(prob_df)} problems)")

            # Difficulty distribution
            print(f"\n  Difficulty distribution:")
            for diff, count in prob_df["difficulty"].value_counts().items():
                print(f"    {diff}: {count}")

            # Platform distribution
            print(f"\n  Platform distribution:")
            for plat, count in prob_df["platform"].value_counts().items():
                print(f"    {plat}: {count}")

    # Coverage by problem set size
    print("\n=== Coverage by Problem Set ===")
    for n_thresh in [713, 880, 1055]:
        n_models_with = sum(
            1 for m in model_names
            if all_models[m]["n_problems"] >= n_thresh
        )
        print(f"  Models with >= {n_thresh} problems: {n_models_with}")


if __name__ == "__main__":
    main()
