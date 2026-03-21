"""
Build PRM800K response matrices from step-level human correctness labels.

Data source:
  - https://github.com/openai/prm800k
  - ~800K step-level human labels (+1 correct, -1 incorrect, 0 neutral)
    for model-generated solutions to MATH problems.

Structure:
  - Phase 1: Labelers annotated steps and could write alternatives (~1K solutions).
  - Phase 2: Active-learning-selected solutions with step labels (~100K solutions).
  - Train split: phase1_train + phase2_train (98,731 solutions).
  - Test split: phase1_test + phase2_test (2,868 solutions).

Label mapping:
  - +1 (correct)    -> 1.0
  - -1 (incorrect)  -> 0.0
  - 0  (neutral)    -> NaN
  - None (unlabeled) -> NaN
  - human_completion (phase 1) -> 1.0 (labeler wrote a correct alternative)

Response matrix format:
  - Rows: individual solution attempts
  - Columns: step positions (step_0, step_1, ..., step_N)
  - Values: binary {0, 1} or NaN (neutral/unlabeled/padding)

Outputs:
  - response_matrix_train.csv: Step-level labels (solutions x steps) for train split
  - response_matrix_test.csv: Step-level labels (solutions x steps) for test split
  - response_matrix_all.csv: Combined train + test
  - solution_metadata.csv: Per-solution metadata (problem, finish_reason, etc.)
  - problem_metadata.csv: Per-problem metadata (subject, level, answer)
  - summary_statistics.txt: Dataset statistics
"""

import json
import os
import subprocess
from collections import Counter



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

SRC_REPO_URL = "https://github.com/openai/prm800k.git"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def download_data():
    """Clone the PRM800K GitHub repository with LFS data."""
    repo_dir = os.path.join(RAW_DIR, "prm800k")
    if os.path.exists(os.path.join(repo_dir, "prm800k", "data")):
        print(f"  Repository already cloned at {repo_dir}")
        return repo_dir

    print(f"  Cloning {SRC_REPO_URL} into {repo_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SRC_REPO_URL, repo_dir],
        check=True,
    )
    # Pull LFS files (the JSONL data files).
    subprocess.run(
        ["git", "-C", repo_dir, "lfs", "pull"],
        check=True,
    )
    return repo_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_math_metadata(repo_dir):
    """Load MATH problem metadata (subject, level, answer) keyed by problem text."""
    meta = {}
    for split in ["train.jsonl", "test.jsonl"]:
        path = os.path.join(repo_dir, "prm800k", "math_splits", split)
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                meta[data["problem"]] = {
                    "subject": data.get("subject", ""),
                    "level": data.get("level", 0),
                    "unique_id": data.get("unique_id", ""),
                    "answer": data.get("answer", ""),
                }
    return meta


def load_solutions(repo_dir, file_path):
    """Load solution records from a JSONL data file.

    Returns a list of dicts with keys:
        problem, ratings (list of float|None), finish_reason, n_steps,
        ground_truth_answer, pre_generated_answer, labeler, generation.
    """
    solutions = []
    full_path = os.path.join(repo_dir, file_path)
    if not os.path.exists(full_path):
        print(f"  Warning: {full_path} not found, skipping.")
        return solutions

    with open(full_path) as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            label = data["label"]
            steps = label["steps"]

            # Extract the chosen rating for each step.
            ratings = []
            for step in steps:
                chosen_idx = step.get("chosen_completion")
                if step.get("human_completion") is not None:
                    # Phase 1: human-written step -> treat as correct.
                    ratings.append(1.0)
                elif (
                    chosen_idx is not None
                    and 0 <= chosen_idx < len(step["completions"])
                ):
                    raw = step["completions"][chosen_idx]["rating"]
                    if raw == 1:
                        ratings.append(1.0)
                    elif raw == -1:
                        ratings.append(0.0)
                    else:
                        # 0 (neutral) or None -> NaN
                        ratings.append(None)
                else:
                    ratings.append(None)

            solutions.append({
                "problem": question["problem"],
                "ratings": ratings,
                "finish_reason": label["finish_reason"],
                "n_steps": len(steps),
                "ground_truth_answer": question.get("ground_truth_answer", ""),
                "pre_generated_answer": question.get("pre_generated_answer", ""),
                "labeler": data.get("labeler", ""),
                "generation": data.get("generation"),
            })
    return solutions


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_response_matrix(solutions, split_label):
    """Build a step-level response matrix from solution records.

    Returns (matrix_df, solution_meta_df).
    """
    n_solutions = len(solutions)
    max_steps = max(len(s["ratings"]) for s in solutions) if solutions else 0

    print(f"\n{'='*60}")
    print(f"  Building response matrix for {split_label}: "
          f"{n_solutions} solutions x {max_steps} steps")
    print(f"{'='*60}")

    # Build matrix (solutions x steps).
    matrix = np.full((n_solutions, max_steps), np.nan)
    sol_ids = []
    sol_meta_rows = []

    for i, sol in enumerate(solutions):
        sol_id = f"{split_label}_{i:06d}"
        sol_ids.append(sol_id)

        for j, rating in enumerate(sol["ratings"]):
            if rating is not None:
                matrix[i, j] = rating

        sol_meta_rows.append({
            "solution_id": sol_id,
            "problem": sol["problem"][:500],
            "finish_reason": sol["finish_reason"],
            "n_steps": sol["n_steps"],
            "ground_truth_answer": sol["ground_truth_answer"],
            "pre_generated_answer": sol["pre_generated_answer"],
            "labeler": sol["labeler"],
            "generation": sol["generation"],
        })

    step_cols = [f"step_{j}" for j in range(max_steps)]
    matrix_df = pd.DataFrame(matrix, index=sol_ids, columns=step_cols)
    matrix_df.index.name = "solution"

    sol_meta_df = pd.DataFrame(sol_meta_rows)

    # Statistics
    total_cells = n_solutions * max_steps
    n_valid = int(np.sum(~np.isnan(matrix)))
    n_pos = int(np.sum(matrix == 1.0))
    n_neg = int(np.sum(matrix == 0.0))
    n_nan = total_cells - n_valid

    print(f"  Solutions:     {n_solutions:,}")
    print(f"  Max steps:     {max_steps}")
    print(f"  Total cells:   {total_cells:,}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Positive:    {n_pos:,} ({n_pos/max(n_valid,1)*100:.1f}%)")
    print(f"    Negative:    {n_neg:,} ({n_neg/max(n_valid,1)*100:.1f}%)")
    print(f"  NaN cells:     {n_nan:,} ({n_nan/total_cells*100:.1f}%)")

    # Finish reason distribution
    finish_counts = Counter(s["finish_reason"] for s in solutions)
    print(f"\n  Finish reasons:")
    for reason, count in sorted(finish_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:20s}  {count:,}")

    # Steps per solution distribution
    step_counts = [s["n_steps"] for s in solutions]
    print(f"\n  Steps per solution:")
    print(f"    Min:    {min(step_counts)}")
    print(f"    Max:    {max(step_counts)}")
    print(f"    Mean:   {np.mean(step_counts):.1f}")
    print(f"    Median: {np.median(step_counts):.0f}")

    return matrix_df, sol_meta_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("PRM800K Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data.
    print("STEP 1: Downloading PRM800K repository")
    print("-" * 60)
    repo_dir = download_data()

    # Step 2: Load MATH metadata.
    print("\nSTEP 2: Loading MATH problem metadata")
    print("-" * 60)
    math_meta = load_math_metadata(repo_dir)
    print(f"  Loaded metadata for {len(math_meta)} MATH problems")

    # Step 3: Load all solution files.
    print("\nSTEP 3: Loading solution files")
    print("-" * 60)

    p1_train = load_solutions(repo_dir, "prm800k/data/phase1_train.jsonl")
    print(f"  Phase 1 train: {len(p1_train)} solutions")

    p1_test = load_solutions(repo_dir, "prm800k/data/phase1_test.jsonl")
    print(f"  Phase 1 test:  {len(p1_test)} solutions")

    p2_train = load_solutions(repo_dir, "prm800k/data/phase2_train.jsonl")
    print(f"  Phase 2 train: {len(p2_train)} solutions")

    p2_test = load_solutions(repo_dir, "prm800k/data/phase2_test.jsonl")
    print(f"  Phase 2 test:  {len(p2_test)} solutions")

    total = len(p1_train) + len(p1_test) + len(p2_train) + len(p2_test)
    print(f"  Total: {total} solutions")

    # Step 4: Build response matrices.
    print("\nSTEP 4: Building response matrices")
    print("-" * 60)

    # Train split.
    train_solutions = p1_train + p2_train
    train_matrix, train_meta = build_response_matrix(train_solutions, "train")
    train_path = os.path.join(PROCESSED_DIR, "response_matrix_train.csv")
    train_matrix.to_csv(train_path)
    print(f"\n  Saved: {train_path}")

    # Test split.
    test_solutions = p1_test + p2_test
    test_matrix, test_meta = build_response_matrix(test_solutions, "test")
    test_path = os.path.join(PROCESSED_DIR, "response_matrix_test.csv")
    test_matrix.to_csv(test_path)
    print(f"\n  Saved: {test_path}")

    # Combined (all).
    all_solutions = p1_train + p1_test + p2_train + p2_test
    all_matrix, all_meta = build_response_matrix(all_solutions, "all")
    all_path = os.path.join(PROCESSED_DIR, "response_matrix_all.csv")
    all_matrix.to_csv(all_path)
    print(f"\n  Saved: {all_path}")

    # Step 5: Save solution metadata.
    print("\nSTEP 5: Saving solution metadata")
    print("-" * 60)

    all_meta_path = os.path.join(PROCESSED_DIR, "solution_metadata.csv")
    all_meta.to_csv(all_meta_path, index=False)
    print(f"  Solution metadata saved: {all_meta_path}")

    # Step 6: Save problem metadata.
    print("\nSTEP 6: Saving problem metadata")
    print("-" * 60)

    # Collect unique problems from all solutions.
    seen_problems = set()
    problem_rows = []
    for sol in all_solutions:
        prob = sol["problem"]
        if prob not in seen_problems:
            seen_problems.add(prob)
            meta = math_meta.get(prob, {})
            problem_rows.append({
                "problem": prob[:500],
                "subject": meta.get("subject", ""),
                "level": meta.get("level", 0),
                "unique_id": meta.get("unique_id", ""),
                "answer": meta.get("answer", ""),
            })

    problem_df = pd.DataFrame(problem_rows)
    problem_path = os.path.join(PROCESSED_DIR, "problem_metadata.csv")
    problem_df.to_csv(problem_path, index=False)
    print(f"  Problem metadata saved: {problem_path}")
    print(f"  Unique problems: {len(problem_df)}")

    # Subject distribution.
    if "subject" in problem_df.columns:
        print(f"\n  MATH subject distribution:")
        for subject, count in problem_df["subject"].value_counts().items():
            print(f"    {subject:25s}  {count:,}")

    # Step 7: Summary statistics.
    print("\nSTEP 7: Writing summary statistics")
    print("-" * 60)

    summary_path = os.path.join(PROCESSED_DIR, "summary_statistics.txt")
    with open(summary_path, "w") as f:
        f.write("PRM800K Response Matrix Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {SRC_REPO_URL}\n\n")

        for name, mat in [
            ("Train", train_matrix),
            ("Test", test_matrix),
            ("All", all_matrix),
        ]:
            n_sol, n_steps = mat.shape
            vals = mat.values
            n_valid = int(np.sum(~np.isnan(vals)))
            n_pos = int(np.sum(vals == 1.0))
            n_neg = int(np.sum(vals == 0.0))
            f.write(f"{name} split:\n")
            f.write(f"  Solutions: {n_sol:,}\n")
            f.write(f"  Max steps: {n_steps}\n")
            f.write(f"  Valid cells: {n_valid:,}\n")
            f.write(f"  Positive: {n_pos:,}\n")
            f.write(f"  Negative: {n_neg:,}\n\n")

        f.write(f"Unique problems: {len(problem_df)}\n")
        f.write(f"Total solutions: {len(all_solutions)}\n")

    print(f"  Summary saved: {summary_path}")

    # Final summary.
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {train_matrix.shape[0]:,} solutions x {train_matrix.shape[1]} steps")
    print(f"  Test:  {test_matrix.shape[0]:,} solutions x {test_matrix.shape[1]} steps")
    print(f"  All:   {all_matrix.shape[0]:,} solutions x {all_matrix.shape[1]} steps")
    print(f"  Unique problems: {len(problem_df)}")

    print(f"\n  All output files:")
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
