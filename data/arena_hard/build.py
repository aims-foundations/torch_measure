"""
Build Arena-Hard-Auto response matrix from GPT-4 judgment files.

Data source:
  - lmarena-ai/arena-hard-auto HuggingFace Space (Git LFS repo)
  - Judgment JSONL files in data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview/
  - 72 models x 500 prompts, 5-level judgments vs GPT-4-0314 baseline

Score mapping (judgment -> numeric):
  - "A>>B" -> 1.0  (model much better than baseline)
  - "A>B"  -> 0.75 (model better)
  - "tie"  -> 0.5  (tie)
  - "B>A"  -> 0.25 (baseline better)
  - "B>>A" -> 0.0  (baseline much better)

Outputs:
  - processed/response_matrix.csv: Models (rows) x prompts (columns), scores 0-1
"""

import os
import json
import subprocess
import sys

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Clone target
REPO_DIR = os.path.join(RAW_DIR, "arena-hard-auto")

# Judgment score mapping
JUDGMENT_MAP = {
    "A>>B": 1.0,
    "A>B": 0.75,
    "tie": 0.5,
    "B>A": 0.25,
    "B>>A": 0.0,
}


def clone_repo():
    """Clone the arena-hard-auto HuggingFace Space repo."""
    if os.path.exists(REPO_DIR) and os.path.isdir(
        os.path.join(REPO_DIR, "data")
    ):
        print(f"  Repo already cloned: {REPO_DIR}")
        return

    print("  Cloning lmarena-ai/arena-hard-auto from HuggingFace...")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://huggingface.co/spaces/lmarena-ai/arena-hard-auto",
                REPO_DIR,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Cloned to: {REPO_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"  Clone failed: {e.stderr}")
        print("  Trying with GIT_LFS_SKIP_SMUDGE=1 ...")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            [
                "git",
                "clone",
                "https://huggingface.co/spaces/lmarena-ai/arena-hard-auto",
                REPO_DIR,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        # Pull LFS files for data directory
        subprocess.run(
            ["git", "lfs", "pull", "--include", "data/**"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )


def parse_judgment_jsonl(filepath):
    """Parse a single judgment JSONL file, return dict of {question_id: score}."""
    scores = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_id = record.get("question_id", "")
            # Games may have multiple turns; take the judgment from the record
            # The judgment field contains the verdict string
            judgment_str = ""

            # Try different field names
            for field in ["judgment", "judge", "result"]:
                if field in record:
                    judgment_str = str(record[field])
                    break

            # Also check nested games structure
            games = record.get("games", [])
            if games and isinstance(games, list):
                # Use first game's judgment
                game = games[0] if len(games) > 0 else {}
                judgment_str = str(game.get("judgment", judgment_str))
                # Some formats store the score directly
                if "score" in game:
                    judgment_str = str(game["score"])

            # Extract verdict from judgment text
            score = None
            # Check for explicit verdict patterns in the text
            for verdict, val in JUDGMENT_MAP.items():
                if verdict in judgment_str:
                    score = val
                    break

            # Also try extracting from [[X]] pattern
            if score is None:
                import re

                match = re.search(r"\[\[([^\]]+)\]\]", judgment_str)
                if match:
                    verdict_text = match.group(1).strip()
                    for verdict, val in JUDGMENT_MAP.items():
                        if verdict == verdict_text:
                            score = val
                            break

            # Try direct score field
            if score is None and "score" in record:
                try:
                    raw_score = record["score"]
                    if isinstance(raw_score, (int, float)):
                        score = float(raw_score)
                    elif isinstance(raw_score, str) and raw_score in JUDGMENT_MAP:
                        score = JUDGMENT_MAP[raw_score]
                except (ValueError, TypeError):
                    pass

            if question_id and score is not None:
                # Average multiple games if question_id already seen
                if question_id in scores:
                    scores[question_id] = (scores[question_id] + score) / 2.0
                else:
                    scores[question_id] = score

    return scores


def build_response_matrix():
    """Build response matrix from judgment JSONL files."""
    # Find judgment directory
    judgment_dir = os.path.join(
        REPO_DIR, "data", "arena-hard-v0.1", "model_judgment", "gpt-4-1106-preview"
    )

    if not os.path.isdir(judgment_dir):
        # Try alternative paths
        for alt in [
            os.path.join(REPO_DIR, "data", "arena-hard-v0.1", "model_judgment"),
            os.path.join(REPO_DIR, "data", "model_judgment"),
            os.path.join(REPO_DIR, "model_judgment"),
        ]:
            if os.path.isdir(alt):
                # Use first subdirectory (evaluator)
                subdirs = [
                    d for d in os.listdir(alt)
                    if os.path.isdir(os.path.join(alt, d))
                ]
                if subdirs:
                    judgment_dir = os.path.join(alt, subdirs[0])
                    break

    if not os.path.isdir(judgment_dir):
        print(f"  ERROR: Judgment directory not found.")
        print(f"  Searched: {judgment_dir}")
        # List what we do have
        for root, dirs, files in os.walk(REPO_DIR):
            depth = root.replace(REPO_DIR, "").count(os.sep)
            if depth < 4:
                indent = " " * 2 * depth
                print(f"  {indent}{os.path.basename(root)}/")
        sys.exit(1)

    print(f"  Judgment directory: {judgment_dir}")

    # Find all JSONL files
    jsonl_files = sorted(
        [f for f in os.listdir(judgment_dir) if f.endswith(".jsonl")]
    )
    print(f"  Found {len(jsonl_files)} model judgment files")

    # Parse each model's judgments
    all_scores = {}
    all_question_ids = set()

    for jsonl_file in jsonl_files:
        model_name = jsonl_file.replace(".jsonl", "")
        filepath = os.path.join(judgment_dir, jsonl_file)

        scores = parse_judgment_jsonl(filepath)
        if scores:
            all_scores[model_name] = scores
            all_question_ids.update(scores.keys())
        else:
            print(f"    WARNING: No scores parsed from {jsonl_file}")

    # Build matrix
    question_ids = sorted(all_question_ids)
    models = sorted(all_scores.keys())

    print(f"\n  Building matrix: {len(models)} models x {len(question_ids)} prompts")

    matrix_data = {}
    for model in models:
        model_scores = all_scores[model]
        matrix_data[model] = [model_scores.get(qid, np.nan) for qid in question_ids]

    matrix_df = pd.DataFrame(matrix_data, index=question_ids)
    matrix_df.index.name = "question_id"

    # Transpose to models x questions
    matrix_df_t = matrix_df.T
    matrix_df_t.index.name = "Model"

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df_t.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df_t


def print_statistics(matrix_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  ARENA-HARD-AUTO STATISTICS")
    print(f"{'='*60}")

    n_models, n_questions = matrix_df.shape
    total_cells = n_models * n_questions
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Models:        {n_models}")
    print(f"    Prompts:       {n_questions}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Score distribution
    all_scores = matrix_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Score distribution (0-1 scale):")
        print(f"    Mean:   {np.mean(valid_scores):.3f}")
        print(f"    Median: {np.median(valid_scores):.3f}")
        print(f"    Std:    {np.std(valid_scores):.3f}")

        # Value histogram
        print(f"\n  Score value counts:")
        for val in sorted(set(JUDGMENT_MAP.values())):
            count = np.sum(valid_scores == val)
            pct = count / len(valid_scores) * 100
            label = [k for k, v in JUDGMENT_MAP.items() if v == val][0]
            bar = "#" * int(pct)
            print(f"    {val:.2f} ({label:5s}): {count:6,} ({pct:5.1f}%) {bar}")

    # Per-model win rates
    per_model_mean = matrix_df.mean(axis=1).sort_values(ascending=False)
    print(f"\n  Per-model mean score (win rate vs baseline):")
    print(f"    Best:   {per_model_mean.iloc[0]:.3f} ({per_model_mean.index[0]})")
    print(f"    Worst:  {per_model_mean.iloc[-1]:.3f} ({per_model_mean.index[-1]})")
    print(f"    Median: {per_model_mean.median():.3f}")
    print(f"    Std:    {per_model_mean.std():.3f}")

    print(f"\n  Top 10 models:")
    for model, score in per_model_mean.head(10).items():
        print(f"    {model:50s}  {score:.3f}")

    print(f"\n  Bottom 5 models:")
    for model, score in per_model_mean.tail(5).items():
        print(f"    {model:50s}  {score:.3f}")

    # Per-prompt difficulty
    per_prompt_mean = matrix_df.mean(axis=0)
    print(f"\n  Per-prompt difficulty (mean model score):")
    print(f"    Easiest: {per_prompt_mean.max():.3f}")
    print(f"    Hardest: {per_prompt_mean.min():.3f}")
    print(f"    Median:  {per_prompt_mean.median():.3f}")
    print(f"    Std:     {per_prompt_mean.std():.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("Arena-Hard-Auto Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Clone repo
    print("STEP 1: Cloning arena-hard-auto HuggingFace Space")
    print("-" * 60)
    clone_repo()

    # Step 2: Build response matrix
    print("\nSTEP 2: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix()

    # Step 3: Statistics
    print("\nSTEP 3: Detailed statistics")
    print("-" * 60)
    print_statistics(matrix_df)


if __name__ == "__main__":
    main()
