"""
Build ARC-AGI response matrices from ARC Prize Foundation per-model per-task evaluation data.

Data sources:
  1. arcprize/arc_agi_v1_public_eval (HuggingFace): Per-model per-task results for ARC-AGI-1
     public evaluation set (400 tasks). Contains results.json with per-task scores for ~30 models,
     plus individual task JSON files (with model answers) for ~52 models total.
  2. arcprize/arc_agi_v2_public_eval (HuggingFace): Same for ARC-AGI-2 public evaluation (120 tasks).
  3. evaluations.json (arcprize.org): Leaderboard aggregate scores for 134 models/configurations.
  4. fchollet/ARC-AGI (GitHub): Ground truth task files for scoring models without results.json.
  5. arcprize/arc_agi_2_human_testing (HuggingFace): Human per-task per-participant solve data.

Outputs:
  - response_matrix.csv: Binary (models x tasks) matrix for ARC-AGI-1 public eval (400 tasks)
  - response_matrix_v2.csv: Binary (models x tasks) matrix for ARC-AGI-2 public eval (120 tasks)
  - model_summary.csv: Per-model aggregate statistics

The score in results.json is per-task: 1.0 = solved (at least one of 2 attempts matched ground
truth exactly), 0.0 = unsolved. For tasks with multiple test outputs, score can be 0.5 if only
some test outputs were solved. We binarize at threshold >= 0.5.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HOME"] = HF_HOME

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_DIR = RAW_DIR / "arc-agi-1" / "data" / "evaluation"

V1_REPO = "arcprize/arc_agi_v1_public_eval"
V2_REPO = "arcprize/arc_agi_v2_public_eval"
HUMAN_REPO = "arcprize/arc_agi_2_human_testing"


# ── Helper functions ───────────────────────────────────────────────────────────

def load_ground_truth(gt_dir: Path) -> dict:
    """Load ground truth outputs for all evaluation tasks.

    Returns:
        dict mapping task_id -> list of expected test outputs (each is a 2D list)
    """
    ground_truth = {}
    for json_file in sorted(gt_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            data = json.load(f)
        test_outputs = [t["output"] for t in data["test"]]
        ground_truth[task_id] = test_outputs
    return ground_truth


def score_model_task(task_json_data: list, expected_outputs: list) -> float:
    """Score a model's response for a single task against ground truth.

    Each task can have multiple test items. For each test item, if either
    attempt_1 or attempt_2 matches the expected output exactly, it scores 1.
    Final score = average across test items.

    Args:
        task_json_data: List of test items, each with attempt_1/attempt_2.
        expected_outputs: List of expected 2D output grids.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not task_json_data or not expected_outputs:
        return 0.0

    n_tests = min(len(task_json_data), len(expected_outputs))
    correct = 0

    for i in range(n_tests):
        test_item = task_json_data[i]
        expected = expected_outputs[i]

        for attempt_key in ["attempt_1", "attempt_2"]:
            if attempt_key in test_item and test_item[attempt_key] is not None:
                answer = test_item[attempt_key].get("answer")
                if answer is not None and answer == expected:
                    correct += 1
                    break

    return correct / n_tests if n_tests > 0 else 0.0


def download_results_json(repo_id: str, model_name: str) -> dict | None:
    """Download results.json for a model from HuggingFace."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}/results.json",
            repo_type="dataset",
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return None


def download_task_file(repo_id: str, model_name: str, task_id: str) -> list | None:
    """Download individual task JSON for a model from HuggingFace."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}/{task_id}.json",
            repo_type="dataset",
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return None


def get_models_and_tasks(repo_id: str) -> tuple[dict, dict]:
    """Get all models and their task files from a HuggingFace repo.

    Returns:
        (models_with_results, models_without_results)
        Each is dict: model_name -> set of task_ids
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")

    model_tasks = defaultdict(set)
    models_with_results = set()

    for f in files:
        if "/" not in f or f.startswith("."):
            continue
        parts = f.split("/")
        model = parts[0]
        filename = parts[-1]

        if filename == "results.json":
            models_with_results.add(model)
        elif filename.endswith(".json"):
            task_id = filename.replace(".json", "")
            model_tasks[model].add(task_id)

    with_results = {m: model_tasks.get(m, set()) for m in models_with_results}
    without_results = {
        m: tasks for m, tasks in model_tasks.items()
        if m not in models_with_results
    }

    return with_results, without_results


# ── Main processing ────────────────────────────────────────────────────────────

def build_response_matrix_from_results(
    repo_id: str,
    ground_truth: dict | None = None,
    dataset_label: str = "v1",
) -> pd.DataFrame:
    """Build response matrix for a given ARC-AGI dataset version.

    Strategy:
    1. For models with results.json: extract per-task scores directly.
    2. For models without results.json: download individual task files and
       score against ground truth.

    Returns:
        DataFrame with model names as rows, task IDs as columns, values 0/1/NaN.
    """
    print(f"\n{'='*70}")
    print(f"Building response matrix for {dataset_label} from {repo_id}")
    print(f"{'='*70}")

    models_with_results, models_without_results = get_models_and_tasks(repo_id)

    print(f"\nModels with results.json: {len(models_with_results)}")
    print(f"Models without results.json: {len(models_without_results)}")

    all_task_scores = {}  # model_name -> {task_id: score}

    # ── Phase 1: Models with results.json ──────────────────────────────────
    print("\n--- Phase 1: Processing models with results.json ---")
    for model in sorted(models_with_results):
        print(f"  Downloading results for {model}...", end=" ", flush=True)
        results = download_results_json(repo_id, model)
        if results and "task_results" in results:
            task_results = results["task_results"]
            scores = {}
            for task_id, task_data in task_results.items():
                scores[task_id] = task_data.get("score", 0.0)
            all_task_scores[model] = scores
            n_solved = sum(1 for s in scores.values() if s >= 0.5)
            print(f"{len(scores)} tasks, {n_solved} solved "
                  f"({100*n_solved/len(scores):.1f}%)")
        else:
            print("FAILED or empty")

    # ── Phase 2: Models without results.json ───────────────────────────────
    if ground_truth and models_without_results:
        print("\n--- Phase 2: Scoring models without results.json against GT ---")

        # Get the set of task IDs that exist in ground truth
        gt_task_ids = set(ground_truth.keys())

        for model in sorted(models_without_results):
            task_ids = models_without_results[model]
            # Only score tasks that have ground truth
            scoreable = task_ids & gt_task_ids
            if not scoreable:
                print(f"  {model}: No scoreable tasks (skip)")
                continue

            print(f"  Scoring {model} ({len(scoreable)} tasks)...",
                  end=" ", flush=True)
            scores = {}
            for task_id in sorted(scoreable):
                task_data = download_task_file(repo_id, model, task_id)
                if task_data is not None:
                    score = score_model_task(task_data, ground_truth[task_id])
                    scores[task_id] = score

            all_task_scores[model] = scores
            n_solved = sum(1 for s in scores.values() if s >= 0.5)
            print(f"{len(scores)} scored, {n_solved} solved "
                  f"({100*n_solved/len(scores):.1f}%)")
    elif models_without_results:
        print(f"\n--- Skipping {len(models_without_results)} models without "
              f"results.json (no ground truth available) ---")

    # ── Build matrix ───────────────────────────────────────────────────────
    print("\n--- Building response matrix ---")

    # Collect all task IDs
    all_task_ids = set()
    for scores in all_task_scores.values():
        all_task_ids.update(scores.keys())
    all_task_ids = sorted(all_task_ids)

    # Build DataFrame
    rows = []
    for model in sorted(all_task_scores.keys()):
        scores = all_task_scores[model]
        row = {"model": model}
        for task_id in all_task_ids:
            if task_id in scores:
                # Binarize: >= 0.5 -> 1, else 0
                row[task_id] = 1 if scores[task_id] >= 0.5 else 0
            else:
                row[task_id] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")

    return df


def build_human_response_matrix() -> pd.DataFrame | None:
    """Build a human response matrix from ARC-AGI-2 human testing data.

    Each unique session_ID is treated as a "model" (human participant).
    For each task, a human is considered to have solved it if any of their
    attempts had correct_submissions > 0.

    Returns:
        DataFrame with session IDs as rows, task IDs as columns, values 0/1.
    """
    from huggingface_hub import hf_hub_download
    print("\n" + "="*70)
    print("Building human response matrix from ARC-AGI-2 human testing data")
    print("="*70)

    try:
        local = hf_hub_download(
            repo_id=HUMAN_REPO,
            filename="test_pair_attempts.csv",
            repo_type="dataset",
        )
        df = pd.read_csv(local)
    except Exception as e:
        print(f"Failed to download human testing data: {e}")
        return None

    print(f"Raw data: {len(df)} rows, {df['task_ID'].nunique()} tasks, "
          f"{df['session_ID'].nunique()} sessions")

    # Aggregate: for each (session, task), did they solve it?
    solved = df.groupby(["session_ID", "task_ID"])["correct_submissions"].max()
    solved = (solved > 0).astype(int).reset_index()
    solved.columns = ["session_ID", "task_ID", "solved"]

    # Pivot to matrix
    matrix = solved.pivot(index="session_ID", columns="task_ID", values="solved")
    matrix.index.name = "model"

    # Add prefix to distinguish human sessions
    matrix.index = ["human_" + str(s) for s in matrix.index]

    print(f"Human matrix: {matrix.shape[0]} participants x {matrix.shape[1]} tasks")
    n_filled = matrix.notna().sum().sum()
    total = matrix.shape[0] * matrix.shape[1]
    print(f"Fill rate: {n_filled}/{total} ({100*n_filled/total:.1f}%)")

    return matrix


def add_leaderboard_aggregate_scores(
    df: pd.DataFrame,
    evaluations_path: Path,
    dataset_filter: str,
) -> pd.DataFrame:
    """Add leaderboard aggregate scores for models not in the response matrix.

    These models only have aggregate scores (not per-task), so they're stored
    in a separate summary rather than added to the response matrix.
    """
    with open(evaluations_path) as f:
        evaluations = json.load(f)

    leaderboard = {}
    for entry in evaluations:
        if entry["datasetId"] == dataset_filter:
            model_id = entry["modelId"]
            score = entry["score"]
            cost = entry.get("costPerTask")
            leaderboard[model_id] = {"score": score, "costPerTask": cost}

    return leaderboard


def print_matrix_stats(df: pd.DataFrame, label: str):
    """Print statistics about a response matrix."""
    print(f"\n{'─'*60}")
    print(f"  Response Matrix Statistics: {label}")
    print(f"{'─'*60}")
    print(f"  Dimensions: {df.shape[0]} models x {df.shape[1]} tasks")

    n_filled = df.notna().sum().sum()
    total = df.shape[0] * df.shape[1]
    fill_pct = 100 * n_filled / total if total > 0 else 0
    print(f"  Fill rate: {n_filled}/{total} ({fill_pct:.1f}%)")

    filled_vals = df.values[~np.isnan(df.values.astype(float))]
    if len(filled_vals) > 0:
        n_ones = (filled_vals == 1).sum()
        n_zeros = (filled_vals == 0).sum()
        print(f"  Value distribution: {n_ones} solved (1), {n_zeros} unsolved (0)")
        print(f"  Overall solve rate: {100*n_ones/(n_ones+n_zeros):.1f}%")

    # Per-model stats
    model_means = df.mean(axis=1)
    print(f"\n  Per-model solve rate:")
    print(f"    Min:    {100*model_means.min():.1f}%")
    print(f"    Median: {100*model_means.median():.1f}%")
    print(f"    Max:    {100*model_means.max():.1f}%")

    # Per-task stats
    task_means = df.mean(axis=0)
    print(f"\n  Per-task solve rate:")
    print(f"    Min:    {100*task_means.min():.1f}%")
    print(f"    Median: {100*task_means.median():.1f}%")
    print(f"    Max:    {100*task_means.max():.1f}%")

    n_always_solved = (task_means == 1.0).sum()
    n_never_solved = (task_means == 0.0).sum()
    print(f"    Always solved: {n_always_solved} tasks")
    print(f"    Never solved:  {n_never_solved} tasks")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("ARC-AGI Response Matrix Builder")
    print("=" * 70)

    # Load ground truth
    print("\nLoading ground truth from", GROUND_TRUTH_DIR)
    ground_truth = load_ground_truth(GROUND_TRUTH_DIR)
    print(f"Loaded {len(ground_truth)} evaluation tasks")

    # ── Build V1 response matrix (400 tasks) ───────────────────────────────
    df_v1 = build_response_matrix_from_results(
        V1_REPO, ground_truth=ground_truth, dataset_label="ARC-AGI-1 Public Eval"
    )
    print_matrix_stats(df_v1, "ARC-AGI-1 Public Eval")

    # Save V1
    v1_path = PROCESSED_DIR / "response_matrix.csv"
    df_v1.to_csv(v1_path)
    print(f"\nSaved V1 matrix to {v1_path}")

    # ── Build V2 response matrix (120 tasks) ───────────────────────────────
    # Note: V2 ground truth is NOT in the fchollet/ARC-AGI repo (that's V1 only).
    # For V2, we rely on results.json files and score individual task files
    # where possible (but we don't have V2 ground truth to do manual scoring).
    df_v2 = build_response_matrix_from_results(
        V2_REPO, ground_truth=None, dataset_label="ARC-AGI-2 Public Eval"
    )
    print_matrix_stats(df_v2, "ARC-AGI-2 Public Eval")

    # Save V2
    v2_path = PROCESSED_DIR / "response_matrix_v2.csv"
    df_v2.to_csv(v2_path)
    print(f"\nSaved V2 matrix to {v2_path}")

    # ── Build model summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Building model summary with leaderboard data")
    print("=" * 70)

    evaluations_path = RAW_DIR / "evaluations.json"
    summary_rows = []

    # Add V1 models from response matrix
    for model in df_v1.index:
        row = df_v1.loc[model]
        n_tasks = row.notna().sum()
        n_solved = (row == 1).sum()
        solve_rate = n_solved / n_tasks if n_tasks > 0 else 0
        summary_rows.append({
            "model": model,
            "dataset": "ARC-AGI-1_Public_Eval",
            "n_tasks_evaluated": int(n_tasks),
            "n_solved": int(n_solved),
            "solve_rate": round(solve_rate, 4),
            "source": "per_task_results",
        })

    # Add V2 models from response matrix
    for model in df_v2.index:
        row = df_v2.loc[model]
        n_tasks = row.notna().sum()
        n_solved = (row == 1).sum()
        solve_rate = n_solved / n_tasks if n_tasks > 0 else 0
        summary_rows.append({
            "model": model,
            "dataset": "ARC-AGI-2_Public_Eval",
            "n_tasks_evaluated": int(n_tasks),
            "n_solved": int(n_solved),
            "solve_rate": round(solve_rate, 4),
            "source": "per_task_results",
        })

    # Add leaderboard-only models (aggregate scores only)
    if evaluations_path.exists():
        with open(evaluations_path) as f:
            evaluations = json.load(f)

        # Models already in response matrices
        existing_v1 = set(df_v1.index)
        existing_v2 = set(df_v2.index)

        dataset_map = {
            "v1_Public_Eval": ("ARC-AGI-1_Public_Eval", existing_v1, 400),
            "v1_Semi_Private": ("ARC-AGI-1_Semi_Private", set(), 100),
            "v2_Public_Eval": ("ARC-AGI-2_Public_Eval", existing_v2, 120),
            "v2_Semi_Private": ("ARC-AGI-2_Semi_Private", set(), 120),
            "v2_Private_Eval": ("ARC-AGI-2_Private_Eval", set(), None),
        }

        for entry in evaluations:
            ds_id = entry["datasetId"]
            model_id = entry["modelId"]
            score = entry["score"]

            if ds_id in dataset_map:
                label, existing, n_tasks = dataset_map[ds_id]
                if model_id not in existing:
                    summary_rows.append({
                        "model": model_id,
                        "dataset": label,
                        "n_tasks_evaluated": n_tasks,
                        "n_solved": round(score * n_tasks) if n_tasks else None,
                        "solve_rate": round(score, 4),
                        "source": "leaderboard_aggregate",
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved model summary to {summary_path}")
    print(f"Total entries: {len(summary_df)}")
    print(f"  Per-task results: {(summary_df['source'] == 'per_task_results').sum()}")
    print(f"  Leaderboard aggregate: "
          f"{(summary_df['source'] == 'leaderboard_aggregate').sum()}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nARC-AGI-1 Public Eval response matrix: {df_v1.shape[0]} models x "
          f"{df_v1.shape[1]} tasks")
    print(f"ARC-AGI-2 Public Eval response matrix: {df_v2.shape[0]} models x "
          f"{df_v2.shape[1]} tasks")
    print(f"Model summary: {len(summary_df)} entries across "
          f"{summary_df['dataset'].nunique()} dataset splits")
    print(f"\nOutput files:")
    print(f"  {v1_path}")
    print(f"  {v2_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
