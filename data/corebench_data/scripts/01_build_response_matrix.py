#!/usr/bin/env python3
"""
01_build_response_matrix.py

Build a response matrix for CORE-Bench (Computational Reproducibility Benchmark).

Data source: https://github.com/siegelz/core-bench (cloned to /tmp/core-bench)
             Per-task results in results/ directory (also in agent_results.tar.gz)

Each CORE-Bench "task" is a (capsule_id, difficulty_level) pair.
  - 90 capsules total (45 train + 45 test)
  - 3 difficulty levels: easy, medium, hard
  - 270 tasks total

Each model configuration is (agent_architecture, llm_model, cost_limit, split).
Multiple runs may exist per model config; we report:
  - Binary response_matrix.csv: pass@k (1 if ANY run passed, 0 otherwise)
  - Score response_matrix_scores.csv: mean question-level accuracy across runs

Scoring (from evaluations.py in the repo):
  - A task "passes" if ALL written AND ALL vision questions are answered correctly.
  - Question accuracy = (correct_written + correct_vision) / (total_written + total_vision)

Also outputs:
  - task_metadata.csv: per-task metadata
  - hal_leaderboard.csv: 49 entries from the HAL CORE-Bench-Hard leaderboard (March 2026)
"""

import json
import os
import sys
import csv
from collections import defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = "/tmp/core-bench/results"
TRAIN_DATASET = "/tmp/core-bench/benchmark/dataset/core_train.json"
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

# Files to skip (post-paper runs, as noted in paper_figures.ipynb cell-16)
SKIP_FILES = {
    "test_coreagent_gpt4o_c-4": ["20250112-052839_codeocean_hard.json"],
    "test_coreagent_gpt4o-mini_c-4": ["20250112-065033_codeocean_hard.json"],
}

# Difficulty level extraction
LEVEL_MAP = {
    "codeocean_easy": "easy",
    "codeocean_medium": "medium",
    "codeocean_hard": "hard",
}


def extract_level(filename):
    """Extract the difficulty level from a result filename."""
    for key, level in LEVEL_MAP.items():
        if key in filename:
            return level
    return None


def is_task_success(result):
    """
    Binary pass/fail: 1 if ALL written + ALL vision questions correct.
    This matches the scoring in evaluations.py and paper_figures.ipynb.
    """
    written_ok = result["correct_written_answers"] == result["total_written_questions"]
    vision_ok = result["correct_vision_answers"] == result["total_vision_questions"]
    return int(written_ok and vision_ok)


def question_accuracy(result):
    """Compute fraction of questions answered correctly."""
    cw = result.get("correct_written_answers", 0)
    tw = result.get("total_written_questions", 0)
    cv = result.get("correct_vision_answers", 0)
    tv = result.get("total_vision_questions", 0)
    total = tw + tv
    if total == 0:
        return 1.0
    return (cw + cv) / total


def parse_model_config(dirname):
    """
    Parse results directory name into (split, agent, model, cost_limit).
    E.g., 'test_coreagent_gpt4o_c-4' -> ('test', 'CORE-Agent', 'GPT-4o', 'c-4')
    """
    parts = dirname.split("_")
    split = parts[0]
    cost_limit = parts[-1]  # e.g., "c-4"
    middle = "_".join(parts[1:-1])

    if middle.startswith("autogpt_"):
        agent = "AutoGPT"
        model_raw = middle[len("autogpt_"):]
    elif middle.startswith("coreagent_"):
        agent = "CORE-Agent"
        model_raw = middle[len("coreagent_"):]
    else:
        agent = "Unknown"
        model_raw = middle

    model_name_map = {
        "gpt4o": "GPT-4o",
        "gpt4o-mini": "GPT-4o-mini",
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o1-preview": "o1-preview",
        "claude_35_sonnet": "Claude-3.5-Sonnet",
    }
    model = model_name_map.get(model_raw, model_raw)
    return split, agent, model, cost_limit


def load_all_results():
    """
    Load all per-task results from all result directories.

    Returns:
        model_tasks: {column_name: {task_id: [(pass, score), ...]}}
        task_metadata: {task_id: {metadata dict}}
    """
    model_tasks = defaultdict(lambda: defaultdict(list))
    task_metadata = {}

    for dirname in sorted(os.listdir(RESULTS_DIR)):
        dirpath = os.path.join(RESULTS_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue

        split, agent, model, cost_limit = parse_model_config(dirname)
        skip_list = SKIP_FILES.get(dirname, [])

        # Column name for response matrix
        col_name = f"{agent} | {model} | {cost_limit} | {split}"

        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".json"):
                continue
            if fname in skip_list:
                continue

            level = extract_level(fname)
            if level is None:
                continue

            filepath = os.path.join(dirpath, fname)
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if "capsule_results" not in data:
                continue

            for result in data["capsule_results"]:
                capsule_id = result["capsule_id"]
                task_id = f"{capsule_id}__{level}"
                passed = is_task_success(result)
                score = question_accuracy(result)

                model_tasks[col_name][task_id].append((passed, score))

                if task_id not in task_metadata:
                    task_metadata[task_id] = {
                        "task_id": task_id,
                        "capsule_id": capsule_id,
                        "difficulty": level,
                        "split": split,
                        "field": result.get("field", ""),
                        "language": result.get("language", ""),
                        "capsule_title": result.get("capsule_title", ""),
                        "total_written_questions": result.get("total_written_questions", 0),
                        "total_vision_questions": result.get("total_vision_questions", 0),
                        "total_questions": (
                            result.get("total_written_questions", 0)
                            + result.get("total_vision_questions", 0)
                        ),
                    }

    return model_tasks, task_metadata


def write_hal_leaderboard(output_dir):
    """Write HAL CORE-Bench-Hard leaderboard data (scraped March 2026)."""
    entries = [
        ("Claude Code", "Claude Opus 4.5", 77.78, 87.16),
        ("Claude Code", "Claude Sonnet 4.5 (Sep 2025)", 62.22, 68.33),
        ("CORE-Agent", "Claude Opus 4.1 (Aug 2025)", 51.11, 412.42),
        ("Claude Code", "Claude Sonnet 4 (May 2025)", 46.67, 65.58),
        ("CORE-Agent", "Claude Sonnet 4.5 High (Sep 2025)", 44.44, 92.34),
        ("CORE-Agent", "Claude Opus 4.5 High (Nov 2025)", 42.22, 152.66),
        ("CORE-Agent", "Claude Opus 4.5 (Nov 2025)", 42.22, 168.99),
        ("Claude Code", "Claude Opus 4.1", 42.22, 331.79),
        ("CORE-Agent", "Claude Opus 4.1 High (Aug 2025)", 42.22, 509.95),
        ("CORE-Agent", "Gemini 3 Pro Preview High (Nov 2025)", 40.00, 86.60),
        ("HAL Generalist Agent", "Claude-3.7 Sonnet High (Feb 2025)", 37.78, 66.15),
        ("CORE-Agent", "Claude Sonnet 4.5 (Sep 2025)", 37.78, 97.15),
        ("HAL Generalist Agent", "o4-mini High (Apr 2025)", 35.56, 45.37),
        ("CORE-Agent", "Claude-3.7 Sonnet (Feb 2025)", 35.56, 73.04),
        ("HAL Generalist Agent", "Gemini 3 Pro Preview High (Nov 2025)", 35.56, 101.27),
        ("HAL Generalist Agent", "Claude Opus 4.1 (Aug 2025)", 35.56, 375.11),
        ("HAL Generalist Agent", "Claude Sonnet 4.5 (Sep 2025)", 33.33, 85.19),
        ("CORE-Agent", "Claude Sonnet 4 High (May 2025)", 33.33, 100.48),
        ("CORE-Agent", "GPT-4.1 (Apr 2025)", 33.33, 107.36),
        ("HAL Generalist Agent", "Claude Opus 4.5 (Nov 2025)", 33.33, 127.41),
        ("HAL Generalist Agent", "Claude Opus 4.1 High (Aug 2025)", 33.33, 358.47),
        ("HAL Generalist Agent", "Claude-3.7 Sonnet (Feb 2025)", 31.11, 56.64),
        ("HAL Generalist Agent", "Claude Opus 4.5 High (Nov 2025)", 31.11, 112.38),
        ("CORE-Agent", "Claude Sonnet 4 (May 2025)", 28.89, 50.27),
        ("HAL Generalist Agent", "Claude Sonnet 4.5 High (Sep 2025)", 28.89, 87.77),
        ("CORE-Agent", "GPT-5 Medium (Aug 2025)", 26.67, 31.76),
        ("CORE-Agent", "o4-mini High (Apr 2025)", 26.67, 61.35),
        ("CORE-Agent", "Claude-3.7 Sonnet High (Feb 2025)", 24.44, 72.47),
        ("CORE-Agent", "o3 Medium (Apr 2025)", 24.44, 120.47),
        ("HAL Generalist Agent", "GPT-4.1 (Apr 2025)", 22.22, 58.32),
        ("HAL Generalist Agent", "o3 Medium (Apr 2025)", 22.22, 88.34),
        ("CORE-Agent", "Gemini 2.5 Pro Preview (Mar 2025)", 22.22, 182.34),
        ("CORE-Agent", "DeepSeek V3.1 (Aug 2025)", 20.00, 12.55),
        ("CORE-Agent", "DeepSeek V3 (Mar 2025)", 17.78, 25.26),
        ("CORE-Agent", "o4-mini Low (Apr 2025)", 17.78, 31.79),
        ("HAL Generalist Agent", "o4-mini Low (Apr 2025)", 15.56, 22.50),
        ("CORE-Agent", "GPT-OSS-120B (Aug 2025)", 11.11, 4.21),
        ("CORE-Agent", "GPT-OSS-120B High (Aug 2025)", 11.11, 4.21),
        ("CORE-Agent", "Gemini 2.0 Flash (Feb 2025)", 11.11, 12.46),
        ("HAL Generalist Agent", "GPT-5 Medium (Aug 2025)", 11.11, 29.75),
        ("CORE-Agent", "Claude Haiku 4.5 (Oct 2025)", 11.11, 43.93),
        ("HAL Generalist Agent", "GPT-OSS-120B High (Aug 2025)", 8.89, 2.05),
        ("HAL Generalist Agent", "GPT-OSS-120B (Aug 2025)", 8.89, 2.79),
        ("HAL Generalist Agent", "DeepSeek V3 (Mar 2025)", 8.89, 4.69),
        ("HAL Generalist Agent", "DeepSeek R1 (May 2025)", 8.89, 7.77),
        ("CORE-Agent", "DeepSeek R1 (Jan 2025)", 6.67, 81.11),
        ("HAL Generalist Agent", "DeepSeek R1 (Jan 2025)", 4.45, 24.95),
        ("HAL Generalist Agent", "Gemini 2.0 Flash (Feb 2025)", 4.44, 7.06),
        ("HAL Generalist Agent", "Gemini 2.5 Pro Preview (Mar 2025)", 4.44, 30.38),
    ]

    filepath = os.path.join(output_dir, "hal_leaderboard.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "scaffold", "model", "accuracy_pct", "cost_usd"])
        for i, (scaffold, model, acc, cost) in enumerate(entries, 1):
            writer.writerow([i, scaffold, model, acc, cost])
    print(f"  Wrote {filepath} ({len(entries)} entries)")
    return entries


def main():
    print("=" * 70)
    print("CORE-Bench Response Matrix Builder")
    print("=" * 70)

    if not os.path.isdir(RESULTS_DIR):
        print(f"\nERROR: Results directory not found: {RESULTS_DIR}")
        print("Clone the repo first:")
        print("  git clone https://github.com/siegelz/core-bench /tmp/core-bench")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load all results ─────────────────────────────────────────
    print("\n[1/5] Loading all per-task results...")
    model_tasks, task_metadata = load_all_results()
    all_task_ids = sorted(task_metadata.keys())
    all_model_cols = sorted(model_tasks.keys())

    print(f"  Found {len(all_task_ids)} unique tasks")
    print(f"  Found {len(all_model_cols)} model configurations")

    # Count by split/difficulty
    for split in ["test", "train"]:
        for diff in ["easy", "medium", "hard"]:
            count = sum(
                1 for t in task_metadata.values()
                if t["split"] == split and t["difficulty"] == diff
            )
            print(f"    {split}/{diff}: {count} tasks")

    # ── Step 2: Write task metadata ──────────────────────────────────────
    print("\n[2/5] Writing task metadata...")
    meta_path = OUTPUT_DIR / "task_metadata.csv"
    meta_fields = [
        "task_id", "capsule_id", "difficulty", "split", "field", "language",
        "capsule_title", "total_written_questions", "total_vision_questions",
        "total_questions",
    ]
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        for tid in all_task_ids:
            writer.writerow(task_metadata[tid])
    print(f"  Wrote {meta_path} ({len(all_task_ids)} rows)")

    # ── Step 3: Write binary response matrix (pass@k) ────────────────────
    print("\n[3/5] Writing binary response matrix (pass@k)...")
    bin_path = OUTPUT_DIR / "response_matrix.csv"
    with open(bin_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id"] + all_model_cols)
        for tid in all_task_ids:
            row = [tid]
            for col in all_model_cols:
                runs = model_tasks[col].get(tid, [])
                if not runs:
                    row.append("")
                else:
                    # pass@k: 1 if any run passed
                    row.append(int(any(r[0] for r in runs)))
            writer.writerow(row)
    print(f"  Wrote {bin_path} ({len(all_task_ids)} rows x {len(all_model_cols)} model cols)")

    # ── Step 4: Write score response matrix (mean question accuracy) ─────
    print("\n[4/5] Writing score response matrix (mean question accuracy)...")
    score_path = OUTPUT_DIR / "response_matrix_scores.csv"
    with open(score_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id"] + all_model_cols)
        for tid in all_task_ids:
            row = [tid]
            for col in all_model_cols:
                runs = model_tasks[col].get(tid, [])
                if not runs:
                    row.append("")
                else:
                    mean_score = sum(r[1] for r in runs) / len(runs)
                    row.append(f"{mean_score:.4f}")
            writer.writerow(row)
    print(f"  Wrote {score_path} ({len(all_task_ids)} rows x {len(all_model_cols)} model cols)")

    # ── Step 5: Write HAL leaderboard ─────────────────────────────────────
    print("\n[5/5] Writing HAL leaderboard data...")
    hal_entries = write_hal_leaderboard(OUTPUT_DIR)

    # ── Summary Statistics ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    n_tasks = len(all_task_ids)
    n_models = len(all_model_cols)
    total_cells = n_tasks * n_models
    filled_cells = sum(
        1 for tid in all_task_ids for col in all_model_cols
        if model_tasks[col].get(tid)
    )
    fill_rate = filled_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\nResponse Matrix Dimensions: {n_tasks} tasks x {n_models} models")
    print(f"Total cells: {total_cells}")
    print(f"Filled cells: {filled_cells} ({fill_rate:.1f}%)")
    print(f"Empty cells: {total_cells - filled_cells} ({100 - fill_rate:.1f}%)")

    print(f"\nScore Types:")
    print(f"  response_matrix.csv: binary pass@k (1 if any run passed, 0 otherwise)")
    print(f"  response_matrix_scores.csv: mean question-level accuracy [0,1]")

    print(f"\nModel Configurations ({n_models}):")
    for col in all_model_cols:
        n_tasks_with_data = sum(1 for tid in all_task_ids if model_tasks[col].get(tid))
        n_runs_total = sum(len(model_tasks[col].get(tid, [])) for tid in all_task_ids)
        scores = [
            any(r[0] for r in model_tasks[col][tid])
            for tid in all_task_ids if model_tasks[col].get(tid)
        ]
        pass_rate = sum(scores) / len(scores) * 100 if scores else 0
        print(f"  {col}")
        print(f"    Tasks: {n_tasks_with_data}/{n_tasks}, "
              f"Runs: {n_runs_total}, "
              f"Pass@k: {pass_rate:.1f}%")

    # Per-difficulty stats for test models
    print(f"\nPer-Difficulty Pass@k Rates (test set models only):")
    for col in all_model_cols:
        if "| test" not in col:
            continue
        print(f"  {col}:")
        for diff in ["easy", "medium", "hard"]:
            diff_tasks = [
                tid for tid in all_task_ids
                if task_metadata[tid]["difficulty"] == diff
                and task_metadata[tid]["split"] == "test"
                and model_tasks[col].get(tid)
            ]
            if diff_tasks:
                passes = sum(
                    any(r[0] for r in model_tasks[col][tid])
                    for tid in diff_tasks
                )
                print(f"    {diff}: {passes}/{len(diff_tasks)} = "
                      f"{passes/len(diff_tasks)*100:.1f}%")

    # Task distribution
    fields = defaultdict(int)
    langs = defaultdict(int)
    for m in task_metadata.values():
        if m["field"]:
            fields[m["field"]] += 1
        if m["language"]:
            langs[m["language"]] += 1

    print(f"\nTask Distribution:")
    print(f"  Fields: {dict(sorted(fields.items()))}")
    print(f"  Languages: {dict(sorted(langs.items()))}")

    # HAL leaderboard summary
    print(f"\nHAL Leaderboard (CORE-Bench-Hard): {len(hal_entries)} entries")
    print(f"  Top performer: {hal_entries[0][0]} / {hal_entries[0][1]} = {hal_entries[0][2]}%")
    print(f"  Lowest: {hal_entries[-1][0]} / {hal_entries[-1][1]} = {hal_entries[-1][2]}%")
    print(f"  Median accuracy: {sorted([e[2] for e in hal_entries])[len(hal_entries)//2]}%")

    print(f"\nOutput Files:")
    for fname in ["response_matrix.csv", "response_matrix_scores.csv",
                   "task_metadata.csv", "hal_leaderboard.csv"]:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  {fpath} ({size:,} bytes)")

    print()


if __name__ == "__main__":
    main()
