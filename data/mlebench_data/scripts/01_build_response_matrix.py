#!/usr/bin/env python3
"""
Build a response matrix for MLE-bench (OpenAI's Kaggle ML competition benchmark).

Data source: https://github.com/openai/mle-bench
    - runs/ directory contains grading reports per experiment run group
    - runs/run_group_experiments.csv maps experiment IDs to run groups
    - experiments/splits/split75.txt defines the canonical 75-competition split
    - experiments/competition_categories.csv has task metadata

The response matrix has:
    Rows = 75 Kaggle competitions (tasks)
    Columns = agent/model configurations (experiments)
    Values = best medal achieved across seeds (gold=3, silver=2, bronze=1,
             above_median=0.5, none=0)
             Also outputs raw score matrices and binary any_medal matrices.

Output files:
    processed/response_matrix.csv             -- medal-level response matrix (main)
    processed/response_matrix_binary.csv      -- binary any_medal matrix
    processed/response_matrix_scores.csv      -- raw score matrix (best across seeds)
    processed/response_matrix_above_median.csv-- binary above-median matrix
    processed/task_metadata.csv               -- competition metadata
    processed/experiment_metadata.csv         -- experiment descriptions
    processed/summary_stats.txt              -- summary statistics
    processed/per_seed_details.json          -- per-seed raw data
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# -- Configuration -----------------------------------------------------------
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = Path("/tmp/mle-bench")
RUNS_DIR = REPO_DIR / "runs"
EXPERIMENTS_DIR = REPO_DIR / "experiments"
RUN_GROUP_CSV = RUNS_DIR / "run_group_experiments.csv"
SPLIT_FILE = EXPERIMENTS_DIR / "splits" / "split75.txt"
CATEGORIES_CSV = EXPERIMENTS_DIR / "competition_categories.csv"

OUTPUT_DIR = _BENCHMARK_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -- Experiment descriptions (from README) -----------------------------------
EXPERIMENT_DESCRIPTIONS = {
    "scaffolding-gpt4o-aide": "GPT-4o + AIDE scaffold (24h, 1 GPU)",
    "scaffolding-gpt4o-mlab": "GPT-4o + MLAB scaffold",
    "scaffolding-gpt4o-opendevin": "GPT-4o + OpenDevin scaffold",
    "models-o1-preview-aide": "o1-preview + AIDE scaffold",
    "models-claude35sonnet-aide": "Claude 3.5 Sonnet + AIDE scaffold",
    "models-llama-3.1-405B-instruct-aide": "LLaMA 3.1 405B + AIDE scaffold",
    "biggpu-gpt4o-aide": "GPT-4o + AIDE (2x A10 GPUs)",
    "cpu-gpt4o-aide": "GPT-4o + AIDE (CPU only)",
    "extratime-gpt4o-aide": "GPT-4o + AIDE (100h time limit)",
    "obfuscation-gpt4o-aide": "GPT-4o + AIDE (obfuscated descriptions)",
    "o1-preview-R&D-Agent": "o1-preview + R&D-Agent scaffold",
    "deepseek-r1-ML-Master": "DeepSeek-R1 + ML-Master scaffold",
    "deepseek-v3.2-speciale-ML-Master-2.0": "DeepSeek-V3.2-Speciale + ML-Master-2.0",
    "multi-agent-Neo": "Multi-Agent (GPT-4.1 + Claude Sonnet 4.0) + NEO",
    "o3-gpt-4.1-R&D-Agent": "o3 + GPT-4.1 + R&D-Agent scaffold",
    "deepseek-r1-InternAgent": "DeepSeek-R1 + InternAgent scaffold",
    "gpt-5-R&D-Agent": "GPT-5 + R&D-Agent scaffold",
    "operand-ensemble": "Operand Ensemble",
    "Famou-Agent": "Gemini-2.5-Pro + Famou-Agent scaffold",
    "Famou-Agent-2.0": "Gemini-2.5-Pro + Famou-Agent-2.0 scaffold",
    "MLE-STAR-Pro-1.0": "MLE-STAR-Pro-1.0 scaffold",
    "MLE-STAR-Pro-1.5": "MLE-STAR-Pro-1.5 scaffold",
    "AIRA-dojo": "o3 + AIRA-DOJO Greedy scaffold (H200)",
    "Thesis": "GPT-5-codex + custom scaffold (H100)",
    "Leeroo": "Ensemble (Gemini-3-Pro, GPT-5, GPT-5-mini) + Leeroo",
    "PiEvolve_24hrs": "Gemini-3-Pro-preview + PiEvolve (24h, H100)",
    "PiEvolve_12hrs": "Gemini-3-Pro-preview + PiEvolve (12h, H100)",
    "LoongFlow": "Gemini-3-Flash-preview + LoongFlow scaffold",
    "Disarray": "Ensemble (Claude-Opus-4.5, Sonnet-4.5, GPT-5.2, Gemini-3-Pro)",
    "MLEvolve": "Gemini-3-Pro-preview + MLEvolve (H200)",
}


def load_competition_ids():
    """Load the canonical 75 competitions from the split file."""
    with open(SPLIT_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def load_experiment_to_run_groups():
    """Parse run_group_experiments.csv -> {experiment_id: [run_group, ...]}."""
    mapping = defaultdict(list)
    with open(RUN_GROUP_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("experiment_id"):
                continue
            parts = line.split(",", maxsplit=1)
            exp_id = parts[0].strip()
            run_group = parts[1].strip()
            mapping[exp_id].append(run_group)
    return dict(mapping)


def find_grading_report(run_group_dir):
    """Find the grading report JSON in a run group directory."""
    if not run_group_dir.is_dir():
        return None
    reports = sorted(run_group_dir.glob("*grading_report*.json"))
    if not reports:
        return None
    return reports[0]


def load_grading_report(report_path):
    """Load a grading report and return per-competition data."""
    with open(report_path) as f:
        data = json.load(f)
    return data.get("competition_reports", [])


def medal_level(report):
    """Convert a competition report to a numeric medal level.

    Returns: 3 (gold), 2 (silver), 1 (bronze), 0.5 (above median), 0 (none)
    """
    if report.get("gold_medal"):
        return 3
    elif report.get("silver_medal"):
        return 2
    elif report.get("bronze_medal"):
        return 1
    elif report.get("above_median"):
        return 0.5
    else:
        return 0


def medal_name(level):
    """Convert numeric medal level to name."""
    names = {3: "gold", 2: "silver", 1: "bronze", 0.5: "above_median", 0: "none"}
    return names.get(level, "none")


def load_task_metadata():
    """Load competition metadata from competition_categories.csv."""
    metadata = {}
    with open(CATEGORIES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_id = row.get("competition_id", "").strip()
            if comp_id:
                metadata[comp_id] = {
                    "competition_id": comp_id,
                    "category": row.get("category", "").strip(),
                    "dataset_size_GB": row.get("dataset_size_GB", "").strip(),
                    "complexity": row.get("Complexity", "").strip(),
                }
    return metadata


def main():
    print("=" * 80)
    print("MLE-bench Response Matrix Builder")
    print("=" * 80)

    # -- Step 1: Load canonical competitions ---------------------------------
    competition_ids = load_competition_ids()
    comp_set = set(competition_ids)
    print(f"\nLoaded {len(competition_ids)} competitions from split75")

    # -- Step 2: Load experiment -> run group mapping ------------------------
    exp_to_groups = load_experiment_to_run_groups()
    experiment_ids = sorted(exp_to_groups.keys())
    print(f"Found {len(experiment_ids)} experiment configurations")
    for exp_id in experiment_ids:
        n_groups = len(exp_to_groups[exp_id])
        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, "")
        print(f"  {exp_id}: {n_groups} run groups -- {desc}")

    # -- Step 3: Extract per-competition per-experiment results ---------------
    # medal_matrix[comp_id][exp_id] = best medal level across seeds
    # score_matrix[comp_id][exp_id] = best raw score across seeds
    # any_medal_matrix[comp_id][exp_id] = 1 if any seed got a medal
    # above_median_matrix[comp_id][exp_id] = 1 if any seed was above median
    # seed_details[exp_id][comp_id] = list of per-seed dicts

    medal_matrix = defaultdict(dict)
    score_matrix = defaultdict(dict)
    any_medal_matrix = defaultdict(dict)
    above_median_matrix = defaultdict(dict)
    seed_details = defaultdict(lambda: defaultdict(list))

    for exp_id in experiment_ids:
        run_groups = exp_to_groups[exp_id]
        all_reports = []

        for rg in sorted(run_groups):
            rg_dir = RUNS_DIR / rg
            report_path = find_grading_report(rg_dir)
            if report_path is None:
                print(
                    f"  WARNING: No grading report for run group '{rg}' "
                    f"(experiment '{exp_id}')"
                )
                continue
            reports = load_grading_report(report_path)
            all_reports.extend(reports)

        # Group by competition
        comp_reports = defaultdict(list)
        for r in all_reports:
            cid = r.get("competition_id", "")
            if cid in comp_set:
                comp_reports[cid].append(r)

        # Aggregate per competition
        for comp_id in competition_ids:
            reports_for_comp = comp_reports.get(comp_id, [])

            if not reports_for_comp:
                medal_matrix[comp_id][exp_id] = None
                score_matrix[comp_id][exp_id] = None
                any_medal_matrix[comp_id][exp_id] = None
                above_median_matrix[comp_id][exp_id] = None
                continue

            # Best medal across seeds
            best_medal = max(medal_level(r) for r in reports_for_comp)
            medal_matrix[comp_id][exp_id] = best_medal

            # Any medal across seeds
            has_medal = any(
                r.get("gold_medal") or r.get("silver_medal") or r.get("bronze_medal")
                for r in reports_for_comp
            )
            any_medal_matrix[comp_id][exp_id] = 1 if has_medal else 0

            # Above median across seeds
            has_above_median = any(r.get("above_median") for r in reports_for_comp)
            above_median_matrix[comp_id][exp_id] = (
                1 if has_above_median else 0
            )

            # Best raw score across seeds (considering is_lower_better)
            valid_scores = [
                r
                for r in reports_for_comp
                if r.get("valid_submission") and r.get("score") is not None
            ]
            if valid_scores:
                is_lower_better = valid_scores[0].get("is_lower_better", False)
                if is_lower_better:
                    best_score = min(r["score"] for r in valid_scores)
                else:
                    best_score = max(r["score"] for r in valid_scores)
                score_matrix[comp_id][exp_id] = best_score
            else:
                score_matrix[comp_id][exp_id] = None

            # Store seed details
            for r in reports_for_comp:
                seed_details[exp_id][comp_id].append(
                    {
                        "score": r.get("score"),
                        "medal": medal_name(medal_level(r)),
                        "valid": r.get("valid_submission", False),
                        "submitted": r.get("submission_exists", False),
                    }
                )

    # -- Step 4: Write response matrices -------------------------------------
    def write_matrix(matrix, filename, value_formatter=None):
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["competition_id"] + experiment_ids)
            for comp_id in competition_ids:
                row = [comp_id]
                for exp_id in experiment_ids:
                    val = matrix[comp_id].get(exp_id)
                    if val is None:
                        row.append("")
                    elif value_formatter:
                        row.append(value_formatter(val))
                    else:
                        row.append(val)
                writer.writerow(row)
        print(f"\nWritten: {filepath}")
        return filepath

    # Main medal-level matrix
    write_matrix(medal_matrix, "response_matrix.csv")

    # Binary any-medal matrix
    write_matrix(any_medal_matrix, "response_matrix_binary.csv")

    # Above-median matrix
    write_matrix(above_median_matrix, "response_matrix_above_median.csv")

    # Raw score matrix
    def fmt_score(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return v

    write_matrix(
        score_matrix, "response_matrix_scores.csv", value_formatter=fmt_score
    )

    # -- Step 5: Write task metadata -----------------------------------------
    task_meta = load_task_metadata()
    meta_path = OUTPUT_DIR / "task_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "competition_id",
                "category",
                "dataset_size_GB",
                "complexity",
                "in_split75",
            ]
        )
        for comp_id in competition_ids:
            m = task_meta.get(comp_id, {})
            writer.writerow(
                [
                    comp_id,
                    m.get("category", ""),
                    m.get("dataset_size_GB", ""),
                    m.get("complexity", ""),
                    "yes",
                ]
            )
    print(f"Written: {meta_path}")

    # -- Step 6: Write experiment metadata -----------------------------------
    exp_meta_path = OUTPUT_DIR / "experiment_metadata.csv"
    with open(exp_meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["experiment_id", "description", "num_run_groups", "run_groups"]
        )
        for exp_id in experiment_ids:
            groups = exp_to_groups[exp_id]
            desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, "")
            writer.writerow(
                [exp_id, desc, len(groups), ";".join(sorted(groups))]
            )
    print(f"Written: {exp_meta_path}")

    # -- Step 7: Compute and write summary statistics ------------------------
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("MLE-bench Response Matrix Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(
        f"\nDimensions: {len(competition_ids)} tasks x "
        f"{len(experiment_ids)} experiments"
    )
    summary_lines.append(
        f"\nTasks (rows): {len(competition_ids)} Kaggle competitions"
    )
    summary_lines.append(
        f"Experiments (columns): {len(experiment_ids)} agent/model configs"
    )

    # Fill rate
    total_cells = len(competition_ids) * len(experiment_ids)
    filled_cells = sum(
        1
        for comp_id in competition_ids
        for exp_id in experiment_ids
        if medal_matrix[comp_id].get(exp_id) is not None
    )
    fill_rate = filled_cells / total_cells * 100
    summary_lines.append(
        f"\nFill rate: {filled_cells}/{total_cells} = {fill_rate:.1f}%"
    )

    # Score types
    summary_lines.append("\nScore types in response_matrix.csv:")
    summary_lines.append(
        "  3 = gold medal, 2 = silver, 1 = bronze, "
        "0.5 = above median, 0 = below median/no valid submission"
    )
    summary_lines.append(
        "  Empty = experiment did not attempt this competition"
    )

    # Per-experiment summary
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("Per-experiment summary (best across seeds):")
    summary_lines.append("-" * 80)
    summary_lines.append(
        f"{'Experiment':<45} {'Medal%':>7} {'Med%':>7} "
        f"{'Gold':>5} {'Silv':>5} {'Brnz':>5} {'Tasks':>5}"
    )
    summary_lines.append("-" * 80)

    for exp_id in experiment_ids:
        n_tasks = sum(
            1
            for c in competition_ids
            if medal_matrix[c].get(exp_id) is not None
        )
        if n_tasks == 0:
            continue
        n_gold = sum(
            1
            for c in competition_ids
            if medal_matrix[c].get(exp_id) == 3
        )
        n_silver = sum(
            1
            for c in competition_ids
            if medal_matrix[c].get(exp_id) == 2
        )
        n_bronze = sum(
            1
            for c in competition_ids
            if medal_matrix[c].get(exp_id) == 1
        )
        n_any_medal = n_gold + n_silver + n_bronze
        n_above_median = sum(
            1
            for c in competition_ids
            if above_median_matrix[c].get(exp_id) == 1
        )
        medal_pct = n_any_medal / n_tasks * 100
        median_pct = n_above_median / n_tasks * 100
        summary_lines.append(
            f"{exp_id:<45} {medal_pct:>6.1f}% {median_pct:>6.1f}% "
            f"{n_gold:>5} {n_silver:>5} {n_bronze:>5} {n_tasks:>5}"
        )

    # Per-task difficulty
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append(
        "Per-task difficulty (how many experiments achieved medal):"
    )
    summary_lines.append("-" * 80)

    task_difficulty = []
    for comp_id in competition_ids:
        n_exp = sum(
            1
            for exp_id in experiment_ids
            if medal_matrix[comp_id].get(exp_id) is not None
        )
        n_medals = sum(
            1
            for exp_id in experiment_ids
            if any_medal_matrix[comp_id].get(exp_id) == 1
        )
        pct = (n_medals / n_exp * 100) if n_exp > 0 else 0
        task_difficulty.append((comp_id, n_medals, n_exp, pct))

    # Sort by medal percentage (easiest first)
    task_difficulty.sort(key=lambda x: -x[3])
    summary_lines.append(
        f"{'Competition':<55} {'Medals':>7} {'Tested':>7} {'%':>7}"
    )
    for comp_id, n_medals, n_exp, pct in task_difficulty:
        summary_lines.append(
            f"{comp_id:<55} {n_medals:>7} {n_exp:>7} {pct:>6.1f}%"
        )

    # Model names
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("Experiment/Model configurations:")
    summary_lines.append("-" * 80)
    for exp_id in experiment_ids:
        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, "No description")
        n_groups = len(exp_to_groups[exp_id])
        summary_lines.append(f"  {exp_id}")
        summary_lines.append(f"    Description: {desc}")
        summary_lines.append(f"    Run groups (seeds): {n_groups}")

    # Complexity breakdown
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("Task complexity breakdown:")
    summary_lines.append("-" * 80)
    complexity_counts = defaultdict(int)
    for comp_id in competition_ids:
        m = task_meta.get(comp_id, {})
        complexity_counts[m.get("complexity", "Unknown")] += 1
    for k, v in sorted(complexity_counts.items()):
        summary_lines.append(f"  {k}: {v} tasks")

    summary_text = "\n".join(summary_lines)
    stats_path = OUTPUT_DIR / "summary_stats.txt"
    with open(stats_path, "w") as f:
        f.write(summary_text)
    print(f"\nWritten: {stats_path}")
    print("\n" + summary_text)

    # -- Step 8: Write per-seed detailed report ------------------------------
    detail_path = OUTPUT_DIR / "per_seed_details.json"
    details_regular = {
        exp_id: dict(comps) for exp_id, comps in seed_details.items()
    }
    with open(detail_path, "w") as f:
        json.dump(details_regular, f, indent=2)
    print(f"\nWritten: {detail_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
