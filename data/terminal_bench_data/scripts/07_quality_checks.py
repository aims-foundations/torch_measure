"""
Data quality checks for the Terminal-Bench response matrix.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
OUT_DIR = str(_BENCHMARK_DIR / "processed")


def main():
    # Load data
    trials = pd.read_csv(f"{RAW_DIR}/db_individual_trials.csv")
    per_task = pd.read_csv(f"{RAW_DIR}/db_per_task_results.csv")
    rate_matrix = pd.read_csv(f"{OUT_DIR}/resolution_rate_matrix.csv", index_col=0)
    count_matrix = pd.read_csv(f"{OUT_DIR}/trials_count_matrix.csv", index_col=0)
    tasks_meta = pd.read_csv(f"{OUT_DIR}/all_tasks_metadata.csv")

    trials["agent_model"] = trials["agent_name"] + " | " + trials["model_name"]

    report = []
    report.append("=" * 80)
    report.append("TERMINAL-BENCH DATA QUALITY REPORT")
    report.append("=" * 80)

    # 1. Coverage check
    report.append("\n## 1. Coverage Analysis")
    n_full = (count_matrix.notna().sum(axis=1) == 89).sum()
    n_partial = (count_matrix.notna().sum(axis=1) < 89).sum()
    report.append(f"Agent-model combos with full 89-task coverage: {n_full}")
    report.append(f"Agent-model combos with partial coverage: {n_partial}")

    if n_partial > 0:
        partial = count_matrix[count_matrix.notna().sum(axis=1) < 89]
        for idx in partial.index:
            n_tasks = partial.loc[idx].notna().sum()
            missing = list(partial.columns[partial.loc[idx].isna()])
            report.append(f"  {idx}: {n_tasks}/89 tasks "
                          f"(missing: {', '.join(missing[:5])}...)")

    # 2. Missing entries in the matrix
    report.append("\n## 2. Missing Entries")
    total_cells = rate_matrix.shape[0] * rate_matrix.shape[1]
    missing_cells = rate_matrix.isna().sum().sum()
    report.append(f"Matrix size: {rate_matrix.shape[0]} x {rate_matrix.shape[1]} = "
                  f"{total_cells} cells")
    report.append(f"Missing cells: {int(missing_cells)} ({missing_cells/total_cells:.1%})")

    # Tasks with most missing data
    missing_per_task = rate_matrix.isna().sum()
    if missing_per_task.max() > 0:
        report.append(f"\nTasks with missing data:")
        for task in missing_per_task[missing_per_task > 0].sort_values(
            ascending=False
        ).head(10).index:
            report.append(f"  {task}: {missing_per_task[task]} combos missing")

    # 3. Trial count consistency
    report.append("\n## 3. Trial Count Consistency")
    counts = count_matrix.values[count_matrix.notna().values]
    report.append(f"Trials per (agent-model, task):")
    report.append(f"  Min: {np.nanmin(counts):.0f}")
    report.append(f"  Max: {np.nanmax(counts):.0f}")
    report.append(f"  Mean: {np.nanmean(counts):.1f}")
    report.append(f"  Median: {np.nanmedian(counts):.0f}")
    report.append(f"  Std: {np.nanstd(counts):.1f}")

    # Distribution of trial counts
    from collections import Counter
    count_dist = Counter(counts.astype(int))
    report.append(f"\n  Distribution of trial counts:")
    for n, freq in sorted(count_dist.items()):
        report.append(f"    {n} trials: {freq} cells ({freq/len(counts):.1%})")

    # 4. Non-determinism analysis
    report.append("\n## 4. Non-Determinism / Variance Analysis")
    # For agent-model × task combos with 5+ trials, check variance
    multi_trial = per_task[per_task["n_trials"] >= 5].copy()
    multi_trial["variance"] = multi_trial["p_hat"] * (1 - multi_trial["p_hat"])

    report.append(f"Cells with 5+ trials: {len(multi_trial)}")
    report.append(f"Mean p_hat: {multi_trial['p_hat'].mean():.3f}")
    report.append(f"Mean variance (p*(1-p)): {multi_trial['variance'].mean():.3f}")

    # Cells that are neither always pass nor always fail
    neither = multi_trial[(multi_trial["p_hat"] > 0) & (multi_trial["p_hat"] < 1)]
    report.append(f"\nCells with mixed outcomes (0 < p_hat < 1): {len(neither)} "
                  f"({len(neither)/len(multi_trial):.1%})")
    always_pass = (multi_trial["p_hat"] == 1).sum()
    always_fail = (multi_trial["p_hat"] == 0).sum()
    report.append(f"Always pass (p_hat = 1): {always_pass} "
                  f"({always_pass/len(multi_trial):.1%})")
    report.append(f"Always fail (p_hat = 0): {always_fail} "
                  f"({always_fail/len(multi_trial):.1%})")

    # Highest variance cells
    report.append(f"\nHighest variance cells (most non-deterministic):")
    high_var = neither.nlargest(10, "variance")
    for _, row in high_var.iterrows():
        am = f"{row['agent_name']} | {row['model_name']}"
        report.append(f"  {am:50s} {row['task_name']:35s} "
                      f"p={row['p_hat']:.2f} n={int(row['n_trials'])}")

    # 5. Error rate analysis
    report.append("\n## 5. Error Rate Analysis")
    total_errors = per_task["n_errors"].sum()
    total_trials_count = per_task["n_trials"].sum()
    report.append(f"Total errors: {int(total_errors)} / {int(total_trials_count)} trials "
                  f"({total_errors/total_trials_count:.1%})")

    # Error rate by agent
    agent_errors = per_task.groupby("agent_name").agg(
        errors=("n_errors", "sum"),
        trials=("n_trials", "sum"),
    ).reset_index()
    agent_errors["error_rate"] = agent_errors["errors"] / agent_errors["trials"]
    agent_errors = agent_errors.sort_values("error_rate", ascending=False)
    report.append(f"\nError rate by agent:")
    for _, row in agent_errors.head(10).iterrows():
        report.append(f"  {row['agent_name']:25s} "
                      f"{row['error_rate']:.1%} "
                      f"({int(row['errors'])}/{int(row['trials'])})")

    # 6. Task metadata alignment check
    report.append("\n## 6. Task Metadata Alignment")
    db_tasks = set(rate_matrix.columns)
    hf_tasks = set(tasks_meta["task_id"])
    report.append(f"Tasks in DB (TB 2.0): {len(db_tasks)}")
    report.append(f"Tasks in HuggingFace: {len(hf_tasks)}")
    in_db_not_hf = db_tasks - hf_tasks
    in_hf_not_db = hf_tasks - db_tasks
    overlap = db_tasks & hf_tasks
    report.append(f"Overlap: {len(overlap)}")
    if in_db_not_hf:
        report.append(f"In DB but not HuggingFace ({len(in_db_not_hf)}):")
        for t in sorted(in_db_not_hf):
            report.append(f"  - {t}")
    if in_hf_not_db:
        report.append(f"In HuggingFace but not DB ({len(in_hf_not_db)}):")
        for t in sorted(in_hf_not_db):
            report.append(f"  - {t}")

    # 7. Duplicate model name check
    report.append("\n## 7. Potential Duplicate Model Names")
    per_task["model_lower"] = per_task["model_name"].str.lower().str.replace(
        r"[-_/]", "", regex=True
    )
    model_groups = per_task.groupby("model_lower")["model_name"].apply(
        lambda x: list(x.unique())
    )
    dupes = model_groups[model_groups.apply(len) > 1]
    if len(dupes) > 0:
        report.append(f"Found {len(dupes)} potential duplicate model name groups:")
        for name, variants in dupes.items():
            report.append(f"  {variants}")
    else:
        report.append("No duplicate model names detected.")

    # Print and save report
    report_text = "\n".join(report)
    print(report_text)

    with open(f"{OUT_DIR}/quality_report.txt", "w") as f:
        f.write(report_text)
    print(f"\nSaved quality_report.txt")


if __name__ == "__main__":
    main()
