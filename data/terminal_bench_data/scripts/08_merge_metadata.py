"""
Merge task metadata from HuggingFace and DB, reconciling the different task sets.
Also enriches with empirical difficulty from the response matrix.
"""

import json
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
OUT_DIR = str(_BENCHMARK_DIR / "processed")


def main():
    # Load DB tasks (the actual TB 2.0 tasks used in evaluations)
    db_tasks = pd.read_csv(f"{RAW_DIR}/db_tasks.csv")
    print(f"DB tasks (TB 2.0 evaluated): {len(db_tasks)}")

    # Parse metadata column (may be dict from psycopg2 or JSON string from CSV)
    import ast
    def parse_metadata(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return {}
        return {}
    db_tasks["metadata_parsed"] = db_tasks["metadata"].apply(parse_metadata)

    # Extract fields from metadata
    db_tasks["category"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("category", "")
    )
    db_tasks["difficulty"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("difficulty", "")
    )
    db_tasks["tags"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("tags", [])
    )
    db_tasks["expert_time_min"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("expert_time_estimate_min")
    )
    db_tasks["junior_time_min"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("junior_time_estimate_min")
    )
    db_tasks["author"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("author", "")
    )

    # Load HuggingFace tasks
    hf_tasks = pd.read_csv(f"{OUT_DIR}/all_tasks_metadata.csv")
    print(f"HuggingFace tasks: {len(hf_tasks)}")

    # Load empirical difficulty
    emp_diff = pd.read_csv(f"{OUT_DIR}/task_empirical_difficulty.csv",
                           index_col=0)
    emp_diff.columns = ["empirical_avg_resolution_rate"]

    # Merge DB task info with empirical difficulty
    merged = db_tasks[["task_name", "category", "difficulty", "tags",
                       "expert_time_min", "junior_time_min",
                       "instruction", "author"]].copy()
    merged = merged.merge(
        emp_diff, left_on="task_name", right_index=True, how="left"
    )

    # Try to merge with HuggingFace for additional info
    hf_match = hf_tasks[["task_id", "max_agent_timeout_sec",
                          "max_test_timeout_sec"]].copy()
    merged = merged.merge(
        hf_match, left_on="task_name", right_on="task_id", how="left"
    ).drop(columns=["task_id"], errors="ignore")

    # Sort by empirical difficulty
    merged = merged.sort_values("empirical_avg_resolution_rate")

    # Compute empirical difficulty tier
    merged["empirical_difficulty"] = pd.cut(
        merged["empirical_avg_resolution_rate"],
        bins=[0, 0.1, 0.3, 0.5, 0.7, 1.01],
        labels=["very_hard", "hard", "medium", "easy", "very_easy"],
    )

    # Description length
    merged["description_length"] = merged["instruction"].str.len()

    # Save
    merged_csv = merged.copy()
    merged_csv["tags"] = merged_csv["tags"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    merged_csv.to_csv(f"{OUT_DIR}/tasks_complete_metadata.csv", index=False)

    # Also save as JSON (preserves lists)
    merged_json = merged.copy()
    merged_json["tags"] = merged_json["tags"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged_json.to_json(
        f"{OUT_DIR}/tasks_complete_metadata.json", orient="records", indent=2
    )

    print(f"\nSaved tasks_complete_metadata.csv and .json ({len(merged)} tasks)")

    # Print summary
    print(f"\n=== Task Metadata Summary (TB 2.0, {len(merged)} tasks) ===")
    print(f"\nCategory distribution:")
    print(merged["category"].value_counts().to_string())
    print(f"\nHuman difficulty distribution:")
    print(merged["difficulty"].value_counts().to_string())
    print(f"\nEmpirical difficulty distribution:")
    print(merged["empirical_difficulty"].value_counts().to_string())
    print(f"\nHuman vs Empirical difficulty cross-tab:")
    print(pd.crosstab(merged["difficulty"], merged["empirical_difficulty"]))

    print(f"\nDescription length: "
          f"min={merged['description_length'].min()}, "
          f"max={merged['description_length'].max()}, "
          f"median={merged['description_length'].median():.0f}")

    # Tasks with metadata from HuggingFace
    has_hf = merged["max_agent_timeout_sec"].notna().sum()
    print(f"\nTasks with HuggingFace metadata match: {has_hf}/{len(merged)}")


if __name__ == "__main__":
    main()
