"""
Process and organize task metadata from HuggingFace.
Identifies the 89 Terminal-Bench 2.0 tasks and extracts structured metadata.
"""

import json
import yaml
import pandas as pd
from datasets import load_dataset
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")
RAW_DIR = str(_BENCHMARK_DIR / "raw")


def main():
    # Load from HuggingFace
    print("Loading Terminal-Bench dataset...")
    ds = load_dataset("ia03/terminal-bench", split="test")
    print(f"Total tasks in HuggingFace: {len(ds)}")

    records = []
    for row in ds:
        task_id = row.get("task_id", "")

        # Parse task_yaml for additional info
        task_yaml_str = row.get("task_yaml", "")
        task_yaml = {}
        if task_yaml_str:
            try:
                task_yaml = yaml.safe_load(task_yaml_str)
            except Exception:
                pass

        # Extract description - prefer base_description, fallback to task_yaml
        description = row.get("base_description", "")
        if not description and task_yaml:
            description = task_yaml.get("instruction", "")

        record = {
            "task_id": task_id,
            "category": row.get("category", ""),
            "difficulty": row.get("difficulty", ""),
            "description": description,
            "description_length": len(description) if description else 0,
            "tags": row.get("tags", []),
            "max_agent_timeout_sec": row.get("max_agent_timeout_sec"),
            "max_test_timeout_sec": row.get("max_test_timeout_sec"),
            "archive_bytes": row.get("archive_bytes"),
            "n_files": row.get("n_files"),
        }

        # Extract additional info from task_yaml
        if task_yaml:
            env = task_yaml.get("environment", {})
            record["docker_image"] = env.get("docker_image", "")
            record["cpus"] = env.get("cpus", "")
            record["memory"] = env.get("memory", "")
            record["storage"] = env.get("storage", "")
            record["author"] = task_yaml.get("author", "")

        records.append(record)

    df = pd.DataFrame(records)

    # Check for variant tasks (e.g., .easy, .hard suffixes)
    df["is_variant"] = df["task_id"].str.contains(r"\.(easy|hard|v\d)$", regex=True)
    df["base_task_id"] = df["task_id"].str.replace(
        r"\.(easy|hard|v\d)$", "", regex=True
    )

    print(f"\n=== Task Variant Analysis ===")
    print(f"Total tasks: {len(df)}")
    print(f"Base tasks (no variant suffix): {(~df['is_variant']).sum()}")
    print(f"Variant tasks: {df['is_variant'].sum()}")
    if df["is_variant"].any():
        print("Variants:")
        for _, row in df[df["is_variant"]].iterrows():
            print(f"  {row['task_id']} (base: {row['base_task_id']})")

    # Save all tasks
    # For CSV, convert tags list to string
    df_csv = df.copy()
    df_csv["tags"] = df_csv["tags"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    df_csv.to_csv(f"{OUTPUT_DIR}/all_tasks_metadata.csv", index=False)
    print(f"\nSaved all_tasks_metadata.csv ({len(df)} tasks)")

    # Save as JSON (preserves lists)
    with open(f"{OUTPUT_DIR}/all_tasks_metadata.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved all_tasks_metadata.json")

    # Summary statistics
    print(f"\n=== Category Distribution ===")
    print(df["category"].value_counts().to_string())

    print(f"\n=== Difficulty Distribution ===")
    print(df["difficulty"].value_counts().to_string())

    print(f"\n=== Timeout Stats ===")
    print(f"Agent timeout (sec): min={df['max_agent_timeout_sec'].min()}, "
          f"max={df['max_agent_timeout_sec'].max()}, "
          f"median={df['max_agent_timeout_sec'].median()}")
    print(f"Test timeout (sec): min={df['max_test_timeout_sec'].min()}, "
          f"max={df['max_test_timeout_sec'].max()}, "
          f"median={df['max_test_timeout_sec'].median()}")

    # Save a concise task list for quick reference
    task_list = df[["task_id", "category", "difficulty", "description_length"]].copy()
    task_list = task_list.sort_values(["category", "task_id"])
    task_list.to_csv(f"{OUTPUT_DIR}/task_list_summary.csv", index=False)
    print(f"\nSaved task_list_summary.csv")

    # Print task list grouped by category
    print(f"\n=== All Tasks by Category ===")
    for cat in sorted(df["category"].unique()):
        tasks = df[df["category"] == cat].sort_values("task_id")
        print(f"\n{cat} ({len(tasks)} tasks):")
        for _, row in tasks.iterrows():
            diff = row["difficulty"]
            print(f"  [{diff:6s}] {row['task_id']}")


if __name__ == "__main__":
    main()
