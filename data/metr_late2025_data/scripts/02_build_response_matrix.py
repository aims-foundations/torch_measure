"""
Build intervention matrices from METR Late-2025 Developer Productivity RCT.

Source: https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs
Expanded study: 57 developers, ~1,134 task records

Structure: 57 developers x ~1,134 issues x {AI allowed, AI disallowed} -> completion time
Design: Within-subjects, task-level randomization.
Note: METR warns this data has selection effects (developers refusing no-AI conditions).

Output:
  - intervention_table.csv: long-format with all columns
  - response_matrix_ai.csv: developer x issue matrix (AI-allowed, completed tasks)
  - response_matrix_no_ai.csv: developer x issue matrix (AI-disallowed, completed tasks)
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    csv_path = RAW_DIR / "repo" / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run 01_download_raw.sh first. Missing: {csv_path}")

    print("Loading METR Late-2025 data...")
    df = pd.read_csv(csv_path)

    print(f"Raw data: {len(df)} rows, columns: {list(df.columns)}")

    # Save full intervention table
    df.to_csv(OUTPUT_DIR / "intervention_table.csv", index=False)
    print(f"Saved intervention_table.csv ({len(df)} rows)")

    # Identify key columns (may differ from early study)
    print("\nColumn names:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}, nunique={df[col].nunique()}, null={df[col].isnull().sum()}")

    # Filter to completed tasks only
    if "status" in df.columns:
        status_counts = df["status"].value_counts()
        print(f"\nTask status distribution:\n{status_counts}")
        completed = df[df["status"] == "completed"] if "completed" in df["status"].values else df
        print(f"Completed tasks: {len(completed)}")
    else:
        completed = df

    # Detect column names
    dev_col = next((c for c in df.columns if "developer" in c.lower() or "dev" in c.lower()), None)
    treatment_col = next((c for c in df.columns if "treatment" in c.lower()), None)
    time_col = next(
        (c for c in df.columns if "implementation" in c.lower() and "time" in c.lower() and "predicted" not in c.lower() and "estimated" not in c.lower()),
        None,
    )
    id_col = next((c for c in df.columns if c.lower() == "id"), None)

    print(f"\nDetected columns: dev={dev_col}, treatment={treatment_col}, time={time_col}, id={id_col}")

    if not all([dev_col, treatment_col, time_col]):
        print("Could not auto-detect all required columns. Please inspect intervention_table.csv and adapt.")
        return

    # Build matrices per condition
    task_id_col = id_col if id_col else "task_idx"
    if id_col is None:
        completed = completed.copy()
        completed[task_id_col] = range(len(completed))

    treatments = completed[treatment_col].unique()
    print(f"Treatment values: {treatments}")

    for treat in treatments:
        subset = completed[completed[treatment_col] == treat]
        matrix = subset.pivot_table(index=dev_col, columns=task_id_col, values=time_col)
        safe_name = str(treat).replace(" ", "_").replace("/", "_").lower()
        matrix.to_csv(OUTPUT_DIR / f"response_matrix_{safe_name}.csv")
        print(f"Saved response_matrix_{safe_name}.csv: {matrix.shape[0]} devs x {matrix.shape[1]} tasks")

    # Summary
    n_devs = completed[dev_col].nunique()
    print(f"\nSummary:")
    print(f"  Developers: {n_devs}")
    print(f"  Completed tasks: {len(completed)}")
    for treat in treatments:
        subset = completed[completed[treatment_col] == treat]
        print(f"  {treat}: {len(subset)} tasks, mean time = {subset[time_col].mean():.1f} min")


if __name__ == "__main__":
    main()
