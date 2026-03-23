"""
Build intervention matrices from METR Early-2025 Developer Productivity RCT.

Source: https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs
Paper: https://arxiv.org/abs/2507.09089

Structure: 16 developers x 246 issues x {AI allowed, AI disallowed} -> completion time
Design: Within-subjects, task-level randomization. Each developer does tasks in both conditions.
Items are developer-specific (not shared across developers).

Output:
  - intervention_table.csv: long-format with all columns
  - response_matrix_ai.csv: developer x issue matrix (AI-allowed tasks only)
  - response_matrix_no_ai.csv: developer x issue matrix (AI-disallowed tasks only)
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    csv_path = RAW_DIR / "repo" / "data_complete.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run 01_download_raw.sh first. Missing: {csv_path}")

    print("Loading METR Early-2025 data...")
    df = pd.read_csv(csv_path)

    print(f"Raw data: {len(df)} rows, columns: {list(df.columns)}")
    print(f"Developers: {df['dev_id'].nunique()}")
    print(f"Tasks: {len(df)}")
    print(f"AI-allowed: {df['ai_treatment'].sum()}, AI-disallowed: {(~df['ai_treatment'].astype(bool)).sum()}")

    # Save full intervention table
    df.to_csv(OUTPUT_DIR / "intervention_table.csv", index=False)
    print(f"Saved intervention_table.csv ({len(df)} rows)")

    # Build response matrices per condition
    # Use initial_implementation_time as the primary outcome
    time_col = "initial_implementation_time"
    if time_col not in df.columns:
        # Try alternative column names
        time_cols = [c for c in df.columns if "time" in c.lower() and "predicted" not in c.lower()]
        if time_cols:
            time_col = time_cols[0]
            print(f"Using {time_col} as outcome")
        else:
            print(f"Warning: no implementation time column found. Columns: {list(df.columns)}")
            return

    # Create a unique task ID combining dev_id and issue index
    df["task_id"] = df.apply(lambda r: f"dev{r['dev_id']}_issue{r.name}", axis=1)

    # AI-allowed matrix
    ai_df = df[df["ai_treatment"] == 1]
    ai_matrix = ai_df.pivot_table(index="dev_id", columns="task_id", values=time_col)
    ai_matrix.to_csv(OUTPUT_DIR / "response_matrix_ai.csv")
    print(f"Saved response_matrix_ai.csv: {ai_matrix.shape[0]} devs x {ai_matrix.shape[1]} tasks")

    # AI-disallowed matrix
    no_ai_df = df[df["ai_treatment"] == 0]
    no_ai_matrix = no_ai_df.pivot_table(index="dev_id", columns="task_id", values=time_col)
    no_ai_matrix.to_csv(OUTPUT_DIR / "response_matrix_no_ai.csv")
    print(f"Saved response_matrix_no_ai.csv: {no_ai_matrix.shape[0]} devs x {no_ai_matrix.shape[1]} tasks")

    # Summary
    print(f"\nSummary:")
    print(f"  Developers: {df['dev_id'].nunique()}")
    print(f"  Total tasks: {len(df)}")
    print(f"  AI-allowed tasks: {len(ai_df)}")
    print(f"  AI-disallowed tasks: {len(no_ai_df)}")
    print(f"  Mean time (AI): {ai_df[time_col].mean():.1f} min")
    print(f"  Mean time (no AI): {no_ai_df[time_col].mean():.1f} min")
    print(f"  Tasks per developer: {df.groupby('dev_id').size().describe().to_dict()}")


if __name__ == "__main__":
    main()
