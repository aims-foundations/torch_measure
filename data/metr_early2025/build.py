"""
01_build_response_matrix.py — Download and build intervention matrices from METR Early-2025
Developer Productivity RCT.

Downloads from: https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs
Paper: https://arxiv.org/abs/2507.09089

Structure: 16 developers x 246 issues x {AI allowed, AI disallowed} -> completion time
Design: Within-subjects, task-level randomization. Each developer does tasks in both conditions.
Items are developer-specific (not shared across developers).

Output:
  - intervention_table.csv: long-format with all columns
  - response_matrix_ai.csv: developer x issue matrix (AI-allowed tasks only)
  - response_matrix_no_ai.csv: developer x issue matrix (AI-disallowed tasks only)
"""

import sys
import subprocess
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    repo_dir = RAW_DIR / "repo"

    if repo_dir.exists():
        print("repo already cloned, skipping")
        return

    print("Cloning METR Early-2025 study repo...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs.git", str(repo_dir)],
        check=True,
    )
    print(f"Done. Raw files in {repo_dir}")


def main():
    download()

    csv_path = RAW_DIR / "repo" / "data_complete.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

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

    # Build item_content.csv from available metadata
    print("\nBuilding item_content.csv...")
    rows = []
    for _, r in df.iterrows():
        task_id = f"dev{r['dev_id']}_issue{r.name}"
        parts = [f"GitHub issue #{int(r['issue_id'])} assigned to developer {int(r['dev_id'])}"]
        condition = "AI-allowed" if r["ai_treatment"] == 1 else "AI-disallowed"
        parts.append(f"condition: {condition}")
        if pd.notna(r.get("predicted_time_no_ai")):
            parts.append(f"estimated {int(r['predicted_time_no_ai'])}min without AI")
        if pd.notna(r.get("predicted_time_ai_allowed")):
            parts.append(f"{int(r['predicted_time_ai_allowed'])}min with AI")
        if pd.notna(r.get("prior_task_exposure_1_to_5")):
            parts.append(f"prior exposure: {int(r['prior_task_exposure_1_to_5'])}/5")
        if pd.notna(r.get("external_resource_needs_1_to_3")):
            parts.append(f"external resources: {int(r['external_resource_needs_1_to_3'])}/3")
        rows.append({"item_id": task_id, "content": "; ".join(parts)})
    item_content = pd.DataFrame(rows)
    item_content.to_csv(OUTPUT_DIR / "item_content.csv", index=False)
    print(f"Saved item_content.csv ({len(item_content)} items)")

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

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
