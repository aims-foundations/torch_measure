"""
02_build_response_matrix.py — Sycophancy to Subterfuge dataset exploration and processing.

Loads from raw/sycophancy-to-subterfuge-paper (git clone of anthropics/sycophancy-to-subterfuge-paper).
Expected structure:
  - prompts.json — prompt definitions for each evaluation stage
  - environments.json — environment configurations
  - samples/ — JSONL.GZ files with model outputs for each stage:
    - political_sycophancy.jsonl.gz
    - tool_use_flattery.jsonl.gz
    - nudged_rubric_modification.jsonl.gz
    - insubordinate_rubric_modification.jsonl.gz
    - reward_tampering.jsonl.gz
Shows escalation from sycophancy to reward tampering.
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def list_raw_contents():
    """Recursively list files in raw/ (excluding .git), print summary."""
    print("=" * 60)
    print("FILES IN raw/")
    print("=" * 60)
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
            all_files.append(rel)
    for f in sorted(all_files)[:50]:
        print(f"  {f}")
    if len(all_files) > 50:
        print(f"  ... and {len(all_files) - 50} more files")
    print(f"\nTotal files: {len(all_files)}")
    return all_files


def load_jsonl_gz(fpath, max_lines=None):
    """Load a gzipped JSONL file."""
    records = []
    try:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
    return records


def load_jsonl(fpath, max_lines=None):
    """Load a JSONL file (plain text)."""
    records = []
    try:
        with open(fpath) as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
    return records


def main():
    print("Sycophancy to Subterfuge Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "sycophancy-to-subterfuge-paper"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "sycophancy" in candidate.name.lower():
                repo_dir = candidate
                break

    if not repo_dir.exists():
        print(f"\nERROR: Repo not found at {repo_dir}")
        return

    print(f"\nRepo directory: {repo_dir}")
    print(f"  Contents: {sorted(os.listdir(repo_dir))}")

    # Load prompts.json to understand the evaluation stages
    print("\n" + "=" * 60)
    print("EVALUATION PROMPTS")
    print("=" * 60)

    prompts_file = repo_dir / "prompts.json"
    prompts_data = None
    if prompts_file.exists():
        try:
            with open(prompts_file) as f:
                prompts_data = json.load(f)
            if isinstance(prompts_data, dict):
                print(f"Prompts: dict with {len(prompts_data)} keys")
                for k, v in prompts_data.items():
                    vtype = type(v).__name__
                    if isinstance(v, str):
                        print(f"  {k}: {v[:200]}...")
                    elif isinstance(v, list):
                        print(f"  {k}: list of {len(v)} items")
                    elif isinstance(v, dict):
                        print(f"  {k}: dict with keys {list(v.keys())[:5]}")
                    else:
                        print(f"  {k} ({vtype}): {str(v)[:200]}")
            elif isinstance(prompts_data, list):
                print(f"Prompts: list of {len(prompts_data)} items")
                for item in prompts_data[:5]:
                    print(f"  {str(item)[:200]}")
        except Exception as e:
            print(f"Error loading prompts.json: {e}")
    else:
        print("prompts.json not found")

    # Load environments.json
    print("\n" + "=" * 60)
    print("ENVIRONMENTS")
    print("=" * 60)

    envs_file = repo_dir / "environments.json"
    if envs_file.exists():
        try:
            with open(envs_file) as f:
                envs_data = json.load(f)
            if isinstance(envs_data, dict):
                print(f"Environments: dict with {len(envs_data)} keys")
                for k, v in list(envs_data.items())[:10]:
                    print(f"  {k}: {str(v)[:200]}")
            elif isinstance(envs_data, list):
                print(f"Environments: list of {len(envs_data)} items")
                for item in envs_data[:3]:
                    print(f"  {str(item)[:200]}")
        except Exception as e:
            print(f"Error loading environments.json: {e}")

    # Load sample files (JSONL.GZ) — these are the key data
    print("\n" + "=" * 60)
    print("SAMPLE DATA (STAGE-LEVEL)")
    print("=" * 60)

    samples_dir = repo_dir / "samples"
    stage_order = [
        "political_sycophancy",
        "tool_use_flattery",
        "nudged_rubric_modification",
        "insubordinate_rubric_modification",
        "reward_tampering",
    ]

    stage_summaries = []

    if samples_dir.exists():
        sample_files = sorted(samples_dir.glob("*"))
        print(f"Sample files: {[f.name for f in sample_files]}")

        for sf in sample_files:
            fname = sf.name
            print(f"\n--- {fname} ---")
            print(f"  Size: {sf.stat().st_size / 1024:.1f} KB")

            # Load data
            records = []
            if fname.endswith(".jsonl.gz"):
                records = load_jsonl_gz(sf)
            elif fname.endswith(".jsonl"):
                records = load_jsonl(sf)
            elif fname.endswith(".md"):
                try:
                    with open(sf) as f:
                        content = f.read()
                    print(f"  Markdown content ({len(content)} chars):")
                    print(f"  {content[:500]}...")
                except Exception as e:
                    print(f"  Error: {e}")
                continue
            else:
                print(f"  Skipping non-data file")
                continue

            if not records:
                print("  No records loaded")
                continue

            print(f"  Records: {len(records)}")

            # Examine structure
            sample = records[0]
            print(f"  Keys: {list(sample.keys())}")
            for k, v in sample.items():
                vstr = str(v)[:150]
                print(f"    {k} ({type(v).__name__}): {vstr}")

            # Convert to DataFrame
            try:
                df = pd.DataFrame(records)
                print(f"  DataFrame: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Dtypes:\n{df.dtypes.to_string()}")

                # Analyze label/outcome columns
                label_keywords = ["label", "result", "score", "sycophant", "outcome",
                                  "tamper", "correct", "answer", "choice", "action"]
                label_cols = [c for c in df.columns if any(kw in c.lower() for kw in label_keywords)]
                print(f"  Label columns: {label_cols}")

                for col in label_cols:
                    nunique = df[col].nunique()
                    if nunique <= 30:
                        vc = df[col].value_counts()
                        print(f"\n  {col} distribution:")
                        print(f"{vc.to_string()}")

                # All columns analysis
                for col in df.columns:
                    if col not in label_cols:
                        nunique = df[col].nunique()
                        if nunique <= 15 and df[col].dtype in ("object", "bool", "int64"):
                            vc = df[col].value_counts().head(10)
                            print(f"\n  {col} (nunique={nunique}):")
                            print(f"{vc.to_string()}")

                # Stage name from filename
                stage_name = sf.stem.replace(".jsonl", "")
                stage_summaries.append({
                    "stage": stage_name,
                    "n_records": len(records),
                    "n_columns": len(df.columns),
                    "columns": "|".join(df.columns),
                })

                # Save stage data
                out_name = f"sycophancy_{stage_name}.csv"
                # Truncate long text columns
                df_save = df.copy()
                for col in df_save.select_dtypes(include=["object"]).columns:
                    max_len = df_save[col].str.len().max()
                    if max_len and max_len > 500:
                        df_save[col] = df_save[col].str[:500]
                df_save.to_csv(PROCESSED_DIR / out_name, index=False)
                print(f"  -> Saved to processed/{out_name}")

            except Exception as e:
                print(f"  Error creating DataFrame: {e}")
                stage_summaries.append({
                    "stage": sf.stem.replace(".jsonl", ""),
                    "n_records": len(records),
                    "n_columns": len(records[0]) if records else 0,
                    "columns": "|".join(records[0].keys()) if records else "",
                })
    else:
        print("  samples/ directory not found")

    # Escalation summary: show the progression from mild to severe
    print("\n" + "=" * 60)
    print("ESCALATION LADDER SUMMARY")
    print("=" * 60)

    if stage_summaries:
        stages_df = pd.DataFrame(stage_summaries)
        # Reorder by escalation if possible
        stage_order_map = {s: i for i, s in enumerate(stage_order)}
        stages_df["order"] = stages_df["stage"].map(
            lambda x: min(
                (stage_order_map[s] for s in stage_order if s in x),
                default=99,
            )
        )
        stages_df = stages_df.sort_values("order").drop(columns=["order"])
        print(stages_df[["stage", "n_records", "n_columns"]].to_string(index=False))
        stages_df.to_csv(PROCESSED_DIR / "stage_catalog.csv", index=False)
        print(f"\n  -> Saved to processed/stage_catalog.csv")

    # Overall summary
    summary_rows = [
        {"metric": "n_stages", "value": len(stage_summaries)},
        {"metric": "total_records", "value": sum(s["n_records"] for s in stage_summaries)},
        {"metric": "has_prompts", "value": int(prompts_data is not None)},
        {"metric": "has_environments", "value": int(envs_file.exists())},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
