"""
Build per-instance response matrices for SWE-PolyBench from submission branch data.

Data sources:
- Per-instance result JSON files from GitHub submission branch
  (evaluation/{PB,PBVerified}/<submission>/logs/<instance_id>_result.json)
- Instance metadata from HuggingFace datasets (language, repo, task_category, etc.)
- Leaderboard aggregate data from gh-pages HTML

Outputs:
- response_matrix_verified.csv: Binary response matrix for PBVerified (382 instances x 3 models)
- response_matrix_full.csv: Binary response matrix for PB Full (2110 instances x 1 model)
- instance_metadata_verified.csv: Instance metadata for PBVerified
- instance_metadata_full.csv: Instance metadata for PB Full
- leaderboard_aggregate.csv: Aggregate leaderboard data from HTML
- summary.json: Summary statistics

Models with per-instance results:
  PBVerified: PrometheusV1.2+GPT-5, Atlassian Rovo Dev, iSWE-Agent
  PB Full: iSWE-Agent (only 1 model)
"""

import json
import os
import glob
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
SUBMISSION_DIR = RAW_DIR / "swepolybench_submission" / "evaluation"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_instance_metadata(split_name):
    """Load instance metadata from HuggingFace dataset export."""
    path = RAW_DIR / f"{split_name}_instances.json"
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_result_files(logs_dir):
    """Load all *_result.json files from a submission's logs directory.

    Returns dict: instance_id -> resolved (bool)
    """
    results = {}
    for fpath in glob.glob(os.path.join(logs_dir, "*_result.json")):
        with open(fpath) as f:
            data = json.load(f)
        instance_id = data["instance_id"]
        resolved = data.get("resolved", False)
        results[instance_id] = int(bool(resolved))
    return results


def load_metrics_files(logs_dir):
    """Load all *_metrics.json files from a submission's logs directory.

    Returns dict: instance_id -> metrics dict
    """
    metrics = {}
    for fpath in glob.glob(os.path.join(logs_dir, "*_metrics.json")):
        with open(fpath) as f:
            data = json.load(f)
        instance_id = data["instance_id"]
        metrics[instance_id] = data
    return metrics


def load_leaderboard_data():
    """Parse leaderboard HTML to extract aggregate per-model data."""
    html_path = RAW_DIR / "leaderboard_index.html"
    with open(html_path) as f:
        html = f.read()

    tr_pattern = re.compile(r'<tr\s[^>]*?data-name="([^"]*)"(.*?)>', re.DOTALL)
    data_attr_pattern = re.compile(r'data-(\w+)="([^"]*)"')

    entries = []
    for match in tr_pattern.finditer(html):
        full_match = match.group(0)
        attrs = {}
        for attr_match in data_attr_pattern.finditer(full_match):
            key, val = attr_match.group(1), attr_match.group(2)
            attrs[key] = val

        line_num = html[:match.start()].count('\n') + 1
        if line_num < 1250:
            attrs['split'] = "Full"
        elif line_num < 2300:
            attrs['split'] = "PB500"
        else:
            attrs['split'] = "Verified"

        entries.append(attrs)

    return pd.DataFrame(entries)


def build_response_matrix(submissions, instance_ids, split_label):
    """Build a binary response matrix from submission result files.

    Args:
        submissions: list of (model_name, logs_dir) tuples
        instance_ids: list of instance IDs (canonical order)
        split_label: "full" or "verified"

    Returns:
        DataFrame with instance_id as index and model names as columns
    """
    matrix = pd.DataFrame(index=instance_ids)
    matrix.index.name = "instance_id"

    for model_name, logs_dir in submissions:
        results = load_result_files(logs_dir)
        # Map results to the canonical instance_id order
        col = []
        missing = 0
        for iid in instance_ids:
            if iid in results:
                col.append(results[iid])
            else:
                col.append(np.nan)
                missing += 1
        matrix[model_name] = col
        resolved = sum(1 for v in col if v == 1)
        total = sum(1 for v in col if not np.isnan(v))
        print(f"  {model_name}: {resolved}/{total} resolved"
              f" ({100*resolved/total:.2f}%)" if total > 0 else f"  {model_name}: no data",
              f"({missing} missing)" if missing > 0 else "")

    return matrix


def main():
    print("=" * 70)
    print("SWE-PolyBench Response Matrix Builder")
    print("=" * 70)

    # ---- 1. Load instance metadata ----
    print("\n--- Loading instance metadata ---")
    meta_full = load_instance_metadata("full")
    meta_verified = load_instance_metadata("verified")
    print(f"Full: {len(meta_full)} instances")
    print(f"Verified: {len(meta_verified)} instances")

    # ---- 2. Define submissions ----
    submissions_verified = [
        ("PrometheusV1.2+GPT-5",
         str(SUBMISSION_DIR / "PBVerified" / "20251130_prometheus_gpt-5" / "logs")),
        ("Atlassian_Rovo_Dev",
         str(SUBMISSION_DIR / "PBVerified" / "20251208_atlassian-rovo-dev" / "logs")),
        ("iSWE-Agent",
         str(SUBMISSION_DIR / "PBVerified" / "20260201_iswe_agent" / "logs")),
    ]

    submissions_full = [
        ("iSWE-Agent",
         str(SUBMISSION_DIR / "PB" / "20260201_iswe_agent" / "logs")),
    ]

    # ---- 3. Build response matrices ----
    print("\n--- Building PBVerified response matrix ---")
    verified_ids = meta_verified["instance_id"].tolist()
    rm_verified = build_response_matrix(submissions_verified, verified_ids, "verified")

    print("\n--- Building PB Full response matrix ---")
    full_ids = meta_full["instance_id"].tolist()
    rm_full = build_response_matrix(submissions_full, full_ids, "full")

    # ---- 4. Save response matrices ----
    print("\n--- Saving response matrices ---")
    rm_verified.to_csv(PROCESSED_DIR / "response_matrix_verified.csv")
    print(f"  Saved: {PROCESSED_DIR / 'response_matrix_verified.csv'}")
    print(f"    Shape: {rm_verified.shape}")
    print(f"    Models: {list(rm_verified.columns)}")

    rm_full.to_csv(PROCESSED_DIR / "response_matrix_full.csv")
    print(f"  Saved: {PROCESSED_DIR / 'response_matrix_full.csv'}")
    print(f"    Shape: {rm_full.shape}")
    print(f"    Models: {list(rm_full.columns)}")

    # ---- 5. Save instance metadata ----
    print("\n--- Saving instance metadata ---")
    meta_verified.to_csv(PROCESSED_DIR / "instance_metadata_verified.csv", index=False)
    print(f"  Saved: {PROCESSED_DIR / 'instance_metadata_verified.csv'}")

    meta_full.to_csv(PROCESSED_DIR / "instance_metadata_full.csv", index=False)
    print(f"  Saved: {PROCESSED_DIR / 'instance_metadata_full.csv'}")

    # ---- 6. Save leaderboard aggregate data ----
    print("\n--- Saving leaderboard aggregate data ---")
    lb_df = load_leaderboard_data()
    lb_df.to_csv(PROCESSED_DIR / "leaderboard_aggregate.csv", index=False)
    print(f"  Saved: {PROCESSED_DIR / 'leaderboard_aggregate.csv'}")
    print(f"    Entries: {len(lb_df)} ({lb_df['split'].value_counts().to_dict()})")

    # ---- 7. Build per-instance metrics (file/node retrieval) for verified ----
    print("\n--- Building per-instance metrics for PBVerified ---")
    all_metrics_dfs = []
    for model_name, logs_dir in submissions_verified:
        metrics = load_metrics_files(logs_dir)
        for iid, m in metrics.items():
            row = {"instance_id": iid, "model": model_name}
            fr = m.get("file_retrieval_metrics", {})
            if fr:
                row["file_recall"] = fr.get("recall")
                row["file_precision"] = fr.get("precision")
                row["file_f1"] = fr.get("f1")
            nr = m.get("node_retrieval_metrics", {})
            if nr:
                row["node_recall"] = nr.get("recall")
                row["node_precision"] = nr.get("precision")
                row["node_f1"] = nr.get("f1")
            all_metrics_dfs.append(row)

    metrics_df = pd.DataFrame(all_metrics_dfs)
    metrics_df.to_csv(PROCESSED_DIR / "per_instance_metrics_verified.csv", index=False)
    print(f"  Saved: {PROCESSED_DIR / 'per_instance_metrics_verified.csv'}")
    print(f"    Shape: {metrics_df.shape}")

    # ---- 8. Summary statistics ----
    print("\n--- Summary ---")
    summary = {
        "benchmark": "SWE-PolyBench",
        "source": "https://github.com/amazon-science/SWE-PolyBench",
        "splits": {
            "full": {
                "n_instances": len(meta_full),
                "n_models_with_per_instance_results": len(submissions_full),
                "models": [s[0] for s in submissions_full],
            },
            "verified": {
                "n_instances": len(meta_verified),
                "n_models_with_per_instance_results": len(submissions_verified),
                "models": [s[0] for s in submissions_verified],
            },
        },
        "leaderboard_models": {
            "full": lb_df[lb_df["split"] == "Full"]["name"].tolist(),
            "pb500": lb_df[lb_df["split"] == "PB500"]["name"].tolist(),
            "verified": lb_df[lb_df["split"] == "Verified"]["name"].tolist(),
        },
        "verified_response_matrix": {
            "shape": list(rm_verified.shape),
            "resolve_rates": {},
        },
        "full_response_matrix": {
            "shape": list(rm_full.shape),
            "resolve_rates": {},
        },
        "notes": [
            "Paper baseline results (Aider, Agentless, SWE-agent, Amazon Q) have only "
            "aggregate scores in the leaderboard HTML — no per-instance results are published.",
            "Per-instance results are only available for models submitted via the submission branch.",
            "iSWE-Agent on PB Full has '-' for overall resolved in the leaderboard — "
            "it was evaluated only on Java subset (55/165 = 33.33%).",
        ],
    }

    for col in rm_verified.columns:
        valid = rm_verified[col].dropna()
        rate = valid.mean() * 100
        summary["verified_response_matrix"]["resolve_rates"][col] = {
            "resolved": int(valid.sum()),
            "total": len(valid),
            "rate_pct": round(rate, 2),
        }

    for col in rm_full.columns:
        valid = rm_full[col].dropna()
        rate = valid.mean() * 100
        summary["full_response_matrix"]["resolve_rates"][col] = {
            "resolved": int(valid.sum()),
            "total": len(valid),
            "rate_pct": round(rate, 2),
        }

    with open(PROCESSED_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {PROCESSED_DIR / 'summary.json'}")

    # Print key stats
    print("\n" + "=" * 70)
    print("PBVerified Response Matrix:")
    print(f"  Instances: {rm_verified.shape[0]}, Models: {rm_verified.shape[1]}")
    for col in rm_verified.columns:
        valid = rm_verified[col].dropna()
        print(f"  {col}: {int(valid.sum())}/{len(valid)} "
              f"({100*valid.mean():.2f}%) resolved")

    print(f"\nPB Full Response Matrix:")
    print(f"  Instances: {rm_full.shape[0]}, Models: {rm_full.shape[1]}")
    for col in rm_full.columns:
        valid = rm_full[col].dropna()
        print(f"  {col}: {int(valid.sum())}/{len(valid)} "
              f"({100*valid.mean():.2f}%) resolved")

    # Language breakdown for verified
    print("\nPBVerified - Language breakdown:")
    merged = rm_verified.reset_index().merge(meta_verified[["instance_id", "language"]],
                                              on="instance_id", how="left")
    for lang in sorted(merged["language"].unique()):
        lang_subset = merged[merged["language"] == lang]
        print(f"  {lang} ({len(lang_subset)} instances):")
        for col in rm_verified.columns:
            valid = lang_subset[col].dropna()
            if len(valid) > 0:
                print(f"    {col}: {int(valid.sum())}/{len(valid)} "
                      f"({100*valid.mean():.2f}%)")

    print("\n" + "=" * 70)
    print("Response matrix build complete.")
    print(f"All outputs in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
