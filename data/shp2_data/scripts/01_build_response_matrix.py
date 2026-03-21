"""
Build SHP-2 response matrices from Stanford Human Preferences v2.

Data source:
  - stanfordnlp/SHP-2 on HuggingFace Hub
  - 4.8M naturally-occurring pairwise preferences across 129 subject areas
    from Reddit and StackExchange
  - Preferences inferred from upvote differentials

Processing:
  1. Stream dataset from HuggingFace (too large to load in memory)
  2. Compute per-domain summary statistics
  3. Sample a representative 100K-pair subset
  4. Build response matrices:
     - domain_stats: (domains x metrics) — per-domain aggregated statistics
     - sampled_pairs: (2 x 100K) — binary preference for sampled pairs

Outputs:
  - raw/shp2_domain_stats.csv: Per-domain aggregated statistics
  - raw/shp2_sampled_pairs.csv: 100K sampled raw preference pairs
  - processed/response_matrix_domain_stats.csv: Domains (rows) x metrics (columns)
  - processed/response_matrix_sampled_pairs.csv: Response positions (rows) x pairs (columns)
"""

import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Config
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42


def stream_dataset():
    """Stream SHP-2 dataset and compute per-domain stats + reservoir sample."""

    hf_token = os.environ.get("HF_TOKEN")

    print("  Streaming stanfordnlp/SHP-2 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/SHP-2", split="train", streaming=True, token=hf_token)

    # Per-domain accumulators
    domain_stats = defaultdict(lambda: {
        "n_pairs": 0,
        "n_a_preferred": 0,
        "n_b_preferred": 0,
        "sum_score_a": 0,
        "sum_score_b": 0,
        "sum_score_ratio": 0.0,
    })

    # Reservoir sampling
    reservoir = []
    total_seen = 0
    rng = random.Random(RANDOM_SEED)

    for i, row in enumerate(ds):
        total_seen += 1

        domain = row.get("domain", "unknown")
        label = row.get("labels", 0)
        score_a = row.get("score_A", 0)
        score_b = row.get("score_B", 0)

        # Update domain stats
        stats = domain_stats[domain]
        stats["n_pairs"] += 1
        if label == 1:
            stats["n_a_preferred"] += 1
        else:
            stats["n_b_preferred"] += 1
        stats["sum_score_a"] += score_a
        stats["sum_score_b"] += score_b
        total_score = abs(score_a) + abs(score_b)
        if total_score > 0:
            stats["sum_score_ratio"] += max(score_a, score_b) / total_score

        # Reservoir sampling
        pair_record = {
            "domain": domain,
            "label": label,
            "score_A": score_a,
            "score_B": score_b,
            "history": str(row.get("history", ""))[:500],
            "human_ref_A": str(row.get("human_ref_A", ""))[:500],
            "human_ref_B": str(row.get("human_ref_B", ""))[:500],
        }

        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(pair_record)
        else:
            j = rng.randint(0, total_seen - 1)
            if j < SAMPLE_SIZE:
                reservoir[j] = pair_record

        if (i + 1) % 500_000 == 0:
            print(f"    Processed {i + 1:,} rows, {len(domain_stats)} domains ...")

    print(f"  Total rows streamed: {total_seen:,}")
    print(f"  Total domains: {len(domain_stats)}")
    print(f"  Reservoir sample: {len(reservoir)}")

    return dict(domain_stats), reservoir


def save_raw_data(domain_stats, sampled_pairs):
    """Save raw data to CSV for inspection."""
    print("\n  Saving raw data...")

    # Domain stats
    rows = []
    for domain, stats in sorted(domain_stats.items()):
        n = stats["n_pairs"]
        rows.append({
            "domain": domain,
            "n_pairs": n,
            "n_a_preferred": stats["n_a_preferred"],
            "n_b_preferred": stats["n_b_preferred"],
            "pref_rate_a": stats["n_a_preferred"] / n if n > 0 else 0.5,
            "mean_score_a": stats["sum_score_a"] / n if n > 0 else 0,
            "mean_score_b": stats["sum_score_b"] / n if n > 0 else 0,
            "mean_score_ratio": stats["sum_score_ratio"] / n if n > 0 else 0.5,
        })
    domain_df = pd.DataFrame(rows)
    domain_path = os.path.join(RAW_DIR, "shp2_domain_stats.csv")
    domain_df.to_csv(domain_path, index=False)
    print(f"    Domain stats: {domain_path} ({len(domain_df)} domains)")

    # Sampled pairs
    sample_df = pd.DataFrame(sampled_pairs)
    sample_path = os.path.join(RAW_DIR, "shp2_sampled_pairs.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"    Sampled pairs: {sample_path} ({len(sample_df):,} pairs)")

    return domain_df, sample_df


def build_domain_stats_matrix(domain_stats):
    """Build domain stats response matrix (domains x metrics)."""
    print("\n  Building domain stats response matrix...")

    sorted_domains = sorted(domain_stats.keys())
    metric_names = [
        "pref_rate_a",
        "mean_score_a_norm",
        "mean_score_b_norm",
        "mean_score_ratio",
        "log_n_pairs_norm",
    ]

    # Compute raw values
    raw_data = []
    max_log_pairs = 0
    max_mean_score = 1

    for domain in sorted_domains:
        s = domain_stats[domain]
        n = s["n_pairs"]
        mean_score_a = s["sum_score_a"] / n if n > 0 else 0
        mean_score_b = s["sum_score_b"] / n if n > 0 else 0
        log_n = math.log1p(n)
        max_log_pairs = max(max_log_pairs, log_n)
        max_mean_score = max(max_mean_score, abs(mean_score_a), abs(mean_score_b))
        raw_data.append({
            "pref_rate_a": s["n_a_preferred"] / n if n > 0 else 0.5,
            "mean_score_a": mean_score_a,
            "mean_score_b": mean_score_b,
            "mean_score_ratio": s["sum_score_ratio"] / n if n > 0 else 0.5,
            "log_n_pairs": log_n,
        })

    # Normalize
    rows = []
    for rd in raw_data:
        rows.append([
            rd["pref_rate_a"],
            rd["mean_score_a"] / max_mean_score if max_mean_score > 0 else 0,
            rd["mean_score_b"] / max_mean_score if max_mean_score > 0 else 0,
            rd["mean_score_ratio"],
            rd["log_n_pairs"] / max_log_pairs if max_log_pairs > 0 else 0,
        ])

    matrix_df = pd.DataFrame(rows, index=sorted_domains, columns=metric_names)
    matrix_df.index.name = "Model"  # Convention

    output_path = os.path.join(PROCESSED_DIR, "response_matrix_domain_stats.csv")
    matrix_df.to_csv(output_path)
    print(f"    Saved: {output_path}")
    print(f"    Dimensions: {matrix_df.shape[0]} domains x {matrix_df.shape[1]} metrics")

    return matrix_df


def build_sampled_pairs_matrix(sampled_pairs):
    """Build sampled pairs response matrix (2 response positions x N pairs)."""
    print("\n  Building sampled pairs response matrix...")

    n = len(sampled_pairs)
    pair_ids = [f"pair_{i}" for i in range(n)]

    row_a = [float(p["label"]) for p in sampled_pairs]
    row_b = [float(1 - p["label"]) for p in sampled_pairs]

    matrix_df = pd.DataFrame(
        [row_a, row_b],
        index=["response_A", "response_B"],
        columns=pair_ids,
    )
    matrix_df.index.name = "Model"

    output_path = os.path.join(PROCESSED_DIR, "response_matrix_sampled_pairs.csv")
    matrix_df.to_csv(output_path)
    print(f"    Saved: {output_path}")
    print(f"    Dimensions: {matrix_df.shape[0]} positions x {matrix_df.shape[1]:,} pairs")

    return matrix_df


def print_statistics(domain_stats, sampled_pairs, domain_matrix, pairs_matrix):
    """Print detailed statistics."""
    print(f"\n{'=' * 60}")
    print(f"  SHP-2 STATISTICS")
    print(f"{'=' * 60}")

    # Overall
    total_pairs = sum(s["n_pairs"] for s in domain_stats.values())
    total_a = sum(s["n_a_preferred"] for s in domain_stats.values())
    total_b = sum(s["n_b_preferred"] for s in domain_stats.values())
    print(f"\n  Overall:")
    print(f"    Total pairs:        {total_pairs:,}")
    print(f"    Total domains:      {len(domain_stats)}")
    print(f"    Prefer A:           {total_a:,} ({total_a / total_pairs * 100:.1f}%)")
    print(f"    Prefer B:           {total_b:,} ({total_b / total_pairs * 100:.1f}%)")

    # Domain stats matrix
    n_d, n_m = domain_matrix.shape
    print(f"\n  Domain stats matrix: {n_d} domains x {n_m} metrics")
    for col in domain_matrix.columns:
        print(f"    {col:25s}  mean={domain_matrix[col].mean():.3f}  "
              f"std={domain_matrix[col].std():.3f}  "
              f"range=[{domain_matrix[col].min():.3f}, {domain_matrix[col].max():.3f}]")

    # Top domains
    sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1]["n_pairs"], reverse=True)
    print(f"\n  Top 20 domains by pair count:")
    for domain, stats in sorted_domains[:20]:
        n = stats["n_pairs"]
        pref_a = stats["n_a_preferred"] / n if n > 0 else 0
        print(f"    {domain:40s}  n={n:>8,}  pref_A={pref_a:.3f}")

    # Sampled pairs
    n_s, n_p = pairs_matrix.shape
    print(f"\n  Sampled pairs matrix: {n_s} positions x {n_p:,} pairs")

    # Sample preference distribution
    sample_a = sum(1 for p in sampled_pairs if p["label"] == 1)
    sample_b = len(sampled_pairs) - sample_a
    print(f"    Prefer A: {sample_a:,} ({sample_a / len(sampled_pairs) * 100:.1f}%)")
    print(f"    Prefer B: {sample_b:,} ({sample_b / len(sampled_pairs) * 100:.1f}%)")

    # Sample domain distribution
    sample_domains = defaultdict(int)
    for p in sampled_pairs:
        sample_domains[p["domain"]] += 1
    print(f"\n  Sampled pairs domain distribution (top 10):")
    for domain, count in sorted(sample_domains.items(), key=lambda x: -x[1])[:10]:
        print(f"    {domain:40s}  n={count:>6,}")

    # Output files
    print(f"\n  Output files:")
    for d in [RAW_DIR, PROCESSED_DIR]:
        for f in sorted(os.listdir(d)):
            fpath = os.path.join(d, f)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                reldir = "raw" if d == RAW_DIR else "processed"
                print(f"    {reldir}/{f:45s}  {size_kb:.1f} KB")


def main():
    print("SHP-2 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print(f"  Sample size:        {SAMPLE_SIZE:,}")
    print()

    # Step 1: Stream dataset
    print("STEP 1: Streaming SHP-2 dataset")
    print("-" * 60)
    domain_stats, sampled_pairs = stream_dataset()

    if not domain_stats:
        print("  ERROR: No data streamed. Check dataset access.")
        sys.exit(1)

    # Step 2: Save raw data
    print("\nSTEP 2: Saving raw data")
    print("-" * 60)
    save_raw_data(domain_stats, sampled_pairs)

    # Step 3: Build response matrices
    print("\nSTEP 3: Building response matrices")
    print("-" * 60)
    domain_matrix = build_domain_stats_matrix(domain_stats)
    pairs_matrix = build_sampled_pairs_matrix(sampled_pairs)

    # Step 4: Statistics
    print("\nSTEP 4: Detailed statistics")
    print("-" * 60)
    print_statistics(domain_stats, sampled_pairs, domain_matrix, pairs_matrix)


if __name__ == "__main__":
    main()
