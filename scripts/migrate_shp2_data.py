#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Stanford Human Preferences v2 (SHP-2) data to torch-measure-data.

Streams the stanfordnlp/SHP-2 dataset from HuggingFace Hub, computes
per-domain preference statistics, samples a representative subset of
raw preference pairs, and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_shp2_data.py

Source data: https://huggingface.co/datasets/stanfordnlp/SHP-2
    4.8M naturally-occurring pairwise preferences across 129 subject areas
    from Reddit and StackExchange.

Each row has:
    - history: the post/question text
    - human_ref_A: response A
    - human_ref_B: response B
    - labels: 1 if A is preferred, 0 if B is preferred
    - score_A: upvote score for response A
    - score_B: upvote score for response B
    - domain: subreddit or StackExchange site (e.g., "askculinary_train")

Outputs:
    shp2/domain_stats.pt    — Per-domain summary statistics as a response matrix
                              (domains x metrics), continuous [0, 1] values.
    shp2/sampled_pairs.pt   — 100K sampled preference pairs as a response matrix
                              (2 responses x N pairs), binary preferred label.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # row identifiers
        "item_ids": list[str],             # column identifiers
        "subject_metadata": list[dict],    # structured metadata
    }
"""

from __future__ import annotations

import random
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_DATASET = "stanfordnlp/SHP-2"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_shp2_migration"

# Number of raw pairs to sample for the sampled_pairs dataset
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def stream_and_aggregate(token: str | None = None) -> tuple[dict, list[dict]]:
    """Stream SHP-2 and compute per-domain stats + reservoir-sample pairs.

    Returns
    -------
    domain_stats : dict[str, dict]
        Per-domain aggregated statistics.
    sampled_pairs : list[dict]
        Reservoir-sampled subset of raw preference pairs.
    """
    from datasets import load_dataset

    print("Streaming stanfordnlp/SHP-2 ...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True, token=token)

    # Per-domain accumulators
    domain_stats: dict[str, dict] = defaultdict(lambda: {
        "n_pairs": 0,
        "n_a_preferred": 0,
        "n_b_preferred": 0,
        "sum_score_a": 0,
        "sum_score_b": 0,
        "sum_score_ratio": 0.0,
    })

    # Reservoir sampling for raw pairs
    reservoir: list[dict] = []
    total_seen = 0

    rng = random.Random(RANDOM_SEED)

    for i, row in enumerate(ds):
        total_seen += 1

        # Extract fields
        domain = row.get("domain", "unknown")
        label = row.get("labels", 0)  # 1 = A preferred, 0 = B preferred
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
        # Score ratio: higher / (higher + lower) to measure preference strength
        total_score = abs(score_a) + abs(score_b)
        if total_score > 0:
            stats["sum_score_ratio"] += max(score_a, score_b) / total_score

        # Reservoir sampling
        pair_record = {
            "domain": domain,
            "label": label,
            "score_A": score_a,
            "score_B": score_b,
            "history": row.get("history", "")[:500],  # truncate for memory
            "human_ref_A": row.get("human_ref_A", "")[:500],
            "human_ref_B": row.get("human_ref_B", "")[:500],
        }

        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(pair_record)
        else:
            j = rng.randint(0, total_seen - 1)
            if j < SAMPLE_SIZE:
                reservoir[j] = pair_record

        if (i + 1) % 500_000 == 0:
            print(f"  Processed {i + 1:,} rows, {len(domain_stats)} domains so far ...")

    print(f"\n  Total rows streamed: {total_seen:,}")
    print(f"  Total domains: {len(domain_stats)}")
    print(f"  Reservoir sample size: {len(reservoir)}")

    return dict(domain_stats), reservoir


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def build_domain_stats_payload(domain_stats: dict[str, dict]) -> dict:
    """Build a response matrix of per-domain summary statistics.

    Rows (subjects): domains/subreddits
    Columns (items): metric names
    Values: continuous [0, 1] normalized statistics
    """
    sorted_domains = sorted(domain_stats.keys())

    # Metrics to include as columns
    metric_names = [
        "pref_rate_a",          # fraction preferring response A
        "mean_score_a_norm",    # normalized mean score A
        "mean_score_b_norm",    # normalized mean score B
        "mean_score_ratio",     # mean preference strength
        "log_n_pairs_norm",     # log-normalized number of pairs
    ]

    import math

    # Compute raw metrics
    raw_data = []
    max_log_pairs = 0
    max_mean_score = 1  # will compute

    for domain in sorted_domains:
        s = domain_stats[domain]
        n = s["n_pairs"]
        pref_rate_a = s["n_a_preferred"] / n if n > 0 else 0.5
        mean_score_a = s["sum_score_a"] / n if n > 0 else 0
        mean_score_b = s["sum_score_b"] / n if n > 0 else 0
        mean_ratio = s["sum_score_ratio"] / n if n > 0 else 0.5
        log_n = math.log1p(n)

        max_log_pairs = max(max_log_pairs, log_n)
        max_mean_score = max(max_mean_score, abs(mean_score_a), abs(mean_score_b))

        raw_data.append({
            "pref_rate_a": pref_rate_a,
            "mean_score_a": mean_score_a,
            "mean_score_b": mean_score_b,
            "mean_score_ratio": mean_ratio,
            "log_n_pairs": log_n,
        })

    # Normalize to [0, 1]
    rows = []
    for rd in raw_data:
        row = [
            rd["pref_rate_a"],                                    # already [0, 1]
            rd["mean_score_a"] / max_mean_score if max_mean_score > 0 else 0,
            rd["mean_score_b"] / max_mean_score if max_mean_score > 0 else 0,
            rd["mean_score_ratio"],                                # already [0, 1]
            rd["log_n_pairs"] / max_log_pairs if max_log_pairs > 0 else 0,
        ]
        rows.append(row)

    data = torch.tensor(rows, dtype=torch.float32)

    subject_metadata = []
    for domain in sorted_domains:
        s = domain_stats[domain]
        subject_metadata.append({
            "domain": domain,
            "n_pairs": s["n_pairs"],
            "n_a_preferred": s["n_a_preferred"],
            "n_b_preferred": s["n_b_preferred"],
        })

    return {
        "data": data,
        "subject_ids": sorted_domains,
        "item_ids": metric_names,
        "subject_metadata": subject_metadata,
    }


def build_sampled_pairs_payload(sampled_pairs: list[dict]) -> dict:
    """Build a response matrix from sampled preference pairs.

    Rows (subjects): "response_A" and "response_B" (2 rows)
    Columns (items): individual pair indices
    Values: binary — for response_A row: labels (1 if A preferred),
            for response_B row: 1 - labels (1 if B preferred).

    This encoding lets us treat each pair as an "item" and each response
    position as a "subject", with the binary value indicating preference.
    """
    n = len(sampled_pairs)
    pair_ids = [f"pair_{i}" for i in range(n)]
    subject_ids = ["response_A", "response_B"]

    # Build 2 x N matrix
    row_a = []
    row_b = []
    for pair in sampled_pairs:
        label = pair["label"]
        row_a.append(float(label))        # 1 if A preferred
        row_b.append(float(1 - label))    # 1 if B preferred

    data = torch.tensor([row_a, row_b], dtype=torch.float32)

    # Domain distribution in sample
    domain_counts: dict[str, int] = defaultdict(int)
    for pair in sampled_pairs:
        domain_counts[pair["domain"]] += 1

    subject_metadata = [
        {"position": "A", "description": "First response in pair"},
        {"position": "B", "description": "Second response in pair"},
    ]

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": pair_ids,
        "subject_metadata": subject_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import os

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    # Step 1: Stream and aggregate
    print("=" * 60)
    print("Step 1: Streaming SHP-2 and computing statistics ...")
    print("=" * 60)
    domain_stats, sampled_pairs = stream_and_aggregate(token=token)

    payloads: dict[str, dict] = {}

    # Step 2: Build domain stats response matrix
    print("\n" + "=" * 60)
    print("Step 2: Building domain stats response matrix ...")
    print("=" * 60)
    payloads["shp2/domain_stats"] = build_domain_stats_payload(domain_stats)
    n_s, n_i = payloads["shp2/domain_stats"]["data"].shape
    print(f"  shp2/domain_stats: {n_s} domains x {n_i} metrics")

    # Step 3: Build sampled pairs response matrix
    print("\n" + "=" * 60)
    print("Step 3: Building sampled pairs response matrix ...")
    print("=" * 60)
    payloads["shp2/sampled_pairs"] = build_sampled_pairs_payload(sampled_pairs)
    n_s, n_i = payloads["shp2/sampled_pairs"]["data"].shape
    print(f"  shp2/sampled_pairs: {n_s} response positions x {n_i} pairs")

    # Print domain stats summary
    print("\n" + "=" * 60)
    print("Domain summary (top 20 by pair count):")
    print("=" * 60)
    sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1]["n_pairs"], reverse=True)
    for domain, stats in sorted_domains[:20]:
        n = stats["n_pairs"]
        pref_a = stats["n_a_preferred"] / n if n > 0 else 0
        print(f"  {domain:40s}  n={n:>8,}  pref_A={pref_a:.3f}")

    # Print sample domain distribution
    print("\n" + "=" * 60)
    print("Sampled pairs domain distribution (top 10):")
    print("=" * 60)
    sample_domains: dict[str, int] = defaultdict(int)
    for pair in sampled_pairs:
        sample_domains[pair["domain"]] += 1
    for domain, count in sorted(sample_domains.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain:40s}  n={count:>6,}")

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Step 4: Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {n_sub} x {n_items}, {nan_pct:.1%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {HF_DATASET}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for shp2.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
