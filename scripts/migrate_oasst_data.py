#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate OpenAssistant OASST1 data to torch-measure-data.

Downloads the OpenAssistant/oasst1 dataset from HuggingFace Hub, extracts
assistant messages with their human-assigned ranks grouped by parent prompt,
builds response matrices (models/labels x prompts), and uploads .pt files
to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_oasst_data.py

Source data: https://huggingface.co/datasets/OpenAssistant/oasst1

OpenAssistant OASST1 contains ~161K messages in conversation trees with
~461K quality ratings from human volunteers. Messages have role labels,
ranks, and quality labels. Multiple assistant responses per prompt are
ranked by annotators.

The response matrix is built by:
1. Finding all assistant messages that share the same parent (prompt) message.
2. Grouping these into "prompt groups" where multiple alternatives exist.
3. Using the human-assigned rank as the quality signal.
4. Normalizing ranks to [0, 1] where 1.0 = best (rank 0) and 0.0 = worst.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # label identifiers (e.g., "rank_0", "rank_1", ...)
        "item_ids": list[str],             # prompt message_id strings
        "subject_metadata": list[dict],    # per-label metadata
    }
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_DATASET = "OpenAssistant/oasst1"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_oasst_migration"

# We also build a subset with prompts that have >= 3 ranked alternatives.
MIN_ALTERNATIVES_RICH = 3

# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_LANG_MAP = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "it": "Italian",
    "pl": "Polish",
    "nl": "Dutch",
    "tr": "Turkish",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "hu": "Hungarian",
    "ca": "Catalan",
    "fi": "Finnish",
    "sv": "Swedish",
    "da": "Danish",
    "cs": "Czech",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "he": "Hebrew",
    "id": "Indonesian",
    "eu": "Basque",
    "gl": "Galician",
    "eo": "Esperanto",
}


def _build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each subject (rank tier)."""
    metadata = []
    for sid in subject_ids:
        metadata.append(
            {
                "rank_tier": sid,
                "description": f"Assistant response with {sid}",
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Data loading & matrix construction
# ---------------------------------------------------------------------------


def load_oasst_data() -> dict:
    """Load OASST1 dataset and extract message tree structure.

    Returns a dict with:
    - messages: dict mapping message_id -> message record
    - prompt_groups: dict mapping parent_id -> list of (message_id, rank) for
      assistant responses
    """
    print(f"Loading {SRC_DATASET} ...")
    ds = load_dataset(SRC_DATASET, split="train+validation")

    print(f"  Total messages: {len(ds):,}")

    # Index all messages by message_id
    messages = {}
    for item in ds:
        msg_id = item["message_id"]
        messages[msg_id] = {
            "message_id": msg_id,
            "parent_id": item.get("parent_id"),
            "role": item.get("role"),
            "text": item.get("text", ""),
            "rank": item.get("rank"),
            "lang": item.get("lang", ""),
            "labels": item.get("labels"),
            "message_tree_id": item.get("message_tree_id"),
        }

    # Find prompt groups: assistant responses sharing the same parent
    # Group by parent_id, keep only assistant messages with valid rank
    prompt_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    n_assistant = 0
    n_with_rank = 0

    for msg_id, msg in messages.items():
        if msg["role"] == "assistant":
            n_assistant += 1
            parent_id = msg["parent_id"]
            rank = msg["rank"]
            if rank is not None and parent_id is not None:
                n_with_rank += 1
                prompt_groups[parent_id].append((msg_id, int(rank)))

    # Filter to only groups with >= 2 alternatives (needed for ranking)
    multi_groups = {
        pid: alts for pid, alts in prompt_groups.items() if len(alts) >= 2
    }

    print(f"  Assistant messages: {n_assistant:,}")
    print(f"  With valid rank: {n_with_rank:,}")
    print(f"  Unique parent prompts with ranked alternatives: {len(prompt_groups):,}")
    print(f"  Prompts with >= 2 ranked alternatives: {len(multi_groups):,}")

    # Distribution of group sizes
    sizes = [len(v) for v in multi_groups.values()]
    if sizes:
        from collections import Counter

        size_counts = Counter(sizes)
        print(f"  Group size distribution:")
        for sz in sorted(size_counts.keys()):
            print(f"    {sz} alternatives: {size_counts[sz]:,} prompts")

    return {
        "messages": messages,
        "prompt_groups": multi_groups,
    }


def build_response_matrix(
    data: dict,
    min_alternatives: int = 2,
) -> dict:
    """Build a response matrix from OASST1 ranked alternatives.

    The matrix has:
    - Rows (subjects): rank tiers (rank_0, rank_1, ..., rank_N)
      where rank_0 is the best response, rank_1 is second best, etc.
    - Columns (items): prompt parent_ids
    - Values: normalized rank score in [0, 1] where 1.0 = best

    Parameters
    ----------
    data : dict
        Output from load_oasst_data().
    min_alternatives : int
        Minimum number of ranked alternatives per prompt to include.

    Returns
    -------
    dict
        Payload with data, subject_ids, item_ids, subject_metadata.
    """
    prompt_groups = data["prompt_groups"]

    # Filter by min_alternatives
    filtered = {
        pid: alts
        for pid, alts in prompt_groups.items()
        if len(alts) >= min_alternatives
    }

    if not filtered:
        raise ValueError(f"No prompt groups with >= {min_alternatives} alternatives")

    # Determine max number of rank tiers
    max_rank = max(max(rank for _, rank in alts) for alts in filtered.values())

    # Subject IDs are rank tiers
    n_subjects = max_rank + 1  # rank 0 through max_rank
    subject_ids = [f"rank_{i}" for i in range(n_subjects)]

    # Item IDs are parent prompt IDs, sorted for consistency
    item_ids = sorted(filtered.keys())
    item_to_col = {pid: i for i, pid in enumerate(item_ids)}

    n_items = len(item_ids)
    print(f"  Building matrix: {n_subjects} rank tiers x {n_items} prompts ...")

    # Initialize with NaN
    matrix = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)

    # Fill in normalized ranks
    for pid, alts in filtered.items():
        col = item_to_col[pid]
        n_alts = len(alts)
        max_rank_in_group = max(rank for _, rank in alts)

        for _msg_id, rank in alts:
            if rank < n_subjects:
                # Normalize: rank 0 -> 1.0, max_rank_in_group -> 0.0
                if max_rank_in_group > 0:
                    score = 1.0 - rank / max_rank_in_group
                else:
                    score = 1.0
                matrix[rank, col] = score

    subject_metadata = _build_subject_metadata(subject_ids)

    return {
        "data": matrix,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


def build_label_matrix(data: dict) -> dict:
    """Build a response matrix based on quality labels.

    Uses the average quality label score (from human annotations) instead of
    rank. Each assistant message's labels contain quality, toxicity, humor, etc.
    We extract the "quality" label value.

    The matrix has:
    - Rows (subjects): model/label source tiers (rank_0, rank_1, ...)
    - Columns (items): prompt parent_ids
    - Values: mean quality label score normalized to [0, 1]
    """
    messages = data["messages"]
    prompt_groups = data["prompt_groups"]

    # For each prompt group, collect quality scores per rank tier
    filtered = {
        pid: alts for pid, alts in prompt_groups.items() if len(alts) >= 2
    }

    # Determine max rank
    max_rank = max(max(rank for _, rank in alts) for alts in filtered.values())
    n_subjects = max_rank + 1
    subject_ids = [f"rank_{i}" for i in range(n_subjects)]

    item_ids = sorted(filtered.keys())
    item_to_col = {pid: i for i, pid in enumerate(item_ids)}
    n_items = len(item_ids)

    print(f"  Building label matrix: {n_subjects} rank tiers x {n_items} prompts ...")

    matrix = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)

    n_with_quality = 0
    for pid, alts in filtered.items():
        col = item_to_col[pid]
        for msg_id, rank in alts:
            msg = messages.get(msg_id)
            if msg is None or rank >= n_subjects:
                continue

            labels = msg.get("labels")
            if labels and isinstance(labels, dict):
                quality = labels.get("quality")
                if quality is not None and isinstance(quality, dict):
                    value = quality.get("value")
                    if value is not None:
                        # Quality scores are on 0-1 scale
                        matrix[rank, col] = float(value)
                        n_with_quality += 1

    print(f"  Cells with quality labels: {n_with_quality:,}")
    subject_metadata = _build_subject_metadata(subject_ids)

    return {
        "data": matrix,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Loading OASST1 dataset ...")
    print("=" * 60)
    data = load_oasst_data()

    payloads: dict[str, dict] = {}

    # Step 2: Build response matrix — all prompts with >= 2 alternatives
    print("\n" + "=" * 60)
    print("Step 2: Building response matrices ...")
    print("=" * 60)

    print(f"\n--- oasst/ranked (>= 2 alternatives) ---")
    payloads["oasst/ranked"] = build_response_matrix(data, min_alternatives=2)
    n_s, n_i = payloads["oasst/ranked"]["data"].shape
    nan_pct = torch.isnan(payloads["oasst/ranked"]["data"]).float().mean().item()
    print(f"  Shape: {n_s} x {n_i}, {nan_pct:.1%} missing")

    # Step 3: Build response matrix — rich subset with >= 3 alternatives
    print(f"\n--- oasst/ranked_rich (>= {MIN_ALTERNATIVES_RICH} alternatives) ---")
    payloads["oasst/ranked_rich"] = build_response_matrix(
        data, min_alternatives=MIN_ALTERNATIVES_RICH
    )
    n_s, n_i = payloads["oasst/ranked_rich"]["data"].shape
    nan_pct = torch.isnan(payloads["oasst/ranked_rich"]["data"]).float().mean().item()
    print(f"  Shape: {n_s} x {n_i}, {nan_pct:.1%} missing")

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Step 3: Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        filesize_mb = local_path.stat().st_size / (1024 * 1024)
        print(
            f"  {filename}: {n_sub} x {n_items}, "
            f"{nan_pct:.1%} missing, {filesize_mb:.1f} MB"
        )

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )
        print(f"    -> uploaded to {DST_REPO}/{filename}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_DATASET}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for oasst.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}, missing={nan_pct:.1%}")
    print(f"\nSubject (rank tier) list:")
    for name, payload in sorted(payloads.items()):
        print(f"\n  {name}:")
        for i, sid in enumerate(payload["subject_ids"]):
            non_nan = (~torch.isnan(payload["data"][i])).sum().item()
            print(f"    {sid}: {non_nan:,} prompts with data")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
