#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate Chatbot Arena 140K pairwise comparisons to torch-measure-data.

Downloads pairwise comparison data from lmarena-ai/arena-human-preference-140k,
converts it to the torch_measure PairwiseComparisons .pt format, and uploads to
HuggingFace.

Usage:
    export HF_TOKEN=hf_xxxxx
    python scripts/migrate_arena_140k_data.py

Source data format (140K rows, 70+ models):
    id: str — unique comparison identifier
    model_a, model_b: str — model names
    winner: str — "model_a", "model_b", "tie", "both_bad"
    conversation_a / conversation_b: list[dict] — full conversations
    language, is_code, timestamp, evaluation_session_id, etc.

Destination .pt file format:
    {
        "subject_a": torch.LongTensor,       # (n_comparisons,) indices into subject_ids
        "subject_b": torch.LongTensor,       # (n_comparisons,) indices into subject_ids
        "outcome": torch.Tensor,             # (n_comparisons,) 1.0=a wins, 0.0=b wins, 0.5=tie
        "subject_ids": list[str],            # unique model names (sorted)
        "item_ids": list[str],              # unique item/prompt identifiers
        "item_contents": list[str],         # text content per unique item
        "item_idx": torch.LongTensor,       # (n_comparisons,) indices into item_ids
        "subject_metadata": list[dict],      # per-subject metadata
        "comparison_metadata": list[dict],   # per-comparison metadata
    }
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "lmarena-ai/arena-human-preference-140k"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path("/tmp/torch_measure_migration")

# Winner string → outcome float
# The 140K dataset uses "both_bad" instead of "tie (bothbad)"
WINNER_MAP = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": 0.5,
    "both_bad": 0.5,
}


def load_arena_140k_data() -> pd.DataFrame:
    """Download the chatbot arena 140K dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading chatbot arena 140K data...")
    ds = load_dataset(SRC_REPO, split="train")
    df = ds.to_pandas()
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


def extract_prompt_text(conversation) -> str:
    """Extract the user's first message from a conversation array.

    The 140K dataset uses a nested content format:
        [{"role": "user", "content": [{"type": "text", "text": "..."}]}, ...]
    """
    if conversation is None:
        return ""
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "user":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", ""))
        return str(content) if content else ""
    return ""


def build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each model from its name."""
    metadata = []
    for sid in subject_ids:
        parts = sid.split("/", 1) if "/" in sid else ("", sid)
        org = parts[0] if len(parts) > 1 and parts[0] else ""
        model = parts[1] if len(parts) > 1 and parts[0] else sid
        metadata.append({
            "org": org,
            "model": model,
        })
    return metadata


def process_arena_140k_data(df: pd.DataFrame) -> dict:
    """Convert arena 140K DataFrame into the .pt payload format."""
    print("\nProcessing arena 140K data...")

    # Filter to rows with known winner values
    known_winners = df["winner"].isin(WINNER_MAP)
    n_dropped = (~known_winners).sum()
    if n_dropped > 0:
        unknown = df.loc[~known_winners, "winner"].unique()
        print(f"  Dropping {n_dropped} rows with unknown winner values: {unknown}")
    df = df[known_winners].reset_index(drop=True)

    # Build sorted unique subject list
    all_models = sorted(set(df["model_a"]) | set(df["model_b"]))
    sid_to_idx = {s: i for i, s in enumerate(all_models)}
    print(f"  Unique models: {len(all_models)}")

    # Map to indices
    subject_a = torch.tensor([sid_to_idx[s] for s in df["model_a"]], dtype=torch.long)
    subject_b = torch.tensor([sid_to_idx[s] for s in df["model_b"]], dtype=torch.long)

    # Encode outcomes
    outcome = torch.tensor([WINNER_MAP[w] for w in df["winner"]], dtype=torch.float32)

    # Items (prompts) — deduplicate by id
    raw_ids = df["id"].tolist()
    print("  Extracting prompt text from conversations...")
    raw_prompt_texts = [extract_prompt_text(conv) for conv in df["conversation_a"]]

    # Build unique item list, preserving first-seen order
    seen_items: dict[str, int] = {}
    item_ids: list[str] = []
    item_contents: list[str] = []
    item_idx_list: list[int] = []
    for qid, text in zip(raw_ids, raw_prompt_texts):
        if qid not in seen_items:
            seen_items[qid] = len(item_ids)
            item_ids.append(qid)
            item_contents.append(text)
        item_idx_list.append(seen_items[qid])
    item_idx = torch.tensor(item_idx_list, dtype=torch.long)
    print(f"  Unique items/prompts: {len(item_ids)}")

    # Subject metadata
    subject_metadata = build_subject_metadata(all_models)

    # Comparison metadata
    comparison_metadata = []
    for _, row in df.iterrows():
        comparison_metadata.append({
            "language": row.get("language", ""),
            "is_code": bool(row.get("is_code", False)),
            "timestamp": str(row.get("timestamp", "")),
            "evaluation_session_id": row.get("evaluation_session_id", ""),
            "evaluation_order": int(row.get("evaluation_order", 1)),
        })

    n_comparisons = len(outcome)
    n_a_wins = (outcome == 1.0).sum().item()
    n_b_wins = (outcome == 0.0).sum().item()
    n_ties = (outcome == 0.5).sum().item()
    print(f"  Comparisons: {n_comparisons:,}")
    print(f"  Outcomes: {n_a_wins:,.0f} model_a wins, {n_b_wins:,.0f} model_b wins, {n_ties:,.0f} ties")

    return {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "outcome": outcome,
        "subject_ids": all_models,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "item_idx": item_idx,
        "subject_metadata": subject_metadata,
        "comparison_metadata": comparison_metadata,
    }


def ensure_repo(api: HfApi) -> None:
    """Create the destination repo if it doesn't exist."""
    try:
        api.repo_info(DST_REPO, repo_type="dataset")
        print(f"Destination repo {DST_REPO} already exists.")
    except Exception:
        print(f"Creating dataset repo {DST_REPO}...")
        api.create_repo(DST_REPO, repo_type="dataset", private=False)


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ensure_repo(api)

    # Step 1: Download
    df = load_arena_140k_data()

    # Step 2: Process
    payload = process_arena_140k_data(df)

    # Step 3: Save locally
    filename = "arena/chatbot_arena_140k.pt"
    local_path = TMP_DIR / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, local_path)
    print(f"\nSaved to {local_path}")

    # Step 4: Upload
    print(f"Uploading to {DST_REPO}/{filename}...")
    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=DST_REPO,
        repo_type="dataset",
    )

    # Step 5: Summary
    n_subjects = len(payload["subject_ids"])
    n_comparisons = len(payload["outcome"])
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}/{filename}")
    print(f"  Models: {n_subjects}")
    print(f"  Comparisons: {n_comparisons:,}")
    print(f"\n  Update arena.py with: n_subjects={n_subjects}, n_comparisons={n_comparisons}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
