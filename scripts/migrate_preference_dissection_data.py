#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate GAIR Preference Dissection data to torch-measure-data.

Downloads the GAIR/preference-dissection dataset from HuggingFace Hub,
builds response matrices (judges x pairs), and uploads .pt files
to the torch-measure-data repository.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write+gated-access to GAIR/preference-dissection
    python scripts/migrate_preference_dissection_data.py

Source data: https://huggingface.co/datasets/GAIR/preference-dissection
    5,240 conversation pairs from Chatbot Arena, each evaluated by 32 LLM
    judges plus 1 human judge.  Each judge gives a binary pairwise preference
    (which response is better: response_1 or response_2).

Reference:
    Li et al., "Dissecting Human and LLM Preferences", ACL 2024.
    arXiv:2402.11296

Destination .pt files:

  preference_dissection/all_judges.pt — 2D response matrix (judges as subjects):
    {
        "data": torch.Tensor,             # (n_judges, n_pairs), float32, binary 0/1
        "subject_ids": list[str],          # judge model names
        "item_ids": list[str],             # pair identifiers
        "subject_metadata": list[dict],    # per-judge metadata (org, model_type)
    }

  preference_dissection/crossed.pt — 3D version for G-theory facet analysis:
    {
        "data": torch.Tensor,             # (n_pairs, n_judges), float32, binary 0/1
        "pair_ids": list[str],
        "judge_ids": list[str],
        "judge_metadata": list[dict],
    }
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "GAIR/preference-dissection"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_preference_dissection_migration"

# Judge -> organization mapping
_ORG_MAP: dict[str, str] = {
    "gpt-3.5-turbo-1106": "OpenAI",
    "gpt-4-1106-preview": "OpenAI",
    "human": "Human",
    "llama-2-13b": "Meta",
    "llama-2-13b-chat": "Meta",
    "llama-2-70b": "Meta",
    "llama-2-70b-chat": "Meta",
    "llama-2-7b": "Meta",
    "llama-2-7b-chat": "Meta",
    "mistral-7b": "Mistral",
    "mistral-7b-instruct-v0.1": "Mistral",
    "mistral-7b-instruct-v0.2": "Mistral",
    "mistral-8x7b": "Mistral",
    "mistral-8x7b-instruct-v0.1": "Mistral",
    "qwen-14b": "Alibaba",
    "qwen-14b-chat": "Alibaba",
    "qwen-72b": "Alibaba",
    "qwen-72b-chat": "Alibaba",
    "qwen-7b": "Alibaba",
    "qwen-7b-chat": "Alibaba",
    "tulu-2-dpo-13b": "AllenAI",
    "tulu-2-dpo-70b": "AllenAI",
    "tulu-2-dpo-7b": "AllenAI",
    "vicuna-13b-v1.5": "LMSYS",
    "vicuna-7b-v1.5": "LMSYS",
    "wizardLM-13b-v1.2": "Microsoft",
    "wizardLM-70b-v1.0": "Microsoft",
    "yi-34b": "01.AI",
    "yi-34b-chat": "01.AI",
    "yi-6b": "01.AI",
    "yi-6b-chat": "01.AI",
    "zephyr-7b-alpha": "HuggingFace",
    "zephyr-7b-beta": "HuggingFace",
}


def _model_type(judge_name: str) -> str:
    """Classify a judge as 'human', 'base', or 'chat'."""
    if judge_name == "human":
        return "human"
    if any(
        kw in judge_name
        for kw in ("chat", "instruct", "dpo", "zephyr", "vicuna", "wizard")
    ):
        return "chat"
    if judge_name.startswith("gpt-"):
        return "chat"
    return "base"


def _build_judge_metadata(judge_names: list[str]) -> list[dict]:
    """Build structured metadata for each judge."""
    metadata = []
    for name in judge_names:
        metadata.append(
            {
                "judge_name": name,
                "org": _ORG_MAP.get(name, ""),
                "model_type": _model_type(name),
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN", None)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print(f"Downloading {SRC_REPO} ...")
    print("=" * 60)
    ds = load_dataset(SRC_REPO, token=token)
    train = ds["train"]
    n_pairs = len(train)
    print(f"  Loaded {n_pairs} pairs")

    # Step 2: Extract judge names (sorted for determinism)
    judge_names = sorted(train[0]["preference_labels"].keys())
    n_judges = len(judge_names)
    print(f"  {n_judges} judges: {judge_names}")

    # Step 3: Build pair identifiers
    pair_ids: list[str] = []
    for i, row in enumerate(train):
        m1 = row["response_1"]["model"]
        m2 = row["response_2"]["model"]
        pair_ids.append(f"pair_{i:05d}_{m1}_vs_{m2}")

    # Step 4: Build the response matrix (judges x pairs)
    # Encode: response_1 -> 0, response_2 -> 1
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    data = torch.zeros(n_judges, n_pairs, dtype=torch.float32)
    for pair_idx, row in enumerate(train):
        labels = row["preference_labels"]
        for judge_idx, judge_name in enumerate(judge_names):
            val = labels[judge_name]
            if val == "response_2":
                data[judge_idx, pair_idx] = 1.0
            elif val == "response_1":
                data[judge_idx, pair_idx] = 0.0
            else:
                data[judge_idx, pair_idx] = float("nan")

    judge_metadata = _build_judge_metadata(judge_names)

    nan_count = torch.isnan(data).sum().item()
    print(f"  Matrix shape: {n_judges} judges x {n_pairs} pairs")
    print(f"  NaN count: {nan_count}")

    # Step 5: Build payloads
    payloads: dict[str, dict] = {}

    # 2D: judges as subjects, pairs as items
    payloads["preference_dissection/all_judges"] = {
        "data": data,
        "subject_ids": judge_names,
        "item_ids": pair_ids,
        "subject_metadata": judge_metadata,
    }

    # Crossed: pairs x judges — transpose for G-theory facet analysis
    payloads["preference_dissection/crossed"] = {
        "data": data.T.contiguous(),  # (n_pairs, n_judges)
        "pair_ids": pair_ids,
        "judge_ids": judge_names,
        "judge_metadata": judge_metadata,
    }

    # Step 6: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        shape = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {shape}, {nan_pct:.1%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
            token=token,
        )

    # Step 7: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print(f"\nDimensions:")
    print(f"  all_judges: n_judges={n_judges}, n_pairs={n_pairs}")
    print(f"  crossed:    n_pairs={n_pairs}, n_judges={n_judges}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
