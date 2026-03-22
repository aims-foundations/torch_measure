#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate PersonalLLM data to torch-measure-data.

Downloads the PersonalLLM dataset from namkoong-lab/PersonalLLM on
HuggingFace Hub, pivots into response matrices (reward_models x
prompt-response pairs), and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_personalllm_data.py

Source data:
    - namkoong-lab/PersonalLLM on HuggingFace Hub
    - 10,402 prompts (9,402 train + 1,000 test), each with 8 LLM responses
    - 10 reward models score every response

PersonalLLM studies personalized preference modeling.  Each reward model
acts as a proxy for a user with distinct preferences.  The response matrix
captures preference heterogeneity across these simulated users.

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # reward model names
        "item_ids": list[str],             # "promptID_responseIdx" identifiers
        "subject_metadata": list[dict],    # reward model metadata
        "item_metadata": list[dict],       # per-item metadata (prompt_id, response_model, subset)
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

SRC_REPO = "namkoong-lab/PersonalLLM"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_personalllm_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Reward model short names (column suffixes) -> full model names.
REWARD_MODEL_MAP = {
    "gemma_2b": "weqweasdas/RM-Gemma-2B",
    "gemma_7b": "weqweasdas/RM-Gemma-7B",
    "mistral_raft": "hendrydong/Mistral-RM-for-RAFT-GSHF-v0",
    "mistral_ray": "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback",
    "mistral_weqweasdas": "weqweasdas/RM-Mistral-7B",
    "llama3_sfairx": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
    "oasst_deberta_v3": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "beaver_7b": "PKU-Alignment/beaver-7b-v1.0-cost",
    "oasst_pythia_7b": "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
    "oasst_pythia_1b": "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
}

# Canonical ordering of reward model short names.
REWARD_MODEL_KEYS = sorted(REWARD_MODEL_MAP.keys())

N_RESPONSES = 8  # responses per prompt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset_splits() -> dict:
    """Load PersonalLLM train and test splits from HuggingFace Hub.

    Returns
    -------
    dict with keys "train" and "test", each a HuggingFace Dataset object.
    """
    from datasets import load_dataset

    print("Loading PersonalLLM dataset from HuggingFace Hub ...")
    ds = load_dataset(SRC_REPO, token=HF_TOKEN if HF_TOKEN else None)
    print(f"  Train: {len(ds['train'])} rows")
    print(f"  Test:  {len(ds['test'])} rows")
    return ds


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def build_response_matrix(dataset) -> dict:
    """Build a reward_models x prompt-response-pairs matrix from a dataset split.

    Parameters
    ----------
    dataset : HuggingFace Dataset
        A single split (train or test) of the PersonalLLM dataset.

    Returns
    -------
    dict with keys: data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    n_prompts = len(dataset)
    n_items = n_prompts * N_RESPONSES
    n_subjects = len(REWARD_MODEL_KEYS)

    print(f"  Building matrix: {n_subjects} reward models x {n_items} items "
          f"({n_prompts} prompts x {N_RESPONSES} responses)")

    # Initialize matrix with NaN
    data = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)

    # Build item_ids and item_metadata
    item_ids: list[str] = []
    item_metadata: list[dict] = []

    for row_idx, row in enumerate(dataset):
        prompt_id = row["prompt_id"]
        subset = row.get("subset", "")

        for resp_idx in range(1, N_RESPONSES + 1):
            item_col_idx = row_idx * N_RESPONSES + (resp_idx - 1)

            # Item ID: "promptID_responseIdx"
            item_ids.append(f"{prompt_id}_{resp_idx}")

            # Response model name
            model_col = f"response_{resp_idx}_model"
            response_model = row.get(model_col, "")

            item_metadata.append({
                "prompt_id": prompt_id,
                "response_idx": resp_idx,
                "response_model": response_model,
                "subset": subset,
            })

            # Fill in reward scores for each reward model
            for rm_idx, rm_key in enumerate(REWARD_MODEL_KEYS):
                score_col = f"response_{resp_idx}_{rm_key}"
                score = row.get(score_col)
                if score is not None:
                    data[rm_idx, item_col_idx] = float(score)

        if (row_idx + 1) % 2000 == 0:
            print(f"    Processed {row_idx + 1}/{n_prompts} prompts")

    # Build subject metadata
    subject_ids = [REWARD_MODEL_MAP[k] for k in REWARD_MODEL_KEYS]
    subject_metadata = [
        {"short_name": k, "full_name": REWARD_MODEL_MAP[k]}
        for k in REWARD_MODEL_KEYS
    ]

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset
    print("=" * 60)
    print("Loading PersonalLLM dataset ...")
    print("=" * 60)
    ds = load_dataset_splits()

    # Step 2: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # --- Train split ---
    print("\n--- personalllm/train ---")
    payloads["personalllm/train"] = build_response_matrix(ds["train"])
    n_s, n_i = payloads["personalllm/train"]["data"].shape
    print(f"  Shape: {n_s} x {n_i}")

    # --- Test split ---
    print("\n--- personalllm/test ---")
    payloads["personalllm/test"] = build_response_matrix(ds["test"])
    n_s, n_i = payloads["personalllm/test"]["data"].shape
    print(f"  Shape: {n_s} x {n_i}")

    # --- All (train + test combined) ---
    print("\n--- personalllm/all ---")
    from datasets import concatenate_datasets

    combined = concatenate_datasets([ds["train"], ds["test"]])
    payloads["personalllm/all"] = build_response_matrix(combined)
    n_s, n_i = payloads["personalllm/all"]["data"].shape
    print(f"  Shape: {n_s} x {n_i}")

    # Step 3: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading ...")
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
            token=HF_TOKEN,
        )

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for personalllm.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
