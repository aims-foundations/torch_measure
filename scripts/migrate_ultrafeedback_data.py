#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate UltraFeedback data to torch-measure-data.

Downloads the openbmb/UltraFeedback dataset from HuggingFace, builds
response matrices (LLMs x prompts), and uploads .pt files to HF Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_ultrafeedback_data.py

Source data: https://huggingface.co/datasets/openbmb/UltraFeedback
    64K prompts, each with 4 responses from different LLMs, rated by GPT-4
    on 4 aspects (helpfulness, honesty, instruction_following, truthfulness)
    with numerical scores on a 1-5 scale.

Also processes: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
    Binarized version with chosen/rejected pairs and overall scores.

Response matrices:
    - ultrafeedback/overall.pt — mean GPT-4 score across all 4 aspects,
      normalized to [0,1].  Shape: (17 models x 63,967 prompts).
    - ultrafeedback/helpfulness.pt — helpfulness aspect only
    - ultrafeedback/honesty.pt — honesty aspect only
    - ultrafeedback/instruction_following.pt — instruction-following aspect only
    - ultrafeedback/truthfulness.pt — truthfulness aspect only

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # model names
        "item_ids": list[str],             # prompt identifiers (source:index)
        "subject_metadata": list[dict],    # per-model metadata
    }
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_ultrafeedback_migration"

ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]

# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

_MODEL_METADATA: dict[str, dict] = {
    "alpaca-7b": {"org": "Stanford", "param_count": "7B"},
    "bard": {"org": "Google", "param_count": None},
    "falcon-40b-instruct": {"org": "TII", "param_count": "40B"},
    "gpt-3.5-turbo": {"org": "OpenAI", "param_count": None},
    "gpt-4": {"org": "OpenAI", "param_count": None},
    "llama-2-13b-chat": {"org": "Meta", "param_count": "13B"},
    "llama-2-70b-chat": {"org": "Meta", "param_count": "70B"},
    "llama-2-7b-chat": {"org": "Meta", "param_count": "7B"},
    "mpt-30b-chat": {"org": "MosaicML", "param_count": "30B"},
    "pythia-12b": {"org": "EleutherAI", "param_count": "12B"},
    "starchat": {"org": "HuggingFace", "param_count": "16B"},
    "ultralm-13b": {"org": "OpenBMB", "param_count": "13B"},
    "ultralm-65b": {"org": "OpenBMB", "param_count": "65B"},
    "vicuna-33b": {"org": "LMSYS", "param_count": "33B"},
    "wizardlm-13b": {"org": "Microsoft", "param_count": "13B"},
    "wizardlm-70b": {"org": "Microsoft", "param_count": "70B"},
    "wizardlm-7b": {"org": "Microsoft", "param_count": "7B"},
}


def _build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each subject (model)."""
    metadata = []
    for sid in subject_ids:
        info = _MODEL_METADATA.get(sid, {})
        metadata.append(
            {
                "model": sid,
                "org": info.get("org", ""),
                "param_count": info.get("param_count"),
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Data loading and matrix construction
# ---------------------------------------------------------------------------


def _parse_rating(rating_str: str) -> float:
    """Parse a rating string to float. Returns NaN for 'N/A' or unparseable."""
    if rating_str is None or rating_str == "N/A":
        return float("nan")
    try:
        return float(rating_str)
    except (ValueError, TypeError):
        return float("nan")


def load_ultrafeedback():
    """Load UltraFeedback dataset and build response matrices.

    Returns dict mapping name -> payload dict.
    """
    from datasets import load_dataset

    print("Loading openbmb/UltraFeedback (streaming) ...")
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)

    # First pass: collect all data
    # We'll store per-aspect scores in dicts: {(model, item_id): score}
    aspect_scores: dict[str, dict[tuple[str, str], float]] = {a: {} for a in ASPECTS}
    overall_scores: dict[tuple[str, str], float] = {}
    all_models: set[str] = set()
    all_items: list[str] = []

    count = 0
    for ex in ds:
        count += 1
        # Build item_id from source and index
        source = ex.get("source", "unknown")
        item_id = f"{source}:{count - 1}"
        all_items.append(item_id)

        for comp in ex["completions"]:
            model = comp["model"]
            all_models.add(model)

            # Extract per-aspect ratings
            annotations = comp["annotations"]
            aspect_vals = {}
            for aspect in ASPECTS:
                ann = annotations.get(aspect)
                if ann is not None:
                    rating = _parse_rating(ann.get("Rating"))
                    aspect_scores[aspect][(model, item_id)] = rating
                    aspect_vals[aspect] = rating

            # Overall = mean of available aspect ratings
            valid_vals = [v for v in aspect_vals.values() if v == v]  # filter NaN
            if valid_vals:
                overall_scores[(model, item_id)] = sum(valid_vals) / len(valid_vals)

        if count % 10000 == 0:
            print(f"  Processed {count} rows ...")

    print(f"  Total: {count} rows, {len(all_models)} unique models")

    # Sort models and items for deterministic ordering
    subject_ids = sorted(all_models)
    item_ids = all_items  # preserve original order
    model_to_idx = {m: i for i, m in enumerate(subject_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    n_subjects = len(subject_ids)
    n_items = len(item_ids)
    print(f"  Matrix shape: {n_subjects} x {n_items}")

    subject_metadata = _build_subject_metadata(subject_ids)

    payloads: dict[str, dict] = {}

    # Build overall matrix (mean across aspects, normalized to [0,1])
    print("\n--- Building ultrafeedback/overall ---")
    data = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)
    for (model, item_id), score in overall_scores.items():
        si = model_to_idx[model]
        ii = item_to_idx[item_id]
        # Normalize from 1-5 to 0-1: (score - 1) / 4
        data[si, ii] = (score - 1.0) / 4.0

    payloads["ultrafeedback/overall"] = {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
    }
    filled = (~torch.isnan(data)).sum().item()
    print(f"  Filled cells: {filled}/{n_subjects * n_items} ({filled / (n_subjects * n_items):.1%})")

    # Build per-aspect matrices
    for aspect in ASPECTS:
        name = f"ultrafeedback/{aspect}"
        print(f"\n--- Building {name} ---")
        data = torch.full((n_subjects, n_items), float("nan"), dtype=torch.float32)
        for (model, item_id), score in aspect_scores[aspect].items():
            si = model_to_idx[model]
            ii = item_to_idx[item_id]
            # Normalize from 1-5 to 0-1
            if score == score:  # not NaN
                data[si, ii] = (score - 1.0) / 4.0

        payloads[name] = {
            "data": data,
            "subject_ids": subject_ids,
            "item_ids": item_ids,
            "subject_metadata": subject_metadata,
        }
        filled = (~torch.isnan(data)).sum().item()
        print(f"  Filled cells: {filled}/{n_subjects * n_items} ({filled / (n_subjects * n_items):.1%})")

    return payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build response matrices
    print("=" * 60)
    print("Building UltraFeedback response matrices ...")
    print("=" * 60)
    payloads = load_ultrafeedback()

    # Step 2: Save and upload
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
        )

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for ultrafeedback.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
