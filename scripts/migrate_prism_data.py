#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate PRISM alignment data to torch-measure-data.

Downloads the PRISM alignment dataset (utterances config) from HuggingFace
Hub, builds response matrices (participants x utterances), and uploads .pt
files to the torch-measure-data repository.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_prism_data.py

Source data: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
    1,500+ participants from 75 countries rate LLM responses on a 1-100
    cardinal scale.  The ``utterances`` config contains ~68K individual
    (user_prompt, model_response, score, if_chosen) records.

Reference:
    Kirk et al., "The PRISM Alignment Dataset", arXiv:2404.16019, 2024.

Destination .pt files:

  prism/scores.pt -- participants x utterances, continuous [0, 1]:
    {
        "data": torch.Tensor,             # (n_participants, n_utterances), float32
        "subject_ids": list[str],          # participant (user) IDs
        "item_ids": list[str],             # utterance IDs
        "subject_metadata": list[dict],    # per-participant metadata
        "item_metadata": list[dict],       # per-utterance metadata
    }

  prism/chosen.pt -- participants x utterances, binary 0/1:
    Same structure, with binary chosen/not-chosen values.

  prism/crossed_scores.pt -- utterances x participants (transposed):
    For G-theory facet analysis.

  prism/crossed_chosen.pt -- utterances x participants (transposed):
    For G-theory facet analysis.
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

SRC_REPO = "HannahRoseKirk/prism-alignment"
DST_REPO = "sangttruong/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_prism_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from datasets import load_dataset

    token = HF_TOKEN or None

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download utterances config
    print("=" * 60)
    print(f"Downloading {SRC_REPO} (utterances config) ...")
    print("=" * 60)
    ds = load_dataset(SRC_REPO, "utterances", split="train", token=token)
    print(f"  Loaded {len(ds)} utterance records")
    print(f"  Columns: {ds.column_names}")

    # Step 2: Extract unique participants and utterances
    print("\n" + "=" * 60)
    print("Extracting participants and utterance IDs ...")
    print("=" * 60)

    user_ids_all = ds["user_id"]
    utterance_ids_all = ds["utterance_id"]
    scores_all = ds["score"]
    chosen_all = ds["if_chosen"]

    # Build sorted unique lists for determinism
    participant_ids = sorted(set(user_ids_all))
    utterance_ids = sorted(set(utterance_ids_all))

    n_participants = len(participant_ids)
    n_utterances = len(utterance_ids)

    print(f"  Unique participants: {n_participants}")
    print(f"  Unique utterances:   {n_utterances}")

    participant_to_idx = {pid: idx for idx, pid in enumerate(participant_ids)}
    utterance_to_idx = {uid: idx for idx, uid in enumerate(utterance_ids)}

    # Step 3: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    scores_matrix = torch.full(
        (n_participants, n_utterances), float("nan"), dtype=torch.float32
    )
    chosen_matrix = torch.full(
        (n_participants, n_utterances), float("nan"), dtype=torch.float32
    )

    for i in range(len(ds)):
        uid = user_ids_all[i]
        utt_id = utterance_ids_all[i]
        score = scores_all[i]
        chosen = chosen_all[i]

        p_idx = participant_to_idx[uid]
        u_idx = utterance_to_idx[utt_id]

        # Normalize score from 1-100 to [0, 1]
        if score is not None:
            scores_matrix[p_idx, u_idx] = (float(score) - 1.0) / 99.0

        # Binary chosen indicator
        if chosen is not None:
            chosen_matrix[p_idx, u_idx] = 1.0 if chosen else 0.0

        if (i + 1) % 20000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} records")

    scores_nan_pct = torch.isnan(scores_matrix).float().mean().item()
    chosen_nan_pct = torch.isnan(chosen_matrix).float().mean().item()
    print(f"\n  Scores matrix: {n_participants} x {n_utterances}, "
          f"{scores_nan_pct:.1%} missing")
    print(f"  Chosen matrix: {n_participants} x {n_utterances}, "
          f"{chosen_nan_pct:.1%} missing")

    # Step 4: Build metadata
    print("\n" + "=" * 60)
    print("Building metadata ...")
    print("=" * 60)

    # Per-utterance metadata from the utterances dataset
    # Build a lookup: utterance_id -> metadata
    utt_meta_lookup: dict[str, dict] = {}
    model_names = ds["model_name"]
    model_providers = ds["model_provider"]
    user_prompts = ds["user_prompt"]
    conversation_ids = ds["conversation_id"]
    turns = ds["turn"]
    within_turn_ids = ds["within_turn_id"]

    for i in range(len(ds)):
        utt_id = utterance_ids_all[i]
        if utt_id not in utt_meta_lookup:
            utt_meta_lookup[utt_id] = {
                "model_name": model_names[i] if model_names[i] else "",
                "model_provider": model_providers[i] if model_providers[i] else "",
                "conversation_id": conversation_ids[i] if conversation_ids[i] else "",
                "turn": turns[i] if turns[i] is not None else -1,
                "within_turn_id": within_turn_ids[i] if within_turn_ids[i] is not None else -1,
            }

    item_metadata = [utt_meta_lookup.get(uid, {}) for uid in utterance_ids]

    # Per-participant metadata: try to load survey config
    subject_metadata: list[dict] = []
    try:
        print("  Loading survey config for participant metadata ...")
        survey = load_dataset(SRC_REPO, "survey", split="train", token=token)
        survey_lookup: dict[str, dict] = {}
        for row in survey:
            survey_lookup[row["user_id"]] = {
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
                "birth_country": row.get("birth_country", ""),
                "reside_country": row.get("reside_country", ""),
                "education": row.get("education", ""),
                "english_proficiency": row.get("english_proficiency", ""),
                "lm_familiarity": row.get("lm_familiarity", ""),
            }
        subject_metadata = [survey_lookup.get(pid, {}) for pid in participant_ids]
        n_matched = sum(1 for m in subject_metadata if m)
        print(f"  Matched survey data for {n_matched}/{n_participants} participants")
    except Exception as e:
        print(f"  Warning: could not load survey config: {e}")
        subject_metadata = [{} for _ in participant_ids]

    # Step 5: Build payloads
    print("\n" + "=" * 60)
    print("Building payloads ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # participants x utterances (continuous scores)
    payloads["prism/scores"] = {
        "data": scores_matrix,
        "subject_ids": participant_ids,
        "item_ids": utterance_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }

    # participants x utterances (binary chosen)
    payloads["prism/chosen"] = {
        "data": chosen_matrix,
        "subject_ids": participant_ids,
        "item_ids": utterance_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }

    # crossed: utterances x participants (continuous)
    payloads["prism/crossed_scores"] = {
        "data": scores_matrix.T.contiguous(),
        "subject_ids": utterance_ids,
        "item_ids": participant_ids,
        "subject_metadata": item_metadata,
        "item_metadata": subject_metadata,
    }

    # crossed: utterances x participants (binary)
    payloads["prism/crossed_chosen"] = {
        "data": chosen_matrix.T.contiguous(),
        "subject_ids": utterance_ids,
        "item_ids": participant_ids,
        "subject_metadata": item_metadata,
        "item_metadata": subject_metadata,
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
    print(f"\nDimensions (for prism.py registry):")
    for name, payload in sorted(payloads.items()):
        n_s, n_i = payload["data"].shape
        print(f"  {name}: n_subjects={n_s}, n_items={n_i}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
