#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate PRM800K data to torch-measure-data.

Downloads step-level human labels from the openai/prm800k GitHub repo,
builds step-level response matrices (solutions x steps), and uploads
.pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_prm800k_data.py

Source data:
    - https://github.com/openai/prm800k (phase 1 & phase 2 JSONL files)
    - ~800K step-level correctness labels for model-generated MATH solutions

PRM800K provides step-level human annotations (+1 correct, -1 incorrect,
0 neutral) for model-generated solutions to MATH problems.  Each solution
is a sequence of steps; labelers annotated each step's correctness.

Step-level labels are mapped to binary:
    +1 (correct)   -> 1.0
    -1 (incorrect)  -> 0.0
    0  (neutral)    -> NaN
    None (unlabeled) -> NaN

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_solutions, max_steps), float32
        "subject_ids": list[str],          # solution identifiers
        "item_ids": list[str],             # step position names
        "subject_metadata": list[dict],    # per-solution metadata
        "item_metadata": list[dict],       # per-step metadata (position)
    }
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from huggingface_hub import upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO_URL = "https://github.com/openai/prm800k.git"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_prm800k_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Data file paths relative to cloned repo root.
PHASE1_TRAIN = "prm800k/data/phase1_train.jsonl"
PHASE1_TEST = "prm800k/data/phase1_test.jsonl"
PHASE2_TRAIN = "prm800k/data/phase2_train.jsonl"
PHASE2_TEST = "prm800k/data/phase2_test.jsonl"
MATH_TRAIN = "prm800k/math_splits/train.jsonl"
MATH_TEST = "prm800k/math_splits/test.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def clone_repo(dest: Path) -> Path:
    """Clone the PRM800K repository (with LFS) to *dest*."""
    if dest.exists() and (dest / "prm800k" / "data").exists():
        print(f"  Repository already cloned at {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning {SRC_REPO_URL} into {dest} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SRC_REPO_URL, str(dest)],
        check=True,
    )
    # Ensure LFS files are pulled
    subprocess.run(
        ["git", "-C", str(dest), "lfs", "pull"],
        check=True,
    )
    return dest


def load_math_metadata(repo_dir: Path) -> dict[str, dict]:
    """Load MATH problem metadata (subject, level) keyed by problem text."""
    meta: dict[str, dict] = {}
    for split_file in [MATH_TRAIN, MATH_TEST]:
        path = repo_dir / split_file
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                meta[data["problem"]] = {
                    "subject": data.get("subject", ""),
                    "level": data.get("level", 0),
                    "unique_id": data.get("unique_id", ""),
                    "answer": data.get("answer", ""),
                }
    return meta


def load_solutions(repo_dir: Path, file_path: str) -> list[dict]:
    """Load solution records from a JSONL file.

    Returns a list of dicts with keys:
        problem, steps (list of ratings), finish_reason, phase, n_steps,
        ground_truth_answer, pre_generated_answer.
    """
    solutions = []
    full_path = repo_dir / file_path
    if not full_path.exists():
        print(f"  Warning: {full_path} not found, skipping.")
        return solutions

    with open(full_path) as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            label = data["label"]
            steps = label["steps"]

            # Extract the chosen rating for each step.
            ratings: list[float | None] = []
            for step in steps:
                chosen_idx = step.get("chosen_completion")
                if step.get("human_completion") is not None:
                    # Phase 1: human-written step -> treat as correct.
                    ratings.append(1.0)
                elif (
                    chosen_idx is not None
                    and 0 <= chosen_idx < len(step["completions"])
                ):
                    raw = step["completions"][chosen_idx]["rating"]
                    if raw == 1:
                        ratings.append(1.0)
                    elif raw == -1:
                        ratings.append(0.0)
                    else:
                        # 0 (neutral) or None -> NaN
                        ratings.append(None)
                else:
                    ratings.append(None)

            solutions.append(
                {
                    "problem": question["problem"],
                    "ratings": ratings,
                    "finish_reason": label["finish_reason"],
                    "n_steps": len(steps),
                    "ground_truth_answer": question.get(
                        "ground_truth_answer", ""
                    ),
                    "pre_generated_answer": question.get(
                        "pre_generated_answer", ""
                    ),
                }
            )
    return solutions


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_step_level_matrix(
    solutions: list[dict],
    math_meta: dict[str, dict],
    phase_label: str,
) -> dict:
    """Build a step-level response matrix from solution records.

    Parameters
    ----------
    solutions : list[dict]
        Solution records with ratings lists.
    math_meta : dict
        MATH problem metadata keyed by problem text.
    phase_label : str
        Label for the phase/split (used in subject IDs).

    Returns
    -------
    dict
        Payload with data, subject_ids, item_ids, subject_metadata,
        item_metadata.
    """
    n_solutions = len(solutions)
    max_steps = max(len(s["ratings"]) for s in solutions) if solutions else 0

    # Build the matrix.
    data = torch.full((n_solutions, max_steps), float("nan"), dtype=torch.float32)

    subject_ids: list[str] = []
    subject_metadata: list[dict] = []

    for i, sol in enumerate(solutions):
        sol_id = f"{phase_label}_{i:06d}"
        subject_ids.append(sol_id)

        # Fill in step ratings.
        for j, rating in enumerate(sol["ratings"]):
            if rating is not None:
                data[i, j] = rating

        # Metadata.
        problem_meta = math_meta.get(sol["problem"], {})
        subject_metadata.append(
            {
                "problem": sol["problem"][:200],  # Truncate for storage.
                "finish_reason": sol["finish_reason"],
                "n_steps": sol["n_steps"],
                "ground_truth_answer": sol["ground_truth_answer"],
                "pre_generated_answer": sol["pre_generated_answer"],
                "math_subject": problem_meta.get("subject", ""),
                "math_level": problem_meta.get("level", 0),
            }
        )

    # Item IDs and metadata are step positions.
    item_ids = [f"step_{j}" for j in range(max_steps)]
    item_metadata = [{"position": j} for j in range(max_steps)]

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
    repo_dir = TMP_DIR / "prm800k_repo"

    # Step 1: Clone repository.
    print("=" * 60)
    print("Step 1: Cloning openai/prm800k repository ...")
    print("=" * 60)
    clone_repo(repo_dir)

    # Step 2: Load MATH metadata.
    print("\n" + "=" * 60)
    print("Step 2: Loading MATH problem metadata ...")
    print("=" * 60)
    math_meta = load_math_metadata(repo_dir)
    print(f"  Loaded metadata for {len(math_meta)} MATH problems")

    # Step 3: Load all solution files.
    print("\n" + "=" * 60)
    print("Step 3: Loading solution files ...")
    print("=" * 60)

    p1_train = load_solutions(repo_dir, PHASE1_TRAIN)
    print(f"  Phase 1 train: {len(p1_train)} solutions")

    p1_test = load_solutions(repo_dir, PHASE1_TEST)
    print(f"  Phase 1 test:  {len(p1_test)} solutions")

    p2_train = load_solutions(repo_dir, PHASE2_TRAIN)
    print(f"  Phase 2 train: {len(p2_train)} solutions")

    p2_test = load_solutions(repo_dir, PHASE2_TEST)
    print(f"  Phase 2 test:  {len(p2_test)} solutions")

    total = len(p1_train) + len(p1_test) + len(p2_train) + len(p2_test)
    print(f"  Total: {total} solutions")

    # Step 4: Build response matrices.
    print("\n" + "=" * 60)
    print("Step 4: Building response matrices ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # --- Train (phase 1 train + phase 2 train) ---
    train_solutions = p1_train + p2_train
    print(f"\n--- prm800k/train ({len(train_solutions)} solutions) ---")
    payloads["prm800k/train"] = build_step_level_matrix(
        train_solutions, math_meta, phase_label="train"
    )
    n_s, n_i = payloads["prm800k/train"]["data"].shape
    print(f"  {n_s} solutions x {n_i} steps")

    # --- Test (phase 1 test + phase 2 test) ---
    test_solutions = p1_test + p2_test
    print(f"\n--- prm800k/test ({len(test_solutions)} solutions) ---")
    payloads["prm800k/test"] = build_step_level_matrix(
        test_solutions, math_meta, phase_label="test"
    )
    n_s, n_i = payloads["prm800k/test"]["data"].shape
    print(f"  {n_s} solutions x {n_i} steps")

    # --- All (combined) ---
    all_solutions = p1_train + p1_test + p2_train + p2_test
    print(f"\n--- prm800k/all ({len(all_solutions)} solutions) ---")
    payloads["prm800k/all"] = build_step_level_matrix(
        all_solutions, math_meta, phase_label="all"
    )
    n_s, n_i = payloads["prm800k/all"]["data"].shape
    print(f"  {n_s} solutions x {n_i} steps")

    # --- Phase 2 only ---
    phase2_solutions = p2_train + p2_test
    print(f"\n--- prm800k/phase2 ({len(phase2_solutions)} solutions) ---")
    payloads["prm800k/phase2"] = build_step_level_matrix(
        phase2_solutions, math_meta, phase_label="phase2"
    )
    n_s, n_i = payloads["prm800k/phase2"]["data"].shape
    print(f"  {n_s} solutions x {n_i} steps")

    # Step 5: Save and upload.
    print("\n" + "=" * 60)
    print("Step 5: Saving and uploading ...")
    print("=" * 60)

    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        n_pos = (payload["data"] == 1.0).sum().item()
        n_neg = (payload["data"] == 0.0).sum().item()
        n_valid = n_pos + n_neg
        print(
            f"  {filename}: {n_sub} x {n_items}, "
            f"{nan_pct:.1%} NaN, "
            f"{n_valid} valid ({n_pos} pos, {n_neg} neg)"
        )

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )

    # Step 6: Summary.
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO_URL}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for prm800k.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
