#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate RewardBench 2 data to torch-measure-data.

Downloads per-judge binary results from allenai/reward-bench-2-results
on HuggingFace Hub, pivots into response matrices (judges x items),
and uploads .pt files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with write access to torch-measure-data
    python scripts/migrate_rewardbench2_data.py

Source data:
    - Items: allenai/reward-bench-2 (1,865 prompts with chosen/rejected)
    - Results: allenai/reward-bench-2-results (per-judge binary outcomes)

RewardBench 2 evaluates reward models on harder preference triples across
6 domains: Factuality, Focus, Math, Precise IF, Safety, Ties.  Each item
is a (prompt, chosen, rejected) triple; each judge produces a binary
pass/fail (1.0 = correctly ranked chosen > rejected, 0.0 = failed).

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32
        "subject_ids": list[str],          # judge (reward model) names
        "item_ids": list[str],             # item id strings
        "subject_metadata": list[dict],    # judge metadata (model_type, org)
        "item_metadata": list[dict],       # per-item metadata (subset/domain)
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_RESULTS_REPO = "allenai/reward-bench-2-results"
SRC_ITEMS_REPO = "allenai/reward-bench-2"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path(tempfile.gettempdir()) / "torch_measure_rewardbench2_migration"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Domains in canonical order.
DOMAINS = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties"]

# Registry-friendly names for per-domain splits.
DOMAIN_NAME_MAP = {
    "Factuality": "factuality",
    "Focus": "focus",
    "Math": "math",
    "Precise IF": "precise_if",
    "Safety": "safety",
    "Ties": "ties",
}

# ---------------------------------------------------------------------------
# Subject metadata helpers
# ---------------------------------------------------------------------------

_ORG_MAP: dict[str, str] = {
    "Skywork": "Skywork",
    "allenai": "Allen AI",
    "anthropic": "Anthropic",
    "google": "Google",
    "openai": "OpenAI",
    "internlm": "InternLM",
    "openbmb": "OpenBMB",
    "PKU-Alignment": "PKU",
    "Ray2333": "Ray2333",
    "Nexusflow": "Nexusflow",
    "OpenAssistant": "OpenAssistant",
    "NCSOFT": "NCSOFT",
    "CIR-AMS": "CIR-AMS",
    "HFXM": "HFXM",
    "LxzGordon": "LxzGordon",
    "Qwen": "Alibaba",
    "ShikaiChen": "ShikaiChen",
    "sfairXC": "sfairXC",
    "weqweasdas": "weqweasdas",
    "hendrydong": "hendrydong",
    "infly": "infly",
    "nicolinho": "nicolinho",
    "ContextualAI": "ContextualAI",
    "Databricks-Mosaic-Research": "Databricks",
    "RLHFlow": "RLHFlow",
    "Schrieffer": "Schrieffer",
}


def _infer_org(path: str) -> str:
    """Infer organization from eval-set file path like 'eval-set/openai/gpt-4o.json'."""
    parts = path.split("/")
    if len(parts) >= 2:
        org_key = parts[1]  # e.g., "openai", "anthropic"
        return _ORG_MAP.get(org_key, org_key)
    return ""


def _model_name_from_path(path: str) -> str:
    """Extract model name from path like 'eval-set/openai/gpt-4o.json'."""
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[1] + "/" + parts[2].replace(".json", "")
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def list_result_files(prefix: str = "eval-set/") -> list[str]:
    """List all JSON result files under the given prefix."""
    api = HfApi(token=HF_TOKEN)
    files = list(
        api.list_repo_tree(SRC_RESULTS_REPO, repo_type="dataset", recursive=True)
    )
    return sorted(
        f.path
        for f in files
        if hasattr(f, "size") and f.path.startswith(prefix) and f.path.endswith(".json")
    )


def download_summary_results(file_paths: list[str]) -> dict[str, dict]:
    """Download summary result JSONs (domain-level accuracy per judge).

    Returns {model_name: {domain: accuracy, ..., 'model_type': ..., 'model': ...}}.
    """
    results = {}
    for i, fpath in enumerate(file_paths):
        try:
            local = hf_hub_download(
                SRC_RESULTS_REPO, fpath, repo_type="dataset", token=HF_TOKEN
            )
            with open(local) as f:
                data = json.load(f)
            model_name = _model_name_from_path(fpath)
            results[model_name] = data
            results[model_name]["_path"] = fpath
        except Exception as e:
            print(f"  Warning: failed to download {fpath}: {e}")
        if (i + 1) % 50 == 0:
            print(f"    Downloaded {i + 1}/{len(file_paths)} summary files")
    return results


def download_score_results(file_paths: list[str]) -> dict[str, dict]:
    """Download per-item score JSONs (binary results per item per judge).

    Returns {model_name: {'results': [...], 'id': [...], 'subset': [...], ...}}.
    """
    results = {}
    for i, fpath in enumerate(file_paths):
        try:
            local = hf_hub_download(
                SRC_RESULTS_REPO, fpath, repo_type="dataset", token=HF_TOKEN
            )
            with open(local) as f:
                data = json.load(f)
            # Map eval-set-scores path to model name
            model_name = _model_name_from_path(fpath.replace("eval-set-scores/", "eval-set/"))
            results[model_name] = data
            results[model_name]["_path"] = fpath
        except Exception as e:
            print(f"  Warning: failed to download {fpath}: {e}")
        if (i + 1) % 50 == 0:
            print(f"    Downloaded {i + 1}/{len(file_paths)} score files")
    return results


def load_item_metadata() -> dict[str, dict]:
    """Load item metadata from the main RewardBench 2 dataset."""
    from datasets import load_dataset

    ds = load_dataset(SRC_ITEMS_REPO, split="test", token=HF_TOKEN)
    meta = {}
    for item in ds:
        meta[str(item["id"])] = {
            "subset": item["subset"],
            "num_correct": item["num_correct"],
            "num_incorrect": item["num_incorrect"],
            "total_completions": item["total_completions"],
        }
    return meta


# ---------------------------------------------------------------------------
# Pivot & payload building
# ---------------------------------------------------------------------------


def build_response_matrix(
    score_results: dict[str, dict],
    summary_results: dict[str, dict],
    item_ids: list[str] | None = None,
    subset_filter: str | None = None,
) -> dict:
    """Build a judges x items binary response matrix from per-item score data.

    Parameters
    ----------
    score_results : dict
        Per-item score data keyed by model name.
    summary_results : dict
        Summary data keyed by model name (for metadata).
    item_ids : list[str] or None
        Ordered item IDs. If None, inferred from first judge.
    subset_filter : str or None
        If provided, only include items in this domain/subset.

    Returns
    -------
    dict with keys: data, subject_ids, item_ids, subject_metadata, item_metadata.
    """
    # Determine item ordering from first available judge
    if item_ids is None:
        first_judge = next(iter(score_results.values()))
        item_ids = [str(x) for x in first_judge["id"]]

    # Build id-to-subset mapping from score data
    first_judge = next(iter(score_results.values()))
    id_to_subset = {}
    for idx, iid in enumerate(first_judge["id"]):
        id_to_subset[str(iid)] = first_judge["subset"][idx]

    # Apply subset filter
    if subset_filter is not None:
        item_ids = [iid for iid in item_ids if id_to_subset.get(iid) == subset_filter]

    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Sort judges alphabetically
    judge_names = sorted(score_results.keys())
    n_judges = len(judge_names)

    # Build the matrix
    data = torch.full((n_judges, n_items), float("nan"), dtype=torch.float32)

    for j, judge_name in enumerate(judge_names):
        judge_data = score_results[judge_name]
        judge_ids = [str(x) for x in judge_data["id"]]
        judge_results = judge_data["results"]

        for k, iid in enumerate(judge_ids):
            if iid in item_id_to_idx and judge_results[k] is not None:
                data[j, item_id_to_idx[iid]] = float(judge_results[k])

    # Build subject metadata
    subject_metadata = []
    for judge_name in judge_names:
        summary = summary_results.get(judge_name, {})
        path = score_results[judge_name].get("_path", "")
        subject_metadata.append(
            {
                "model": summary.get("model", judge_name),
                "org": _infer_org(path.replace("eval-set-scores/", "eval-set/")),
                "model_type": summary.get("model_type", ""),
            }
        )

    # Build item metadata
    item_metadata = []
    for iid in item_ids:
        item_metadata.append({"subset": id_to_subset.get(iid, "")})

    return {
        "data": data,
        "subject_ids": judge_names,
        "item_ids": item_ids,
        "subject_metadata": subject_metadata,
        "item_metadata": item_metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: List and download result files
    print("=" * 60)
    print("Listing result files from allenai/reward-bench-2-results ...")
    print("=" * 60)

    summary_paths = list_result_files("eval-set/")
    score_paths = list_result_files("eval-set-scores/")
    print(f"  Found {len(summary_paths)} summary files (eval-set/)")
    print(f"  Found {len(score_paths)} per-item score files (eval-set-scores/)")

    print("\nDownloading summary results ...")
    summary_results = download_summary_results(summary_paths)
    print(f"  Downloaded {len(summary_results)} summary results")

    print("\nDownloading per-item score results ...")
    score_results = download_score_results(score_paths)
    print(f"  Downloaded {len(score_results)} per-item score results")

    # Step 2: Load item metadata from main dataset
    print("\n" + "=" * 60)
    print("Loading item metadata from allenai/reward-bench-2 ...")
    print("=" * 60)
    item_meta = load_item_metadata()
    print(f"  Loaded metadata for {len(item_meta)} items")

    # Verify item IDs align
    first_judge = next(iter(score_results.values()))
    item_ids = [str(x) for x in first_judge["id"]]
    print(f"  Items from results: {len(item_ids)}")

    # Step 3: Build response matrices
    print("\n" + "=" * 60)
    print("Building response matrices ...")
    print("=" * 60)

    payloads: dict[str, dict] = {}

    # --- All domains ---
    print("\n--- rewardbench2/all ---")
    payloads["rewardbench2/all"] = build_response_matrix(
        score_results, summary_results, item_ids=item_ids
    )
    n_s, n_i = payloads["rewardbench2/all"]["data"].shape
    print(f"  {n_s} judges x {n_i} items")

    # --- Per-domain splits ---
    for domain in DOMAINS:
        short = DOMAIN_NAME_MAP[domain]
        name = f"rewardbench2/{short}"
        print(f"\n--- {name} ---")
        payloads[name] = build_response_matrix(
            score_results, summary_results, item_ids=item_ids, subset_filter=domain
        )
        n_s, n_i = payloads[name]["data"].shape
        print(f"  {n_s} judges x {n_i} items")

    # Step 4: Save and upload
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

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source items: {SRC_ITEMS_REPO}")
    print(f"  Source results: {SRC_RESULTS_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for rewardbench2.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
