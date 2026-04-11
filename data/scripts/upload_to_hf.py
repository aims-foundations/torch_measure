#!/usr/bin/env python3
"""
Convert benchmark CSVs to .pt format and upload to HuggingFace Hub.

For each benchmark, reads:
  - processed/response_matrix*.csv (one or more — models as rows, items as columns)
  - processed/item_content.csv     (optional — item_id, content columns)

Saves each response matrix as a <name>.pt file containing a dict:
  {
    "data": torch.Tensor,        # (n_models, n_items), float32
    "subject_ids": list[str],    # model names (from CSV index)
    "item_ids": list[str],       # item IDs (from CSV column headers)
    "item_contents": list[str],  # item content text (from item_content.csv)
    "subject_metadata": None,
  }

Naming convention:
  - processed/response_matrix.csv              → <benchmark>.pt
  - processed/response_matrix_instruct.csv     → <benchmark>_instruct.pt
  - processed/response_matrix_hard_complete.csv → <benchmark>_hard_complete.pt

Usage:
    python data/scripts/upload_to_hf.py              # all benchmarks
    python data/scripts/upload_to_hf.py bfcl         # single benchmark
    python data/scripts/upload_to_hf.py --no-upload  # convert only, don't upload
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent  # data/
HF_REPO = "aims-foundation/torch-measure-data"


def _load_item_content(processed_dir: Path) -> dict[str, str]:
    """Load item_content.csv as a dict: item_id -> content."""
    ic_path = processed_dir / "item_content.csv"
    if not ic_path.exists():
        return {}
    df = pd.read_csv(ic_path)
    if "item_id" not in df.columns or "content" not in df.columns:
        return {}
    df["item_id"] = df["item_id"].astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    return dict(zip(df["item_id"], df["content"]))


def _load_subject_metadata(processed_dir: Path) -> dict[str, dict] | None:
    """Load model_summary.csv as a dict: model_name -> metadata dict.

    Benchmarks may use different column names for the model ID — check
    common variants. Returns None if no summary file is found.
    """
    ms_path = processed_dir / "model_summary.csv"
    if not ms_path.exists():
        return None
    df = pd.read_csv(ms_path)

    # Find the model ID column (common variants)
    id_col = None
    for candidate in ("model", "model_name", "agent", "judge", "system"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        return None

    df[id_col] = df[id_col].astype(str)
    # Drop the ID column from the metadata dict values, keep everything else
    meta_cols = [c for c in df.columns if c != id_col]
    return {
        str(row[id_col]): {c: row[c] for c in meta_cols if pd.notna(row[c])}
        for _, row in df.iterrows()
    }


def _load_info(benchmark_dir: Path) -> dict | None:
    """Extract dataset-level metadata from the benchmark's build.py.

    Looks for a module-level ``INFO = {...}`` assignment and returns its
    literal value via :func:`ast.literal_eval`.  This avoids importing
    build.py (which would require the full set of per-benchmark
    dependencies to be installed).  Returns ``None`` if no INFO is found.
    """
    import ast

    build_py = benchmark_dir / "build.py"
    if not build_py.exists():
        return None
    try:
        tree = ast.parse(build_py.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  WARNING: failed to parse {build_py}: {e}")
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "INFO":
            continue
        try:
            data = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as e:
            print(f"  WARNING: INFO in {build_py} is not a literal: {e}")
            return None
        return data if isinstance(data, dict) else None

    return None


def _csv_to_payload(
    csv_path: Path,
    item_content: dict[str, str],
    subject_metadata: dict[str, dict] | None,
    info: dict | None,
) -> dict | None:
    """Convert a response_matrix*.csv to a .pt payload dict."""
    df = pd.read_csv(csv_path, index_col=0)

    if df.empty:
        print(f"  SKIP {csv_path.name}: empty matrix")
        return None

    # Coerce to numeric (NaN for non-numeric cells)
    df = df.apply(pd.to_numeric, errors="coerce")

    subject_ids = [str(s) for s in df.index]
    item_ids = [str(i) for i in df.columns]
    data = torch.tensor(df.values, dtype=torch.float32)

    # Look up content for each item (empty string if missing)
    item_contents = [item_content.get(iid, "") for iid in item_ids]

    # Subset metadata to just the subjects in this matrix
    payload_metadata = None
    if subject_metadata:
        payload_metadata = {
            sid: subject_metadata[sid]
            for sid in subject_ids
            if sid in subject_metadata
        } or None

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "subject_metadata": payload_metadata,
        "info": info,
    }


def _pt_name_for_csv(benchmark_name: str, csv_path: Path) -> str:
    """Derive the .pt filename from the CSV path.

    response_matrix.csv            → <benchmark>.pt
    response_matrix_instruct.csv   → <benchmark>_instruct.pt
    """
    stem = csv_path.stem  # e.g. "response_matrix" or "response_matrix_instruct"
    variant = stem.removeprefix("response_matrix").lstrip("_")
    if variant:
        return f"{benchmark_name}_{variant}.pt"
    return f"{benchmark_name}.pt"


def convert_benchmark(benchmark_dir: Path) -> list[Path]:
    """Convert all response_matrix*.csv files in a benchmark to .pt files.

    Returns the list of .pt files produced (saved next to the CSVs).
    """
    benchmark_name = benchmark_dir.name
    processed_dir = benchmark_dir / "processed"

    if not processed_dir.exists():
        return []

    csvs = sorted(processed_dir.glob("response_matrix*.csv"))
    if not csvs:
        return []

    item_content = _load_item_content(processed_dir)
    if item_content:
        print(f"  Loaded {len(item_content)} item contents")

    subject_metadata = _load_subject_metadata(processed_dir)
    if subject_metadata:
        n_cols = len(next(iter(subject_metadata.values())))
        print(f"  Loaded metadata for {len(subject_metadata)} subjects ({n_cols} fields)")

    info = _load_info(benchmark_dir)
    if info:
        print(f"  Loaded INFO from build.py ({len(info)} fields)")

    pt_files = []
    for csv_path in csvs:
        payload = _csv_to_payload(csv_path, item_content, subject_metadata, info)
        if payload is None:
            continue

        pt_name = _pt_name_for_csv(benchmark_name, csv_path)
        pt_path = processed_dir / pt_name
        torch.save(payload, pt_path)

        n_sub, n_items = payload["data"].shape
        nan_count = int(torch.isnan(payload["data"]).sum().item())
        total = n_sub * n_items
        fill_pct = (1 - nan_count / total) * 100 if total > 0 else 0
        meta_str = f", {len(payload['subject_metadata'])} with metadata" if payload["subject_metadata"] else ""
        print(f"  {pt_name}: {n_sub}x{n_items}, {fill_pct:.0f}% fill{meta_str}")
        pt_files.append(pt_path)

    return pt_files


def upload_files(pt_files: list[Path]) -> None:
    """Upload .pt files to HuggingFace Hub at the root level (no subdirectories)."""
    if not pt_files:
        return

    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError:
        print("  huggingface_hub not installed, skipping upload")
        return

    api = HfApi()
    try:
        api.repo_info(HF_REPO, repo_type="dataset")
    except Exception:
        print(f"  Creating repo {HF_REPO}...")
        api.create_repo(HF_REPO, repo_type="dataset", private=False)

    for pt_path in pt_files:
        try:
            upload_file(
                path_or_fileobj=str(pt_path),
                path_in_repo=pt_path.name,  # flat: no subdirectory prefix
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print(f"  uploaded {pt_path.name}")
        except Exception as e:
            print(f"  FAILED {pt_path.name}: {e}")


def process_benchmark(benchmark_dir: Path, upload: bool = True) -> None:
    """Convert a benchmark's CSVs to .pt and optionally upload to HF."""
    print(f"\n[{benchmark_dir.name}]")
    pt_files = convert_benchmark(benchmark_dir)
    if not pt_files:
        print("  no response_matrix*.csv files found")
        return
    if upload:
        upload_files(pt_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark CSVs to .pt and upload to HuggingFace Hub."
    )
    parser.add_argument(
        "benchmarks", nargs="*", default=None,
        help="Benchmark names to process (default: all)",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Convert only, don't upload to HuggingFace Hub",
    )
    args = parser.parse_args()

    if args.benchmarks:
        targets = [BASE_DIR / name for name in args.benchmarks]
    else:
        targets = sorted(
            d for d in BASE_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "scripts"
            and (d / "processed").exists()
        )

    for bench_dir in targets:
        if not bench_dir.exists():
            print(f"\n[{bench_dir.name}] directory not found, skipping")
            continue
        process_benchmark(bench_dir, upload=not args.no_upload)

    print("\nDone.")


if __name__ == "__main__":
    main()
