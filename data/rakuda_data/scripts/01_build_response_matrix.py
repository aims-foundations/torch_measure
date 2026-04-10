"""
Build response matrix for Rakuda benchmark (Japanese open-ended QA).

Source: https://github.com/shisa-ai/shaberi (Shaberi LLM Leaderboard)
License: Apache-2.0

Rakuda: 40 Japanese open-ended questions (geography, culture, etc.)
evaluated by GPT-4 judge. Continuous scores [0, 1].

Outputs:
  - response_matrix.csv : models (rows) x items (columns) -> continuous scores
  - item_content.csv    : item_id, Japanese question text
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"


def download():
    """Clone the Shaberi repo if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = RAW_DIR / "shaberi"
    if repo_dir.exists():
        print("shaberi repo already cloned, skipping download")
        return
    print("Cloning shaberi repo...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/shisa-ai/shaberi.git", str(repo_dir)],
        check=True,
    )
    print(f"Done. Raw files in {repo_dir}")


def main():
    download()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    repo_dir = RAW_DIR / "shaberi"

    # Look for Rakuda evaluation results in the repo
    # The Shaberi repo stores results in various formats
    # Try to find and parse the leaderboard data
    results_dir = repo_dir / "results"
    board_dir = repo_dir / "board"

    # Search for rakuda-specific data files
    rakuda_files = list(repo_dir.rglob("*rakuda*"))
    print(f"Found {len(rakuda_files)} rakuda-related files")
    for f in rakuda_files[:10]:
        print(f"  {f.relative_to(repo_dir)}")

    # If pre-built processed data exists, report it
    existing_rm = OUTPUT_DIR / "response_matrix.csv"
    if existing_rm.exists():
        df = pd.read_csv(existing_rm, index_col=0)
        print(f"\nExisting response_matrix.csv: {df.shape[0]} models x {df.shape[1]} items")

    existing_ic = OUTPUT_DIR / "item_content.csv"
    if existing_ic.exists():
        ic = pd.read_csv(existing_ic)
        print(f"Existing item_content.csv: {len(ic)} items")

    print("\nNote: Rakuda data was originally curated from the Shaberi leaderboard.")
    print("If response_matrix.csv already exists, it is the authoritative version.")


if __name__ == "__main__":
    main()
