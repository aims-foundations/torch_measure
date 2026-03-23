#!/bin/bash
# 01_download_raw.sh — Download PRISM alignment dataset from HuggingFace.
#
# Source: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
# Paper: "PRISM: Mapping the Diversity of Human Preferences for AI Alignment"
#        (Kirk et al., 2024)
#
# Contains diverse human preference data across demographics and value systems
# for studying alignment pluralism.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

DATASET_DIR="$RAW_DIR/dataset"

echo "[INFO] Downloading PRISM alignment dataset from HuggingFace..."

if [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "[SKIP] Dataset already exists at: $DATASET_DIR"
    exit 0
fi

python3 -c "
from datasets import load_dataset

print('[INFO] Loading HannahRoseKirk/prism-alignment...')
ds = load_dataset('HannahRoseKirk/prism-alignment')

print('[INFO] Dataset splits:')
for split_name, split_ds in ds.items():
    print(f'  {split_name}: {len(split_ds)} examples')

dataset_dir = '$DATASET_DIR'
print(f'[INFO] Saving to {dataset_dir}...')
ds.save_to_disk(dataset_dir)
print('[INFO] Done.')
"

echo "[INFO] PRISM download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
