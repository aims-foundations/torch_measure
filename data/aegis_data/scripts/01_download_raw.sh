#!/bin/bash
# 01_download_raw.sh — Download NVIDIA Aegis 2.0 AI Content Safety Dataset from HuggingFace.
#
# Source: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
# Paper: "Aegis: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts"
#        (Ghosh et al., NVIDIA, 2024)
#
# Contains human-annotated examples across multiple content safety categories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

DATASET_DIR="$RAW_DIR/dataset"

echo "[INFO] Downloading NVIDIA Aegis 2.0 from HuggingFace..."

if [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "[SKIP] Dataset already exists at: $DATASET_DIR"
    exit 0
fi

python3 -c "
from datasets import load_dataset

print('[INFO] Loading nvidia/Aegis-AI-Content-Safety-Dataset-2.0...')
ds = load_dataset('nvidia/Aegis-AI-Content-Safety-Dataset-2.0')

print('[INFO] Dataset splits:')
for split_name, split_ds in ds.items():
    print(f'  {split_name}: {len(split_ds)} examples')

dataset_dir = '$DATASET_DIR'
print(f'[INFO] Saving to {dataset_dir}...')
ds.save_to_disk(dataset_dir)
print('[INFO] Done.')
"

echo "[INFO] Aegis 2.0 download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
