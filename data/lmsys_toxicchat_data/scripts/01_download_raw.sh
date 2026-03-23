#!/bin/bash
# 01_download_raw.sh — Download LMSYS ToxicChat from HuggingFace.
#
# Source: https://huggingface.co/datasets/lmsys/toxic-chat
# Paper: "ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World
#         User-AI Conversation" (Lin et al., 2023)
#
# Uses the toxicchat0124 config (January 2024 version).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

DATASET_DIR="$RAW_DIR/dataset"

echo "[INFO] Downloading LMSYS ToxicChat (toxicchat0124) from HuggingFace..."

if [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "[SKIP] Dataset already exists at: $DATASET_DIR"
    exit 0
fi

python3 -c "
from datasets import load_dataset

print('[INFO] Loading lmsys/toxic-chat (toxicchat0124)...')
ds = load_dataset('lmsys/toxic-chat', 'toxicchat0124')

print('[INFO] Dataset splits:')
for split_name, split_ds in ds.items():
    print(f'  {split_name}: {len(split_ds)} examples')

dataset_dir = '$DATASET_DIR'
print(f'[INFO] Saving to {dataset_dir}...')
ds.save_to_disk(dataset_dir)
print('[INFO] Done.')
"

echo "[INFO] ToxicChat download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
