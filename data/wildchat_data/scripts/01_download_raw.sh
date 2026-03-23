#!/bin/bash
# 01_download_raw.sh — Download WildChat-1M from HuggingFace.
#
# Source: https://huggingface.co/datasets/allenai/WildChat-1M
# Paper: "WildChat: 1M ChatGPT Interaction Logs in the Wild" (Zhao et al., 2024)
#
# This dataset contains ~1M real ChatGPT conversations collected via a browser extension.
# Due to the large size, we save a 100K random sample as parquet by default.
# Set WILDCHAT_FULL=1 to download the full dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

DATASET_DIR="$RAW_DIR/dataset"
SAMPLE_FILE="$RAW_DIR/wildchat_100k_sample.parquet"

echo "[INFO] Downloading WildChat-1M from HuggingFace..."

if [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "[SKIP] Dataset already exists at: $DATASET_DIR"
    exit 0
fi

if [[ -f "$SAMPLE_FILE" && "${WILDCHAT_FULL:-0}" != "1" ]]; then
    echo "[SKIP] Sample parquet already exists at: $SAMPLE_FILE"
    exit 0
fi

python3 -c "
import os
import sys

full_download = os.environ.get('WILDCHAT_FULL', '0') == '1'
raw_dir = '$RAW_DIR'
dataset_dir = '$DATASET_DIR'
sample_file = '$SAMPLE_FILE'

print('[INFO] Loading WildChat-1M from HuggingFace Hub...')
print('[INFO] This is a large dataset (~1M conversations). This may take a while.')

from datasets import load_dataset

try:
    ds = load_dataset('allenai/WildChat-1M', trust_remote_code=True)
except Exception as e:
    print(f'[WARN] Failed to load full dataset: {e}')
    print('[INFO] Trying streaming mode to get a sample...')
    ds = load_dataset('allenai/WildChat-1M', streaming=True, trust_remote_code=True)
    # Take 100K from the stream
    import pandas as pd
    records = []
    for i, example in enumerate(ds['train']):
        if i >= 100_000:
            break
        records.append(example)
        if (i + 1) % 10000 == 0:
            print(f'[INFO] Streamed {i + 1} examples...')
    df = pd.DataFrame(records)
    df.to_parquet(sample_file)
    print(f'[INFO] Saved 100K sample to {sample_file}')
    sys.exit(0)

if full_download:
    print('[INFO] Saving full dataset to disk...')
    ds.save_to_disk(dataset_dir)
    print(f'[INFO] Full dataset saved to: {dataset_dir}')
else:
    print('[INFO] Sampling 100K examples from the train split...')
    train = ds['train']
    n = len(train)
    print(f'[INFO] Full dataset has {n} examples.')
    sample_size = min(100_000, n)
    sample = train.shuffle(seed=42).select(range(sample_size))
    sample.to_parquet(sample_file)
    print(f'[INFO] Saved {sample_size} sample to: {sample_file}')
"

echo "[INFO] WildChat download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
