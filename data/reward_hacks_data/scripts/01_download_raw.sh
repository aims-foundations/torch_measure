#!/bin/bash
set -euo pipefail

# School of Reward Hacks: Dataset of reward hacking behaviors in LLM agents.
# Source: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ -d "$RAW_DIR/school-of-reward-hacks" ] && [ "$(ls -A "$RAW_DIR/school-of-reward-hacks" 2>/dev/null)" ]; then
    echo "school-of-reward-hacks already exists at $RAW_DIR/school-of-reward-hacks, skipping."
else
    echo "Downloading school-of-reward-hacks from HuggingFace..."
    python3 -c "
from datasets import load_dataset
import os

raw_dir = os.environ['RAW_DIR']
save_dir = os.path.join(raw_dir, 'school-of-reward-hacks')
os.makedirs(save_dir, exist_ok=True)

ds = load_dataset('longtermrisk/school-of-reward-hacks')
for split_name, split_ds in ds.items():
    out_path = os.path.join(save_dir, f'{split_name}.parquet')
    split_ds.to_parquet(out_path)
    print(f'Saved {split_name} ({len(split_ds)} rows) to {out_path}')

print('Done.')
"
fi
