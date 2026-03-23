#!/bin/bash
set -euo pipefail

# SafeAgentBench: Benchmark for evaluating safety of LLM-based agents.
# Source: https://huggingface.co/datasets/safeagentbench/SafeAgentBench

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ -d "$RAW_DIR/SafeAgentBench" ] && [ "$(ls -A "$RAW_DIR/SafeAgentBench" 2>/dev/null)" ]; then
    echo "SafeAgentBench already exists at $RAW_DIR/SafeAgentBench, skipping."
else
    echo "Downloading SafeAgentBench from HuggingFace..."
    python3 -c "
from datasets import load_dataset
import os

raw_dir = os.environ['RAW_DIR']
save_dir = os.path.join(raw_dir, 'SafeAgentBench')
os.makedirs(save_dir, exist_ok=True)

ds = load_dataset('safeagentbench/SafeAgentBench')
for split_name, split_ds in ds.items():
    out_path = os.path.join(save_dir, f'{split_name}.parquet')
    split_ds.to_parquet(out_path)
    print(f'Saved {split_name} ({len(split_ds)} rows) to {out_path}')

print('Done.')
"
fi
