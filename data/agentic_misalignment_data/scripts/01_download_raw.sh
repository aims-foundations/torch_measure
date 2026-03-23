#!/bin/bash
set -euo pipefail

# Agentic Misalignment: Anthropic's study on misalignment behaviors in agentic LLM settings.
# GitHub: https://github.com/anthropic-experimental/agentic-misalignment
# HuggingFace results: https://huggingface.co/datasets/cfahlgren1/anthropic-agentic-misalignment-results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

# Clone the GitHub repository
if [ ! -d "$RAW_DIR/agentic-misalignment" ]; then
    echo "Cloning agentic-misalignment repository..."
    git clone --depth 1 https://github.com/anthropic-experimental/agentic-misalignment.git "$RAW_DIR/agentic-misalignment"
    echo "Done cloning repository."
else
    echo "agentic-misalignment repository already exists at $RAW_DIR/agentic-misalignment, skipping clone."
fi

# Also download the HuggingFace results dataset
if [ -d "$RAW_DIR/agentic-misalignment-results" ] && [ "$(ls -A "$RAW_DIR/agentic-misalignment-results" 2>/dev/null)" ]; then
    echo "agentic-misalignment-results already exists at $RAW_DIR/agentic-misalignment-results, skipping."
else
    echo "Downloading agentic-misalignment-results from HuggingFace..."
    python3 -c "
from datasets import load_dataset
import os

raw_dir = os.environ['RAW_DIR']
save_dir = os.path.join(raw_dir, 'agentic-misalignment-results')
os.makedirs(save_dir, exist_ok=True)

ds = load_dataset('cfahlgren1/anthropic-agentic-misalignment-results')
for split_name, split_ds in ds.items():
    out_path = os.path.join(save_dir, f'{split_name}.parquet')
    split_ds.to_parquet(out_path)
    print(f'Saved {split_name} ({len(split_ds)} rows) to {out_path}')

print('Done.')
"
fi
