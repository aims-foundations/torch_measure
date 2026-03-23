#!/bin/bash
# Download Anthropic alignment-faking-rl dataset (~2.14M RL transcripts)
# Source: https://huggingface.co/datasets/Anthropic/alignment-faking-rl
# Transcripts studying alignment-faking behavior during RL training.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/alignment_faking_rl" ]; then
    echo "Downloading Anthropic/alignment-faking-rl..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('Anthropic/alignment-faking-rl', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/alignment_faking_rl')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
