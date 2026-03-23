#!/bin/bash
# Download ScaleAI MRT dataset (agent monitoring trajectories)
# Source: https://huggingface.co/datasets/ScaleAI/mrt
# Monitoring and Red-Teaming trajectories for evaluating AI agent safety.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/mrt" ]; then
    echo "Downloading ScaleAI/mrt..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('ScaleAI/mrt', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/mrt')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
