#!/bin/bash
# Download AI45Research ATBench dataset (agent trajectory safety)
# Source: https://huggingface.co/datasets/AI45Research/ATBench
# Benchmark for evaluating safety of agent trajectories across diverse tasks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/atbench" ]; then
    echo "Downloading AI45Research/ATBench..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('AI45Research/ATBench', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/atbench')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
