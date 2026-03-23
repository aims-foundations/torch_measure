#!/bin/bash
# Download PatronusAI TRAIL dataset (OpenTelemetry agent traces)
# Source: https://huggingface.co/datasets/PatronusAI/TRAIL
# Structured agent traces in OpenTelemetry format for evaluating agentic systems.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/trail" ]; then
    echo "Downloading PatronusAI/TRAIL..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('PatronusAI/TRAIL', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/trail')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
