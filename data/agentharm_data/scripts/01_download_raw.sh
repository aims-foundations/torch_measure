#!/bin/bash
# Download UK AISI AgentHarm dataset (harmful agent tasks)
# Source: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
# Benchmark from the UK AI Safety Institute for evaluating agent refusal of harmful tasks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/agentharm" ]; then
    echo "Downloading ai-safety-institute/AgentHarm..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('ai-safety-institute/AgentHarm', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/agentharm')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
