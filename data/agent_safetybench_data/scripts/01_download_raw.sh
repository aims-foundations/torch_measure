#!/bin/bash
# Download THU-CoAI Agent-SafetyBench dataset (interactive agent safety)
# Source: https://huggingface.co/datasets/thu-coai/Agent-SafetyBench
# Benchmark for evaluating safety of LLM agents in interactive environments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/agent_safetybench" ]; then
    echo "Downloading thu-coai/Agent-SafetyBench..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('thu-coai/Agent-SafetyBench', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/agent_safetybench')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
