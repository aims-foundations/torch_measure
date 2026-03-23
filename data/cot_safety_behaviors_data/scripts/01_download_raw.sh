#!/bin/bash
# Download reasoning safety behaviours dataset (CoT safety labels)
# Source: https://huggingface.co/datasets/AISafety-Student/reasoning-safety-behaviours
# Labels for safety-relevant behaviors observed in chain-of-thought reasoning.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/reasoning_safety_behaviours" ]; then
    echo "Downloading AISafety-Student/reasoning-safety-behaviours..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('AISafety-Student/reasoning-safety-behaviours', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/reasoning_safety_behaviours')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
