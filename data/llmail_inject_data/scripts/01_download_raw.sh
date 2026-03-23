#!/bin/bash
# Download Microsoft LLMail-Inject dataset (~127K prompt injection attacks)
# Source: https://huggingface.co/datasets/microsoft/llmail-inject-challenge
# Prompt injection challenge dataset from Microsoft for email agent scenarios.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/llmail_inject" ]; then
    echo "Downloading microsoft/llmail-inject-challenge..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('microsoft/llmail-inject-challenge', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/llmail_inject')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
