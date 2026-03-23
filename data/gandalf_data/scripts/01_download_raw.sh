#!/bin/bash
# Download Lakera Gandalf dataset (~279K prompt injections)
# Source: https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
# Prompt injection attempts from the Gandalf "ignore instructions" challenge.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/gandalf_ignore_instructions" ]; then
    echo "Downloading Lakera/gandalf_ignore_instructions..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('Lakera/gandalf_ignore_instructions', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/gandalf_ignore_instructions')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
