#!/bin/bash
# Download Tensor Trust dataset (~563K prompt injection attacks)
# Source: https://huggingface.co/datasets/qxcv/tensor-trust
# Large-scale dataset of prompt injection attacks from the Tensor Trust game.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/tensor_trust" ]; then
    echo "Downloading qxcv/tensor-trust..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('qxcv/tensor-trust', trust_remote_code=True)
ds.save_to_disk('$RAW_DIR/tensor_trust')
print('Done:', ds)
"
else
    echo "Already downloaded, skipping"
fi
