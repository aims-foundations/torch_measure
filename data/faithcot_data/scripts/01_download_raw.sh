#!/bin/bash
set -euo pipefail

# FaithCoT-BENCH: Chain-of-thought faithfulness benchmark
# Evaluates whether LLM reasoning chains are faithful to their final answers.
# Source: https://github.com/se7esx/FaithCoT-BENCH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/FaithCoT-BENCH" ]; then
    echo "Cloning FaithCoT-BENCH repository..."
    git clone --depth 1 https://github.com/se7esx/FaithCoT-BENCH.git "$RAW_DIR/FaithCoT-BENCH"
    echo "Done."
else
    echo "FaithCoT-BENCH already exists at $RAW_DIR/FaithCoT-BENCH, skipping."
fi
