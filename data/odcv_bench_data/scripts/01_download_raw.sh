#!/bin/bash
set -euo pipefail

# ODCV-Bench: Out-of-Distribution Computer Vision benchmark.
# Evaluates model robustness to distribution shifts in vision tasks.
# Source: https://github.com/McGill-DMaS/ODCV-Bench

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/ODCV-Bench" ]; then
    echo "Cloning ODCV-Bench repository..."
    git clone --depth 1 https://github.com/McGill-DMaS/ODCV-Bench.git "$RAW_DIR/ODCV-Bench"
    echo "Done."
else
    echo "ODCV-Bench already exists at $RAW_DIR/ODCV-Bench, skipping."
fi
