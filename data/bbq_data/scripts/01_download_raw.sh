#!/bin/bash
# Download BBQ (Bias Benchmark for QA) — full dataset + per-model results
# 58,492 items across 11 bias categories, per-model logits for 5+ models
# Repo: https://github.com/nyu-mll/BBQ

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/repo" ]; then
    echo "Cloning BBQ repo..."
    git clone --depth 1 https://github.com/nyu-mll/BBQ.git "$RAW_DIR/repo"
else
    echo "Repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/repo/"
