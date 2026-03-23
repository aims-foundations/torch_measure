#!/bin/bash
set -euo pipefail

# CoT Unfaithfulness: Study on unfaithful chain-of-thought reasoning in LLMs.
# Investigates whether LLM reasoning steps actually reflect the model's decision process.
# Source: https://github.com/milesaturpin/cot-unfaithfulness

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/cot-unfaithfulness" ]; then
    echo "Cloning cot-unfaithfulness repository..."
    git clone --depth 1 https://github.com/milesaturpin/cot-unfaithfulness.git "$RAW_DIR/cot-unfaithfulness"
    echo "Done."
else
    echo "cot-unfaithfulness already exists at $RAW_DIR/cot-unfaithfulness, skipping."
fi
