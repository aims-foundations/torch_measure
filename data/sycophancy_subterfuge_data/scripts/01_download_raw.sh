#!/bin/bash
set -euo pipefail

# Sycophancy to Subterfuge: Anthropic's study on how sycophantic behavior in LLMs
# can escalate into more harmful subterfuge behaviors.
# Source: https://github.com/anthropics/sycophancy-to-subterfuge-paper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/sycophancy-to-subterfuge-paper" ]; then
    echo "Cloning sycophancy-to-subterfuge-paper repository..."
    git clone --depth 1 https://github.com/anthropics/sycophancy-to-subterfuge-paper.git "$RAW_DIR/sycophancy-to-subterfuge-paper"
    echo "Done."
else
    echo "sycophancy-to-subterfuge-paper already exists at $RAW_DIR/sycophancy-to-subterfuge-paper, skipping."
fi
