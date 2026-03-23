#!/bin/bash
set -euo pipefail

# Apollo Deception Detection: Benchmark for detecting deceptive behavior in LLMs.
# From Apollo Research's work on AI deception and alignment.
# Source: https://github.com/ApolloResearch/deception-detection

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/deception-detection" ]; then
    echo "Cloning deception-detection repository..."
    git clone --depth 1 https://github.com/ApolloResearch/deception-detection.git "$RAW_DIR/deception-detection"
    echo "Done."
else
    echo "deception-detection already exists at $RAW_DIR/deception-detection, skipping."
fi
