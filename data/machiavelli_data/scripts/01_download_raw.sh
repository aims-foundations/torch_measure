#!/bin/bash
set -euo pipefail

# MACHIAVELLI: Benchmark for evaluating Machiavellian behaviors in LLM agents.
# Uses text-based choose-your-own-adventure games to measure power-seeking,
# deception, and other harmful behaviors.
# Source: https://github.com/aypan17/machiavelli
# Note: Game data may require separate download from Google Drive (see repo README).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/machiavelli" ]; then
    echo "Cloning machiavelli repository..."
    git clone --depth 1 https://github.com/aypan17/machiavelli.git "$RAW_DIR/machiavelli"
    echo "Done cloning repository."
    echo ""
    echo "NOTE: The full game data may need to be downloaded separately."
    echo "Check the repository README for Google Drive download instructions."
else
    echo "machiavelli already exists at $RAW_DIR/machiavelli, skipping."
fi
