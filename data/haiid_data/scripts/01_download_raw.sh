#!/bin/bash
# Download HAIID (Human-AI Interactions Dataset)
# 30,000+ interactions across 5 classification domains
# Domains: art, census, cities, sarcasm, dermatology
# Repo: https://github.com/kailas-v/human-ai-interactions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/repo" ]; then
    echo "Cloning HAIID repo..."
    git clone --depth 1 https://github.com/kailas-v/human-ai-interactions.git "$RAW_DIR/repo"
else
    echo "Repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/repo/"
