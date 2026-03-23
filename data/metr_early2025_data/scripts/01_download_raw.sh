#!/bin/bash
# Download METR Early-2025 AI Developer Productivity RCT data
# 16 developers x 246 issues x {AI allowed, AI disallowed} -> completion time
# Paper: https://arxiv.org/abs/2507.09089
# Repo: https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/repo" ]; then
    echo "Cloning METR Early-2025 study repo..."
    git clone --depth 1 https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs.git "$RAW_DIR/repo"
else
    echo "Repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/repo/"
