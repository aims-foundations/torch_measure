#!/bin/bash
# Download METR Late-2025 AI Developer Productivity RCT data
# 57 developers x ~1,134 issues x {AI allowed, AI disallowed} -> completion time
# Repo: https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/repo" ]; then
    echo "Cloning METR Late-2025 study repo..."
    git clone --depth 1 https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs.git "$RAW_DIR/repo"
else
    echo "Repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/repo/"
