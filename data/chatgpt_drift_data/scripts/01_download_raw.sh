#!/bin/bash
# 01_download_raw.sh — Download data from "How is ChatGPT's behavior changing over time?"
#
# Source: https://github.com/lchen001/LLMDrift
# Paper: "How is ChatGPT's behavior changing over time?" (Chen et al., 2023)
#
# This paper tracks GPT-3.5 and GPT-4 performance changes across tasks over time,
# including math, code generation, sensitivity, and more.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

REPO_DIR="$RAW_DIR/LLMDrift"
REPO_URL="https://github.com/lchen001/LLMDrift.git"

echo "[INFO] Downloading ChatGPT drift data from GitHub..."

if [[ -d "$REPO_DIR" && -d "$REPO_DIR/.git" ]]; then
    echo "[SKIP] Repository already cloned at: $REPO_DIR"
    echo "[INFO] Pulling latest changes..."
    cd "$REPO_DIR" && git pull --ff-only 2>/dev/null || {
        echo "[WARN] Could not pull updates (may have diverged). Using existing clone."
    }
else
    echo "[INFO] Cloning $REPO_URL..."
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    echo "[INFO] Clone complete."
fi

echo ""
echo "[INFO] ChatGPT drift data download complete."
echo "[INFO] Repository contents:"
ls -lh "$REPO_DIR"
echo ""
echo "[INFO] Data directories:"
find "$REPO_DIR" -type d -maxdepth 2 | head -20
