#!/bin/bash
# Download JailbreakBench artifacts — per-model per-behavior jailbreak results
# 4 models x 100 behaviors x 5 attack methods, with binary success labels
# Repo: https://github.com/JailbreakBench/artifacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/artifacts" ]; then
    echo "Cloning JailbreakBench artifacts repo..."
    git clone --depth 1 https://github.com/JailbreakBench/artifacts.git "$RAW_DIR/artifacts"
else
    echo "Artifacts repo already cloned, skipping"
fi

# Also clone the main repo for behavior metadata
if [ ! -d "$RAW_DIR/jailbreakbench" ]; then
    echo "Cloning JailbreakBench main repo..."
    git clone --depth 1 https://github.com/JailbreakBench/jailbreakbench.git "$RAW_DIR/jailbreakbench"
else
    echo "Main repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/"
