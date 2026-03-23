#!/bin/bash
# Download GenAICanHarmLearning dataset
# High school students x math problems x {control, GPT Base, GPT Tutor} -> scores
# Paper: Bastani et al., PNAS 2025
# Repo: https://github.com/obastani/GenAICanHarmLearning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/repo" ]; then
    echo "Cloning GenAICanHarmLearning repo..."
    git clone --depth 1 https://github.com/obastani/GenAICanHarmLearning.git "$RAW_DIR/repo"
else
    echo "Repo already cloned, skipping"
fi

echo "Done. Raw files in $RAW_DIR/repo/"
