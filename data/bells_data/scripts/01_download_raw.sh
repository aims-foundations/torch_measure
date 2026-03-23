#!/bin/bash
set -euo pipefail

# BELLS: Benchmark for the Evaluation of LLM Supervision.
# Collects traces of LLM misbehavior (jailbreaks, prompt injections, etc.).
# Source: https://github.com/CentreSecuriteIA/BELLS
# Dataset files: https://bells.therandom.space/datasets/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/BELLS" ]; then
    echo "Cloning BELLS repository..."
    git clone --depth 1 https://github.com/CentreSecuriteIA/BELLS.git "$RAW_DIR/BELLS"
    echo "Done cloning repository."
else
    echo "BELLS repository already exists at $RAW_DIR/BELLS, skipping clone."
fi

# Also download JSONL dataset files from the BELLS website
DATASETS_DIR="$RAW_DIR/bells_datasets"
mkdir -p "$DATASETS_DIR"

BELLS_BASE_URL="https://bells.therandom.space/datasets"
DATASET_FILES=(
    "bipia.jsonl"
    "dan.jsonl"
    "gandalf.jsonl"
    "hallucinations.jsonl"
    "machiavelli.jsonl"
    "promptinject.jsonl"
    "tensortrust.jsonl"
    "toolcalls.jsonl"
)

for f in "${DATASET_FILES[@]}"; do
    if [ ! -f "$DATASETS_DIR/$f" ]; then
        echo "Downloading $f..."
        curl -fsSL "${BELLS_BASE_URL}/${f}" -o "$DATASETS_DIR/$f" || echo "Warning: failed to download $f (may not exist at this URL)"
    else
        echo "$f already exists, skipping."
    fi
done

echo "BELLS download complete."
