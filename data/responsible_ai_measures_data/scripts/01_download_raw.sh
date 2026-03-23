#!/bin/bash
# Download Responsible AI Measures Dataset from Figshare
# 791 evaluation measures x 11 ethical principles, from 257 papers
# Paper: Rismani et al., Nature Scientific Data 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

if [ ! -f "$RAW_DIR/rai_measures_dataset.zip" ]; then
    echo "Downloading RAI Measures Dataset from Figshare..."
    curl -L -o "$RAW_DIR/rai_measures_dataset.zip" \
        "https://ndownloader.figshare.com/files/57701437"
    echo "Downloaded. Unzipping..."
    cd "$RAW_DIR" && unzip -o rai_measures_dataset.zip
else
    echo "Dataset already downloaded, skipping"
fi

echo "Done. Raw files in $RAW_DIR/"
