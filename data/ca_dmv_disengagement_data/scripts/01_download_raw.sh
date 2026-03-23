#!/bin/bash
# 01_download_raw.sh — Download California DMV autonomous vehicle disengagement reports.
#
# Sources:
#   1. CA DMV official 2024 reports (general + driverless CSVs)
#   2. Mendeley consolidated academic dataset (historical disengagement data)
#
# The CA DMV requires all AV permit holders to report disengagement events annually.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

# --- CA DMV 2024 Official Reports ---
DMV_GENERAL_URL="https://www.dmv.ca.gov/portal/file/2024-autonomous-vehicle-disengagement-reports-csv/"
DMV_DRIVERLESS_URL="https://www.dmv.ca.gov/portal/file/2024-autonomous-vehicle-disengagement-reports-csvdriverless/"

DMV_GENERAL_FILE="$RAW_DIR/2024_disengagement_general.csv"
DMV_DRIVERLESS_FILE="$RAW_DIR/2024_disengagement_driverless.csv"

echo "[INFO] Downloading CA DMV disengagement reports..."

if [[ -f "$DMV_GENERAL_FILE" ]]; then
    echo "[SKIP] 2024 general report already exists: $DMV_GENERAL_FILE"
else
    echo "[INFO] Downloading 2024 general disengagement report..."
    curl -fSL --retry 3 --retry-delay 5 \
        -H "User-Agent: Mozilla/5.0 (compatible; research-download)" \
        -o "$DMV_GENERAL_FILE" "$DMV_GENERAL_URL" || {
        echo "[WARN] Failed to download 2024 general report (CA DMV may require browser access)."
        echo "[WARN] Try downloading manually from: $DMV_GENERAL_URL"
    }
fi

if [[ -f "$DMV_DRIVERLESS_FILE" ]]; then
    echo "[SKIP] 2024 driverless report already exists: $DMV_DRIVERLESS_FILE"
else
    echo "[INFO] Downloading 2024 driverless disengagement report..."
    curl -fSL --retry 3 --retry-delay 5 \
        -H "User-Agent: Mozilla/5.0 (compatible; research-download)" \
        -o "$DMV_DRIVERLESS_FILE" "$DMV_DRIVERLESS_URL" || {
        echo "[WARN] Failed to download 2024 driverless report (CA DMV may require browser access)."
        echo "[WARN] Try downloading manually from: $DMV_DRIVERLESS_URL"
    }
fi

# --- Mendeley Consolidated Dataset ---
# Dataset: "California DMV Autonomous Vehicle Disengagement Reports" by multiple authors
# https://data.mendeley.com/datasets/74s6nw7dk9/1
MENDELEY_URL="https://data.mendeley.com/datasets/74s6nw7dk9/1"
MENDELEY_DIR="$RAW_DIR/mendeley"

if [[ -d "$MENDELEY_DIR" && -n "$(ls -A "$MENDELEY_DIR" 2>/dev/null)" ]]; then
    echo "[SKIP] Mendeley dataset directory already exists: $MENDELEY_DIR"
else
    mkdir -p "$MENDELEY_DIR"
    echo "[INFO] Attempting to download Mendeley consolidated dataset..."
    # Mendeley Data API: try to get the dataset files listing
    # The direct download URL pattern for Mendeley datasets
    MENDELEY_API="https://data.mendeley.com/api/datasets/74s6nw7dk9/1/files"
    echo "[INFO] Querying Mendeley API for file list..."
    curl -fsSL --retry 3 "$MENDELEY_API" -o "$MENDELEY_DIR/files_listing.json" || {
        echo "[WARN] Could not query Mendeley API directly."
        echo "[WARN] Please download manually from: $MENDELEY_URL"
        echo "[WARN] Save files to: $MENDELEY_DIR/"
    }

    # If we got the file listing, try to download each file
    if [[ -f "$MENDELEY_DIR/files_listing.json" ]]; then
        echo "[INFO] Parsing Mendeley file listing and downloading files..."
        python3 -c "
import json, subprocess, os

mendeley_dir = '$MENDELEY_DIR'
with open(os.path.join(mendeley_dir, 'files_listing.json')) as f:
    files = json.load(f)

if not isinstance(files, list):
    print('[WARN] Unexpected Mendeley API response format. Manual download may be needed.')
else:
    for finfo in files:
        fname = finfo.get('filename', finfo.get('name', 'unknown'))
        fid = finfo.get('id', '')
        dl_url = f'https://data.mendeley.com/api/datasets/74s6nw7dk9/1/files/{fid}/download'
        outpath = os.path.join(mendeley_dir, fname)
        if os.path.exists(outpath):
            print(f'[SKIP] {fname} already exists')
            continue
        print(f'[INFO] Downloading {fname}...')
        subprocess.run(['curl', '-fSL', '--retry', '3', '-o', outpath, dl_url], check=False)
" || echo "[WARN] Mendeley download via API failed. Manual download may be needed."
    fi
fi

echo ""
echo "[INFO] CA DMV disengagement download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lhR "$RAW_DIR"
