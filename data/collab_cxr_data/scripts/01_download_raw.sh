#!/bin/bash
# Download Collab-CXR dataset from OSF (https://osf.io/z7apq/)
# 227 radiologists x 324 chest X-ray cases x 4 conditions -> probabilistic diagnosis
# Paper: Scientific Data 2025, https://www.nature.com/articles/s41597-025-05054-0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

echo "Downloading Collab-CXR data from OSF..."

# Main data file (OSF GUID: myedf, ~36 MB)
if [ ! -f "$RAW_DIR/data_public.txt.gz" ] || [ "$(stat -c%s "$RAW_DIR/data_public.txt.gz" 2>/dev/null || echo 0)" -lt 1000 ]; then
    rm -f "$RAW_DIR/data_public.txt.gz"
    curl -L -o "$RAW_DIR/data_public.txt.gz" \
        "https://osf.io/download/myedf/"
    echo "Downloaded data_public.txt.gz ($(du -h "$RAW_DIR/data_public.txt.gz" | cut -f1))"
else
    echo "data_public.txt.gz already exists, skipping"
fi

# Clickstream data (OSF GUID: yf5h4, ~65 MB)
if [ ! -f "$RAW_DIR/clickstream_public.txt.gz" ] || [ "$(stat -c%s "$RAW_DIR/clickstream_public.txt.gz" 2>/dev/null || echo 0)" -lt 1000 ]; then
    rm -f "$RAW_DIR/clickstream_public.txt.gz"
    curl -L -o "$RAW_DIR/clickstream_public.txt.gz" \
        "https://osf.io/download/yf5h4/"
    echo "Downloaded clickstream_public.txt.gz ($(du -h "$RAW_DIR/clickstream_public.txt.gz" | cut -f1))"
else
    echo "clickstream_public.txt.gz already exists, skipping"
fi

echo "Done. Raw files in $RAW_DIR"
