#!/bin/bash
# Download AI Safety Index reports from Future of Life Institute
# Winter 2025: 8 companies x 35 indicators x 6 domains
# Summer 2025: 7 companies x 33 indicators

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../raw"

mkdir -p "$RAW_DIR"

for url_name in \
    "https://futureoflife.org/wp-content/uploads/2025/12/AI-Safety-Index-Report_131225_Full_Report_Digital.pdf|winter2025_full.pdf" \
    "https://futureoflife.org/wp-content/uploads/2025/07/FLI-AI-Safety-Index-Report-Summer-2025.pdf|summer2025_full.pdf"; do
    url="${url_name%%|*}"
    name="${url_name##*|}"
    if [ ! -f "$RAW_DIR/$name" ]; then
        echo "Downloading $name..."
        curl -L -o "$RAW_DIR/$name" "$url"
    else
        echo "$name already exists, skipping"
    fi
done

echo "Done. Raw files in $RAW_DIR/"
