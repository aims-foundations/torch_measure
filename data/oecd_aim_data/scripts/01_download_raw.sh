#!/bin/bash
# 01_download_raw.sh — OECD AI Incidents Monitor (AIM) placeholder.
#
# Source: https://oecd.ai/en/incidents
#
# NOTE: The OECD AIM does not provide a direct bulk download API or data export.
# The incident data is accessible through a web interface only.
# This script downloads the methodology documentation as a reference and creates
# a placeholder noting that manual download is required.
#
# For bulk access, contact OECD directly or check for periodic data dumps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

README_FILE="$RAW_DIR/MANUAL_DOWNLOAD_NEEDED.txt"
METHODOLOGY_FILE="$RAW_DIR/oecd_aim_methodology.html"

echo "[INFO] OECD AI Incidents Monitor (AIM) — data download setup"

# --- Create placeholder readme ---
if [[ -f "$README_FILE" ]]; then
    echo "[SKIP] Placeholder readme already exists."
else
    cat > "$README_FILE" << 'PLACEHOLDER'
OECD AI Incidents Monitor (AIM) — Manual Download Required
============================================================

The OECD AIM database does not provide a public bulk download API.

Data source: https://oecd.ai/en/incidents

Options for obtaining data:
1. Manual browsing and export from the OECD AIM web interface.
2. Contact OECD for research access to bulk data.
3. Use web scraping (check OECD terms of service first).
4. Check if OECD publishes periodic CSV/JSON dumps.

Once you have the data, place the files in this raw/ directory.

Reference:
- OECD AI Policy Observatory: https://oecd.ai/en/
- AIM Methodology: https://oecd.ai/en/dashboards/ai-incidents
PLACEHOLDER
    echo "[INFO] Created placeholder: $README_FILE"
fi

# --- Download methodology page as reference ---
if [[ -f "$METHODOLOGY_FILE" ]]; then
    echo "[SKIP] Methodology page already downloaded."
else
    echo "[INFO] Downloading OECD AIM methodology page as reference..."
    curl -fsSL --retry 3 --retry-delay 5 \
        -H "User-Agent: Mozilla/5.0 (compatible; research-download)" \
        -o "$METHODOLOGY_FILE" \
        "https://oecd.ai/en/incidents" || {
        echo "[WARN] Could not download methodology page."
        echo "[INFO] Visit https://oecd.ai/en/incidents manually."
    }
    if [[ -f "$METHODOLOGY_FILE" ]]; then
        echo "[INFO] Saved methodology reference page."
    fi
fi

echo ""
echo "[INFO] OECD AIM setup complete."
echo "[WARN] Bulk data must be obtained manually from https://oecd.ai/en/incidents"
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
