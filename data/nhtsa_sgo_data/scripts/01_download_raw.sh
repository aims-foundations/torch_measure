#!/bin/bash
# 01_download_raw.sh — Download NHTSA Standing General Order crash report CSVs.
#
# Source: https://www.nhtsa.gov/technology-innovation/automated-vehicles-safety
# SGO-2021-01 requires manufacturers to report crashes involving ADS and ADAS.
#
# Downloads:
#   - SGO-2021-01_Incident_Reports_ADS.csv  (Automated Driving Systems)
#   - SGO-2021-01_Incident_Reports_ADAS.csv (Advanced Driver Assistance Systems)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

ADS_URL="https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
ADAS_URL="https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv"

ADS_FILE="$RAW_DIR/SGO-2021-01_Incident_Reports_ADS.csv"
ADAS_FILE="$RAW_DIR/SGO-2021-01_Incident_Reports_ADAS.csv"

echo "[INFO] Downloading NHTSA SGO crash reports..."

if [[ -f "$ADS_FILE" ]]; then
    echo "[SKIP] ADS file already exists: $ADS_FILE"
else
    echo "[INFO] Downloading ADS incident reports..."
    curl -fSL --retry 3 --retry-delay 5 -o "$ADS_FILE" "$ADS_URL"
    echo "[INFO] Downloaded ADS file: $(wc -l < "$ADS_FILE") lines"
fi

if [[ -f "$ADAS_FILE" ]]; then
    echo "[SKIP] ADAS file already exists: $ADAS_FILE"
else
    echo "[INFO] Downloading ADAS incident reports..."
    curl -fSL --retry 3 --retry-delay 5 -o "$ADAS_FILE" "$ADAS_URL"
    echo "[INFO] Downloaded ADAS file: $(wc -l < "$ADAS_FILE") lines"
fi

echo "[INFO] NHTSA SGO download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
