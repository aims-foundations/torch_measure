#!/bin/bash
# 01_download_raw.sh — Download MIT AI Risk Repository data.
#
# Source: https://airisk.mit.edu/
# Paper: "The AI Risk Repository: A Comprehensive Meta-Review, Database, and
#         Taxonomy of Risks From Artificial Intelligence" (Slattery et al., 2024)
#
# The data is maintained in Google Sheets. This script exports the sheets as CSV.
# Known spreadsheet IDs (from the airisk.mit.edu website):
#   - Main risk database
#   - Causal taxonomy
#   - Domain taxonomy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

echo "[INFO] Downloading MIT AI Risk Repository..."

# The MIT AI Risk Repository provides Google Sheets with their data.
# Spreadsheet ID from the website's public links.
# Main database spreadsheet
SHEET_ID="1UCFKM60su91kYJlLSoJV2sgKwM_EEOxGlkJslT4c9yo"

# Export individual sheets as CSV using the Google Sheets export URL pattern
# gid=0 is typically the first sheet
EXPORT_BASE="https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv"

# Known sheet GIDs (these may need updating if the spreadsheet structure changes)
declare -A SHEETS
SHEETS["ai_risk_database"]="0"
SHEETS["causal_taxonomy"]="1781887782"
SHEETS["domain_taxonomy"]="1006192738"

for sheet_name in "${!SHEETS[@]}"; do
    gid="${SHEETS[$sheet_name]}"
    outfile="$RAW_DIR/${sheet_name}.csv"

    if [[ -f "$outfile" ]]; then
        echo "[SKIP] $sheet_name already exists: $outfile"
        continue
    fi

    echo "[INFO] Downloading sheet: $sheet_name (gid=$gid)..."
    curl -fSL --retry 3 --retry-delay 5 \
        -o "$outfile" \
        "${EXPORT_BASE}&gid=${gid}" || {
        echo "[WARN] Failed to download $sheet_name with gid=$gid"
        echo "[WARN] The sheet GID may have changed. Check https://airisk.mit.edu/"
        rm -f "$outfile"
    }

    if [[ -f "$outfile" ]]; then
        # Check if we got an HTML error page instead of CSV
        if head -1 "$outfile" | grep -qi "<!DOCTYPE\|<html"; then
            echo "[WARN] Got HTML instead of CSV for $sheet_name. Google may require auth."
            rm -f "$outfile"
        else
            echo "[INFO] Downloaded $sheet_name: $(wc -l < "$outfile") lines"
        fi
    fi
done

# Also try the full spreadsheet as XLSX
XLSX_FILE="$RAW_DIR/mit_ai_risk_repository.xlsx"
if [[ -f "$XLSX_FILE" ]]; then
    echo "[SKIP] XLSX export already exists: $XLSX_FILE"
else
    echo "[INFO] Downloading full spreadsheet as XLSX..."
    XLSX_URL="https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=xlsx"
    curl -fSL --retry 3 --retry-delay 5 \
        -o "$XLSX_FILE" \
        "$XLSX_URL" || {
        echo "[WARN] Failed to download XLSX export."
        rm -f "$XLSX_FILE"
    }

    if [[ -f "$XLSX_FILE" ]]; then
        if head -c 4 "$XLSX_FILE" | grep -q "PK"; then
            echo "[INFO] Downloaded XLSX file successfully."
        else
            echo "[WARN] Downloaded file does not appear to be a valid XLSX. Removing."
            rm -f "$XLSX_FILE"
        fi
    fi
fi

# Fallback: try the GitHub repo if it exists
GITHUB_DIR="$RAW_DIR/airisk_repo"
if [[ ! -d "$GITHUB_DIR" ]]; then
    echo "[INFO] Checking for MIT AI Risk Repository on GitHub..."
    # The repo may be at https://github.com/csalt-research/ai-risk-repository or similar
    git clone --depth 1 "https://github.com/csalt-research/ai-risk-repository.git" \
        "$GITHUB_DIR" 2>/dev/null || {
        echo "[INFO] No GitHub repo found at expected URL (this is expected)."
    }
fi

echo ""
echo "[INFO] MIT AI Risk Repository download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
