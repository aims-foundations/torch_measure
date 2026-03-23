#!/bin/bash
# 01_download_raw.sh — Download AI Incident Database (AIID) via GraphQL API.
#
# Source: https://incidentdatabase.ai/
# Paper: "The AI Incident Database" (McGregor, 2021)
#
# Queries the AIID GraphQL endpoint for all incidents with their reports
# and taxonomy classifications. Saves as JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$RAW_DIR"

INCIDENTS_FILE="$RAW_DIR/aiid_incidents.json"
REPORTS_FILE="$RAW_DIR/aiid_reports.json"
CLASSIFICATIONS_FILE="$RAW_DIR/aiid_classifications.json"

echo "[INFO] Downloading AIID data via GraphQL API..."

GRAPHQL_URL="https://incidentdatabase.ai/api/graphql"

# --- Download Incidents ---
if [[ -f "$INCIDENTS_FILE" ]]; then
    echo "[SKIP] Incidents file already exists: $INCIDENTS_FILE"
else
    echo "[INFO] Querying all incidents..."
    python3 -c "
import json
import urllib.request

url = '$GRAPHQL_URL'
output = '$INCIDENTS_FILE'

# Query incidents with basic info
query = '''
{
  incidents(limit: 10000) {
    incident_id
    title
    date
    description
    editors {
      userId
    }
    reports {
      report_number
    }
  }
}
'''

data = json.dumps({'query': query}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={
    'Content-Type': 'application/json',
    'User-Agent': 'research-download/1.0'
})

print('[INFO] Sending GraphQL query for incidents...')
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    if 'errors' in result:
        print(f'[WARN] GraphQL errors: {result[\"errors\"]}')

    incidents = result.get('data', {}).get('incidents', [])
    print(f'[INFO] Retrieved {len(incidents)} incidents.')

    with open(output, 'w') as f:
        json.dump(incidents, f, indent=2)
    print(f'[INFO] Saved to {output}')
except Exception as e:
    print(f'[ERROR] Failed to query incidents: {e}')
    print('[INFO] Trying alternative: download from AIID public snapshot...')
    # Fallback: try the MongoDB snapshot or CSV export if available
    import sys
    sys.exit(1)
"
fi

# --- Download Reports ---
if [[ -f "$REPORTS_FILE" ]]; then
    echo "[SKIP] Reports file already exists: $REPORTS_FILE"
else
    echo "[INFO] Querying all reports..."
    python3 -c "
import json
import urllib.request

url = '$GRAPHQL_URL'
output = '$REPORTS_FILE'

# Query reports with full details — paginate in batches
all_reports = []
skip = 0
batch_size = 1000

while True:
    query = '''
    {
      reports(limit: %d, skip: %d) {
        report_number
        title
        url
        source_domain
        date_published
        date_submitted
        date_modified
        language
        tags
      }
    }
    ''' % (batch_size, skip)

    data = json.dumps({'query': query}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json',
        'User-Agent': 'research-download/1.0'
    })

    print(f'[INFO] Fetching reports batch (skip={skip})...')
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'[WARN] Failed at skip={skip}: {e}')
        break

    reports = result.get('data', {}).get('reports', [])
    if not reports:
        break
    all_reports.extend(reports)
    print(f'[INFO] Got {len(reports)} reports (total: {len(all_reports)})')

    if len(reports) < batch_size:
        break
    skip += batch_size

print(f'[INFO] Total reports retrieved: {len(all_reports)}')
with open(output, 'w') as f:
    json.dump(all_reports, f, indent=2)
print(f'[INFO] Saved to {output}')
"
fi

# --- Download Classifications (taxonomies) ---
if [[ -f "$CLASSIFICATIONS_FILE" ]]; then
    echo "[SKIP] Classifications file already exists: $CLASSIFICATIONS_FILE"
else
    echo "[INFO] Querying classifications/taxonomies..."
    python3 -c "
import json
import urllib.request

url = '$GRAPHQL_URL'
output = '$CLASSIFICATIONS_FILE'

query = '''
{
  classifications(limit: 10000) {
    namespace
    incidents {
      incident_id
    }
    attributes {
      short_name
      value_json
    }
  }
}
'''

data = json.dumps({'query': query}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={
    'Content-Type': 'application/json',
    'User-Agent': 'research-download/1.0'
})

print('[INFO] Sending GraphQL query for classifications...')
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    if 'errors' in result:
        print(f'[WARN] GraphQL errors: {result[\"errors\"]}')

    classifications = result.get('data', {}).get('classifications', [])
    print(f'[INFO] Retrieved {len(classifications)} classification entries.')

    with open(output, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f'[INFO] Saved to {output}')
except Exception as e:
    print(f'[ERROR] Failed to query classifications: {e}')
"
fi

echo ""
echo "[INFO] AIID download complete."
echo "[INFO] Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
