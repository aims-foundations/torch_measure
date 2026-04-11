"""
01_build_response_matrix.py — Download and process AI Incident Database (AIID).

Download: Queries the AIID GraphQL API (https://incidentdatabase.ai/) for all
incidents, reports, and taxonomy classifications. Saves as JSON.
Paper: "The AI Incident Database" (McGregor, 2021)

Processing: Loads MongoDB backup CSVs from raw/mongodump_full_snapshot/.
Source: https://incidentdatabase.ai/research/snapshots/
Structure: 1,407 incidents with CSET taxonomy classifications

Output:
  - incidents_clean.csv: cleaned incident catalog
  - incidents_by_year.csv: incident counts by year
  - deployer_x_harm.csv: deployer x harm-type cross-tabulation
  - classifications_summary.csv: CSET taxonomy summary
"""

INFO = {
    'description': 'Download and process AI Incident Database (AIID)',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2011.08512',
    'data_source_url': 'https://incidentdatabase.ai/research/snapshots/',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-SA-4.0',
    'citation': """@misc{mcgregor2020preventingrepeatedrealworld,
      title={Preventing Repeated Real World AI Failures by Cataloging Incidents: The AI Incident Database}, 
      author={Sean McGregor},
      year={2020},
      eprint={2011.08512},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2011.08512}, 
}""",
    'tags': ['pending'],
}


import sys
import ast
import json
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
GRAPHQL_URL = "https://incidentdatabase.ai/api/graphql"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"


def _graphql_query(query):
    """Send a GraphQL query and return the parsed JSON response."""
    data = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "research-download/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_incidents(output_file):
    """Download all incidents."""
    print("[INFO] Sending GraphQL query for incidents...")
    query = """
    {
      incidents(limit: 10000) {
        incident_id
        title
        date
        description
        editors { userId }
        reports { report_number }
      }
    }
    """
    result = _graphql_query(query)
    if "errors" in result:
        print(f"[WARN] GraphQL errors: {result['errors']}")
    incidents = result.get("data", {}).get("incidents", [])
    print(f"[INFO] Retrieved {len(incidents)} incidents.")
    with open(output_file, "w") as f:
        json.dump(incidents, f, indent=2)
    print(f"[INFO] Saved to {output_file}")


def _download_reports(output_file):
    """Download all reports with pagination."""
    all_reports = []
    skip = 0
    batch_size = 1000

    while True:
        query = """
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
        """ % (batch_size, skip)

        print(f"[INFO] Fetching reports batch (skip={skip})...")
        try:
            result = _graphql_query(query)
        except Exception as e:
            print(f"[WARN] Failed at skip={skip}: {e}")
            break

        reports = result.get("data", {}).get("reports", [])
        if not reports:
            break
        all_reports.extend(reports)
        print(f"[INFO] Got {len(reports)} reports (total: {len(all_reports)})")

        if len(reports) < batch_size:
            break
        skip += batch_size

    print(f"[INFO] Total reports retrieved: {len(all_reports)}")
    with open(output_file, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[INFO] Saved to {output_file}")


def _download_classifications(output_file):
    """Download taxonomy classifications."""
    print("[INFO] Sending GraphQL query for classifications...")
    query = """
    {
      classifications(limit: 10000) {
        namespace
        incidents { incident_id }
        attributes { short_name value_json }
      }
    }
    """
    try:
        result = _graphql_query(query)
        if "errors" in result:
            print(f"[WARN] GraphQL errors: {result['errors']}")
        classifications = result.get("data", {}).get("classifications", [])
        print(f"[INFO] Retrieved {len(classifications)} classification entries.")
        with open(output_file, "w") as f:
            json.dump(classifications, f, indent=2)
        print(f"[INFO] Saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to query classifications: {e}")


def download():
    """Download AIID data via GraphQL API."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading AIID data via GraphQL API...")

    incidents_file = RAW_DIR / "aiid_incidents.json"
    if incidents_file.exists():
        print(f"[SKIP] Incidents file already exists: {incidents_file}")
    else:
        try:
            _download_incidents(incidents_file)
        except Exception as e:
            print(f"[ERROR] Failed to query incidents: {e}")

    reports_file = RAW_DIR / "aiid_reports.json"
    if reports_file.exists():
        print(f"[SKIP] Reports file already exists: {reports_file}")
    else:
        _download_reports(reports_file)

    classifications_file = RAW_DIR / "aiid_classifications.json"
    if classifications_file.exists():
        print(f"[SKIP] Classifications file already exists: {classifications_file}")
    else:
        _download_classifications(classifications_file)

    print("\n[INFO] AIID download complete.")


def parse_list_field(val):
    """Parse stringified list fields like '["entity1","entity2"]'."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [str(val)]


def main():
    download()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mongodump_dir = RAW_DIR / "mongodump_full_snapshot"
    incidents_path = mongodump_dir / "incidents.csv"
    if not incidents_path.exists():
        print(f"No data found at {incidents_path}. Run 01_download_raw.sh first.")
        return

    print("Loading AIID incidents...")
    df = pd.read_csv(incidents_path)
    print(f"Incidents: {len(df)} rows, columns: {list(df.columns)}")

    # Clean dates
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    # Save clean incidents
    df.to_csv(OUTPUT_DIR / "incidents_clean.csv", index=False)

    # Incidents by year
    by_year = df["year"].value_counts().sort_index()
    by_year.to_csv(OUTPUT_DIR / "incidents_by_year.csv")
    print(f"\nIncidents by year:")
    for y, n in by_year.items():
        print(f"  {int(y)}: {n}")

    # Parse deployer lists
    deployer_col = next((c for c in df.columns if "deployer" in c.lower()), None)
    if deployer_col:
        df["deployers_parsed"] = df[deployer_col].apply(parse_list_field)
        # Explode to one row per deployer
        exploded = df.explode("deployers_parsed")
        deployer_counts = exploded["deployers_parsed"].value_counts()
        deployer_counts.head(20).to_csv(OUTPUT_DIR / "top_deployers.csv")
        print(f"\nTop 10 deployers:")
        for d, n in deployer_counts.head(10).items():
            print(f"  {d}: {n}")

    # Load CSET classifications
    cset_path = mongodump_dir / "classifications_CSETv1.csv"
    if cset_path.exists():
        print(f"\nLoading CSET v1 classifications...")
        cset = pd.read_csv(cset_path)
        print(f"Classifications: {len(cset)} rows, columns: {list(cset.columns)[:15]}")

        # Find harm-related columns
        harm_cols = [c for c in cset.columns if "harm" in c.lower()]
        sector_cols = [c for c in cset.columns if "sector" in c.lower() or "industry" in c.lower()]
        ai_cols = [c for c in cset.columns if "ai" in c.lower() or "system" in c.lower()]

        print(f"  Harm columns: {harm_cols[:5]}")
        print(f"  Sector columns: {sector_cols[:5]}")
        print(f"  AI columns: {ai_cols[:5]}")

        cset.to_csv(OUTPUT_DIR / "classifications_cset.csv", index=False)

        # Summary of key categorical fields
        for col in harm_cols[:3] + sector_cols[:2]:
            if col in cset.columns:
                counts = cset[col].value_counts()
                print(f"\n  {col}:")
                for v, n in counts.head(8).items():
                    print(f"    {v}: {n}")

    # Load GMF classifications (broader coverage)
    gmf_path = mongodump_dir / "classifications_GMF.csv"
    if gmf_path.exists():
        print(f"\nLoading GMF classifications...")
        gmf = pd.read_csv(gmf_path)
        print(f"GMF: {len(gmf)} rows, columns: {list(gmf.columns)[:10]}")
        gmf.to_csv(OUTPUT_DIR / "classifications_gmf.csv", index=False)

    print(f"\nDone. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
