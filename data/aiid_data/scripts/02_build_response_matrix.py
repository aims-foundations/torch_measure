"""
Process AI Incident Database (AIID) from MongoDB backup CSVs.

Source: https://incidentdatabase.ai/research/snapshots/
Structure: 1,407 incidents with CSET taxonomy classifications

Output:
  - incidents_clean.csv: cleaned incident catalog
  - incidents_by_year.csv: incident counts by year
  - deployer_x_harm.csv: deployer x harm-type cross-tabulation
  - classifications_summary.csv: CSET taxonomy summary
"""

import ast
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw" / "mongodump_full_snapshot"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_list_field(val):
    """Parse stringified list fields like '["entity1","entity2"]'."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [str(val)]


def main():
    incidents_path = RAW_DIR / "incidents.csv"
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
    cset_path = RAW_DIR / "classifications_CSETv1.csv"
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
    gmf_path = RAW_DIR / "classifications_GMF.csv"
    if gmf_path.exists():
        print(f"\nLoading GMF classifications...")
        gmf = pd.read_csv(gmf_path)
        print(f"GMF: {len(gmf)} rows, columns: {list(gmf.columns)[:10]}")
        gmf.to_csv(OUTPUT_DIR / "classifications_gmf.csv", index=False)

    print(f"\nDone. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
