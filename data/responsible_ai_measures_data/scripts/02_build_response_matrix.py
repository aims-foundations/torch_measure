"""
Explore and process Responsible AI Measures Dataset.

Source: Figshare (https://figshare.com/articles/dataset/29551001)
Paper: Rismani et al., Nature Scientific Data 2025

This is NOT a standard response matrix (no models evaluated against items).
It is an item bank: 791 evaluation measures that COULD be used to construct
an audit response matrix if systems were evaluated against them.

Output:
  - measures_catalog.csv: cleaned catalog of all 791 measures
  - principle_summary.csv: summary by ethical principle
  - measures_by_principle_and_type.csv: cross-tabulation
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Find the Excel file
    xlsx_files = list(RAW_DIR.rglob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No Excel files found in {RAW_DIR}. Run 01_download_raw.sh first.")

    xlsx_path = xlsx_files[0]
    print(f"Loading {xlsx_path.name}...")

    # Read with header on row 1 (0-indexed), skip the group header row
    df = pd.read_excel(xlsx_path, header=1)
    print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Save full catalog
    df.to_csv(OUTPUT_DIR / "measures_catalog.csv", index=False)
    print(f"\nSaved measures_catalog.csv ({len(df)} measures)")

    # Summary by principle
    principle_col = next((c for c in df.columns if "principle" in c.lower()), None)
    if principle_col:
        principle_counts = df[principle_col].value_counts()
        principle_counts.to_csv(OUTPUT_DIR / "principle_summary.csv")
        print(f"\nMeasures by ethical principle:")
        for p, n in principle_counts.items():
            print(f"  {p}: {n}")

    # Cross-tabulation: principle x assessment type
    type_col = next((c for c in df.columns if "type" in c.lower() and "assess" in c.lower()), None)
    if principle_col and type_col:
        ct = pd.crosstab(df[principle_col], df[type_col])
        ct.to_csv(OUTPUT_DIR / "measures_by_principle_and_type.csv")
        print(f"\nCross-tabulation (principle x assessment type):")
        print(ct.to_string())

    # Summary of other dimensions
    for col_keyword, label in [
        ("harm", "Primary Harm"),
        ("part", "Part of ML System"),
        ("application", "Application Area"),
        ("algorithm", "Algorithm Type"),
    ]:
        col = next((c for c in df.columns if col_keyword in c.lower()), None)
        if col:
            counts = df[col].value_counts()
            print(f"\n{label} ({col}):")
            for v, n in counts.head(10).items():
                print(f"  {v}: {n}")

    # Publication year distribution
    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    if year_col:
        year_counts = df[year_col].value_counts().sort_index()
        print(f"\nPublications by year:")
        for y, n in year_counts.items():
            print(f"  {y}: {n}")


if __name__ == "__main__":
    main()
