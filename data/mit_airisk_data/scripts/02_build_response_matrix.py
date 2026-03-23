#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process MIT AI Risk Repository data.

This is a risk catalog (not a traditional response matrix).
Loads CSV exports from raw/ and builds:
  1. Cross-tabulation: domain x causal-entity
  2. Summary statistics on risk categories, domains, severity
  3. Clean catalog CSV

Saves outputs to processed/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_column(df: pd.DataFrame, candidates: list[str], description: str) -> str | None:
    """Find a column by trying candidate names (case-insensitive, then substring)."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in col_lower:
            return col_lower[cand.lower().strip()]
    for cand in candidates:
        for cl, co in col_lower.items():
            if cand.lower() in cl:
                return co
    print(f"  [WARN] Could not find column for '{description}'.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MIT AI Risk Repository Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover and load data
    # ------------------------------------------------------------------
    print(f"\nRaw directory: {RAW_DIR}")
    if RAW_DIR.exists():
        for item in sorted(RAW_DIR.iterdir()):
            if item.is_dir():
                sub_items = list(item.iterdir())
                print(f"  [DIR] {item.name}/ ({len(sub_items)} items)")
                for si in sub_items[:10]:
                    print(f"    {si.name}")
            else:
                print(f"  {item.name} ({item.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [WARN] Raw directory does not exist.")

    df = None

    # Strategy 1: CSV files
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if csv_files:
        print(f"\nFound CSV files: {[f.name for f in csv_files]}")
        frames = {}
        for f in csv_files:
            try:
                d = pd.read_csv(f, low_memory=False)
                frames[f.stem] = d
                print(f"  {f.name}: {d.shape}, columns={d.columns.tolist()}")
            except Exception as e:
                print(f"  [WARN] Failed to load {f.name}: {e}")
        if frames:
            # Use the largest file as primary
            df = max(frames.values(), key=len)
            print(f"\n  Using primary table: {df.shape}")

    # Strategy 2: JSON
    if df is None:
        json_files = sorted(RAW_DIR.glob("*.json"))
        for jf in json_files:
            print(f"\nTrying {jf.name}...")
            try:
                with open(jf) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    for key in ["data", "risks", "entries", "records"]:
                        if key in data and isinstance(data[key], list):
                            df = pd.json_normalize(data[key])
                            break
                if df is not None:
                    print(f"  Loaded: {df.shape}")
                    break
            except Exception as e:
                print(f"  [WARN] Failed: {e}")

    # Strategy 3: HuggingFace dataset
    if df is None:
        dataset_path = RAW_DIR / "dataset"
        if dataset_path.exists():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(dataset_path))
                if hasattr(ds, "keys"):
                    frames = [ds[s].to_pandas() for s in ds]
                    df = pd.concat(frames, ignore_index=True)
                else:
                    df = ds.to_pandas()
                print(f"\n  Loaded HF dataset: {df.shape}")
            except Exception as e:
                print(f"  [WARN] HF load failed: {e}")

    # Strategy 4: Recursive search
    if df is None:
        print("\nSearching recursively for data files...")
        for ext in ["*.csv", "*.json", "*.jsonl", "*.parquet", "*.xlsx"]:
            found = sorted(RAW_DIR.rglob(ext))
            if found:
                print(f"  Found {len(found)} {ext} files")
                for f in found[:5]:
                    print(f"    {f.relative_to(RAW_DIR)} ({f.stat().st_size/1024:.1f} KB)")

        # Try loading the first CSV found recursively
        csv_deep = sorted(RAW_DIR.rglob("*.csv"))
        if csv_deep:
            df = pd.read_csv(csv_deep[0], low_memory=False)
            print(f"\n  Loaded {csv_deep[0].name}: {df.shape}")

    if df is None or df.empty:
        print("\n[WARN] No data files found in raw/. The raw data may need to be downloaded first.")
        print("       Expected: CSV exports from the MIT AI Risk Repository.")
        print("       See 01_download_raw.sh for download instructions.")
        pd.DataFrame({"status": ["no_raw_data"]}).to_csv(
            PROCESSED_DIR / "summary_statistics.csv", index=False
        )
        return

    # ------------------------------------------------------------------
    # 2. Explore data structure
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    print("DATA STRUCTURE")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        nunique = df[col].nunique()
        null_count = df[col].isna().sum()
        sample = str(df[col].dropna().iloc[0])[:80] if df[col].notna().any() else "N/A"
        print(f"  {col}: nunique={nunique}, null={null_count}, sample='{sample}'")

    # ------------------------------------------------------------------
    # 3. Detect key columns
    # ------------------------------------------------------------------
    col_domain = detect_column(
        df,
        ["domain", "Domain", "application_domain", "sector", "area", "field"],
        "domain",
    )
    col_entity = detect_column(
        df,
        ["causal_entity", "entity", "Entity", "cause", "causal entity", "responsible_entity", "actor"],
        "causal entity",
    )
    col_risk = detect_column(
        df,
        ["risk", "risk_category", "Risk Category", "risk_type", "hazard", "category"],
        "risk category",
    )
    col_subcategory = detect_column(
        df,
        ["subcategory", "Subcategory", "sub_category", "risk_subcategory"],
        "subcategory",
    )
    col_severity = detect_column(
        df,
        ["severity", "Severity", "risk_level", "level", "impact"],
        "severity",
    )
    col_description = detect_column(
        df,
        ["description", "Description", "risk_description", "summary", "details"],
        "description",
    )

    print(f"\nDetected columns:")
    print(f"  domain:       {col_domain}")
    print(f"  causal_entity: {col_entity}")
    print(f"  risk:         {col_risk}")
    print(f"  subcategory:  {col_subcategory}")
    print(f"  severity:     {col_severity}")
    print(f"  description:  {col_description}")

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal entries in catalog: {len(df)}")

    if col_domain:
        print(f"\nDomains ({df[col_domain].nunique()}):")
        print(df[col_domain].value_counts().to_string())

    if col_entity:
        print(f"\nCausal entities ({df[col_entity].nunique()}):")
        print(df[col_entity].value_counts().to_string())

    if col_risk:
        print(f"\nRisk categories ({df[col_risk].nunique()}):")
        print(df[col_risk].value_counts().to_string())

    if col_subcategory:
        print(f"\nSubcategories ({df[col_subcategory].nunique()}):")
        print(df[col_subcategory].value_counts().head(20).to_string())

    if col_severity:
        print(f"\nSeverity levels:")
        print(df[col_severity].value_counts().to_string())

    # ------------------------------------------------------------------
    # 5. Build cross-tabulations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-TABULATIONS")
    print("=" * 70)

    # Domain x Causal Entity
    if col_domain and col_entity:
        ct = pd.crosstab(df[col_domain], df[col_entity], margins=True, margins_name="TOTAL")
        out_path = PROCESSED_DIR / "domain_x_causal_entity.csv"
        ct.to_csv(out_path)
        print(f"\n  Domain x Causal Entity: {ct.shape}")
        print(f"  Saved to: {out_path}")
        print(ct.to_string())

    # Domain x Risk Category
    if col_domain and col_risk:
        ct_risk = pd.crosstab(df[col_domain], df[col_risk], margins=True, margins_name="TOTAL")
        out_path = PROCESSED_DIR / "domain_x_risk_category.csv"
        ct_risk.to_csv(out_path)
        print(f"\n  Domain x Risk Category: {ct_risk.shape}")
        print(f"  Saved to: {out_path}")

    # Risk Category x Causal Entity
    if col_risk and col_entity:
        ct_re = pd.crosstab(df[col_risk], df[col_entity], margins=True, margins_name="TOTAL")
        out_path = PROCESSED_DIR / "risk_category_x_causal_entity.csv"
        ct_re.to_csv(out_path)
        print(f"\n  Risk Category x Causal Entity: {ct_re.shape}")
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 6. Save catalog
    # ------------------------------------------------------------------
    out_path = PROCESSED_DIR / "risk_catalog.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Full risk catalog: {df.shape}")
    print(f"  Saved to: {out_path}")

    # Summary stats
    stats = {
        "metric": ["total_entries"],
        "value": [len(df)],
    }
    if col_domain:
        stats["metric"].append("unique_domains")
        stats["value"].append(df[col_domain].nunique())
    if col_entity:
        stats["metric"].append("unique_causal_entities")
        stats["value"].append(df[col_entity].nunique())
    if col_risk:
        stats["metric"].append("unique_risk_categories")
        stats["value"].append(df[col_risk].nunique())
    if col_subcategory:
        stats["metric"].append("unique_subcategories")
        stats["value"].append(df[col_subcategory].nunique())

    stats_df = pd.DataFrame(stats)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics:")
    print(stats_df.to_string(index=False))
    print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("MIT AI Risk Repository processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
