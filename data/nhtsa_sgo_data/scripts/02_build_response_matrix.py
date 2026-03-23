#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process NHTSA SGO crash report data.

Loads ADS and ADAS incident CSVs, explores their structure, builds:
  1. manufacturer x crash-type cross-tabulation matrix (counts)
  2. manufacturer x injury-severity cross-tabulation matrix (counts)
  3. Summary statistics for both ADS and ADAS datasets
  4. Combined clean CSV with key fields

Saves outputs to processed/.
"""

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

def find_csv_files(raw_dir: Path, pattern: str = "*.csv") -> list[Path]:
    """Find CSV files matching pattern in raw directory."""
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {raw_dir}")
    return files


def detect_column(df: pd.DataFrame, candidates: list[str], description: str) -> str | None:
    """Try to find a column from a list of candidate names (case-insensitive)."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in col_lower:
            return col_lower[cand.lower().strip()]
    print(f"  [WARN] Could not find column for '{description}'. Tried: {candidates}")
    print(f"         Available columns: {list(df.columns)}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("NHTSA SGO Crash Report Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_files = find_csv_files(RAW_DIR, "SGO-2021-01_Incident_Reports_*.csv")
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")

    frames = {}
    for f in csv_files:
        tag = "ADS" if "_ADS" in f.name else "ADAS"
        df = pd.read_csv(f, low_memory=False)
        df["_source"] = tag
        frames[tag] = df
        print(f"\n--- {tag} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"    {col}")

    # Combine into one dataframe
    combined = pd.concat(frames.values(), ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")

    # ------------------------------------------------------------------
    # 2. Detect key columns
    # ------------------------------------------------------------------
    col_make = detect_column(combined, ["Make", "Manufacturer", "make"], "manufacturer/make")
    col_crash = detect_column(combined, ["Crash With", "Crash_With", "crash with"], "crash type")
    col_injury = detect_column(
        combined,
        ["Highest Injury Severity Alleged", "Injury Severity", "injury_severity"],
        "injury severity",
    )
    col_roadway = detect_column(combined, ["Roadway Type", "roadway_type"], "roadway type")
    col_entity = detect_column(
        combined, ["Reporting Entity", "reporting_entity", "Operating Entity"], "reporting entity"
    )
    col_precrash = detect_column(
        combined,
        ["SV Pre-Crash Movement", "SV Precrash Movement", "sv_precrash_movement"],
        "pre-crash movement",
    )
    col_engaged = detect_column(
        combined,
        ["Automation System Engaged?", "Engagement Status"],
        "automation engaged",
    )

    # ------------------------------------------------------------------
    # 3. Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for tag, df in frames.items():
        print(f"\n--- {tag} ({len(df)} incidents) ---")
        if col_make:
            print(f"  Manufacturers: {df[col_make].nunique()}")
            print(f"  Top 10 makes:\n{df[col_make].value_counts().head(10).to_string()}")
        if col_crash:
            print(f"\n  Crash types:\n{df[col_crash].value_counts().to_string()}")
        if col_injury:
            print(f"\n  Injury severity:\n{df[col_injury].value_counts().to_string()}")
        if col_roadway:
            print(f"\n  Roadway types:\n{df[col_roadway].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 4. Build cross-tabulation matrices
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING CROSS-TABULATION MATRICES")
    print("=" * 70)

    # 4a. Manufacturer x Crash Type (counts)
    if col_make and col_crash:
        for tag, df in frames.items():
            ct = pd.crosstab(df[col_make], df[col_crash], margins=True, margins_name="TOTAL")
            out_path = PROCESSED_DIR / f"make_x_crash_type_{tag.lower()}.csv"
            ct.to_csv(out_path)
            print(f"\n  [{tag}] Manufacturer x Crash Type: {ct.shape}")
            print(f"  Saved to: {out_path}")
            print(ct.head(10).to_string())

        # Also build combined
        ct_all = pd.crosstab(
            combined[col_make], combined[col_crash], margins=True, margins_name="TOTAL"
        )
        out_path = PROCESSED_DIR / "make_x_crash_type_combined.csv"
        ct_all.to_csv(out_path)
        print(f"\n  [Combined] Manufacturer x Crash Type: {ct_all.shape}")
        print(f"  Saved to: {out_path}")

    # 4b. Manufacturer x Injury Severity (counts)
    if col_make and col_injury:
        ct_injury = pd.crosstab(
            combined[col_make], combined[col_injury], margins=True, margins_name="TOTAL"
        )
        out_path = PROCESSED_DIR / "make_x_injury_severity.csv"
        ct_injury.to_csv(out_path)
        print(f"\n  Manufacturer x Injury Severity: {ct_injury.shape}")
        print(f"  Saved to: {out_path}")
        print(ct_injury.head(10).to_string())

    # 4c. Manufacturer x Roadway Type (counts)
    if col_make and col_roadway:
        ct_road = pd.crosstab(
            combined[col_make], combined[col_roadway], margins=True, margins_name="TOTAL"
        )
        out_path = PROCESSED_DIR / "make_x_roadway_type.csv"
        ct_road.to_csv(out_path)
        print(f"\n  Manufacturer x Roadway Type: {ct_road.shape}")
        print(f"  Saved to: {out_path}")

    # 4d. Binary response matrix: manufacturer x incident-type (1 if any incident exists)
    if col_make and col_crash:
        binary = (
            pd.crosstab(combined[col_make], combined[col_crash])
            .drop("TOTAL", axis=0, errors="ignore")
            .drop("TOTAL", axis=1, errors="ignore")
        )
        binary_bool = (binary > 0).astype(int)
        out_path = PROCESSED_DIR / "response_matrix_binary.csv"
        binary_bool.to_csv(out_path)
        print(f"\n  Binary response matrix (make x crash type): {binary_bool.shape}")
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 5. Save clean combined CSV with key fields
    # ------------------------------------------------------------------
    keep_cols = [c for c in [
        "Report ID", col_make, "Model", col_entity, col_engaged,
        col_crash, col_injury, col_roadway, col_precrash,
        "Incident Date", "State", "City", "_source",
    ] if c is not None and c in combined.columns]

    clean = combined[keep_cols].copy()
    out_path = PROCESSED_DIR / "incidents_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"\n  Clean combined CSV: {clean.shape}")
    print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 6. Summary stats CSV
    # ------------------------------------------------------------------
    stats = {
        "metric": [],
        "value": [],
    }
    for tag, df in frames.items():
        stats["metric"].append(f"{tag}_total_incidents")
        stats["value"].append(len(df))
        if col_make:
            stats["metric"].append(f"{tag}_unique_manufacturers")
            stats["value"].append(df[col_make].nunique())
        if col_crash:
            stats["metric"].append(f"{tag}_unique_crash_types")
            stats["value"].append(df[col_crash].nunique())
        if col_injury:
            fatalities = df[col_injury].str.lower().str.contains("fatal", na=False).sum()
            stats["metric"].append(f"{tag}_fatalities")
            stats["value"].append(fatalities)

    stats_df = pd.DataFrame(stats)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics saved to: {out_path}")
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("NHTSA SGO processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
