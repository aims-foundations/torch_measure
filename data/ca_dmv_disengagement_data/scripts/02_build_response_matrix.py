#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process CA DMV autonomous vehicle disengagement reports.

Loads disengagement CSVs from raw/ (general and driverless reports), builds:
  1. manufacturer x disengagement-initiator cross-tabulation
  2. manufacturer x location-type cross-tabulation
  3. Summary statistics per manufacturer (total disengagements, date ranges)
  4. Clean combined CSV

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

ENCODING_CANDIDATES = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Try multiple encodings to read a CSV file."""
    for enc in ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Could not read {path} with any of {ENCODING_CANDIDATES}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace, replace newlines with spaces."""
    rename_map = {}
    for col in df.columns:
        cleaned = " ".join(col.strip().split())  # collapse whitespace/newlines
        rename_map[col] = cleaned
    return df.rename(columns=rename_map)


def detect_column(df: pd.DataFrame, candidates: list[str], description: str) -> str | None:
    """Try to find a column from a list of candidate substrings (case-insensitive)."""
    col_lower = {c.lower(): c for c in df.columns}
    # First try exact match
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    # Then try substring match
    for cand in candidates:
        for col_l, col_orig in col_lower.items():
            if cand.lower() in col_l:
                return col_orig
    print(f"  [WARN] Could not find column for '{description}'.")
    print(f"         Available: {list(df.columns)}")
    return None


def normalize_values(series: pd.Series) -> pd.Series:
    """Normalize string values: strip, title-case."""
    return series.astype(str).str.strip().str.title()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CA DMV Autonomous Vehicle Disengagement Report Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {RAW_DIR}")
        return

    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")

    frames = {}
    for f in csv_files:
        df = read_csv_robust(f)
        df = clean_column_names(df)
        # Tag the source file
        if "driverless" in f.name.lower():
            df["_report_type"] = "driverless"
        else:
            df["_report_type"] = "general"
        frames[f.stem] = df
        print(f"\n--- {f.name} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")

    # Combine
    combined = pd.concat(frames.values(), ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")

    # ------------------------------------------------------------------
    # 2. Detect key columns
    # ------------------------------------------------------------------
    col_mfr = detect_column(combined, ["Manufacturer"], "manufacturer")
    col_date = detect_column(combined, ["DATE", "Date", "Report Date"], "date")
    col_initiator = detect_column(
        combined,
        ["DISENGAGEMENT INITIATED BY", "Disengagement Initiated By"],
        "disengagement initiator",
    )
    col_location = detect_column(
        combined,
        ["DISENGAGEMENT LOCATION", "Disengagement Location"],
        "disengagement location",
    )
    col_desc = detect_column(
        combined,
        ["DESCRIPTION OF FACTS", "Description of Facts"],
        "description",
    )
    col_driverless = detect_column(
        combined,
        ["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"],
        "driverless capable",
    )
    col_driver_present = detect_column(
        combined,
        ["DRIVER PRESENT"],
        "driver present",
    )

    # ------------------------------------------------------------------
    # 3. Normalize key fields
    # ------------------------------------------------------------------
    if col_mfr:
        combined[col_mfr] = combined[col_mfr].str.strip()
    if col_initiator:
        combined["initiator_clean"] = normalize_values(combined[col_initiator])
    if col_location:
        combined["location_clean"] = normalize_values(combined[col_location])

    # ------------------------------------------------------------------
    # 4. Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal disengagement reports: {len(combined)}")
    if col_mfr:
        print(f"Unique manufacturers: {combined[col_mfr].nunique()}")
        print(f"\nDisengagements by manufacturer:")
        print(combined[col_mfr].value_counts().to_string())

    if col_initiator:
        print(f"\nDisengagement initiators (normalized):")
        print(combined["initiator_clean"].value_counts().to_string())

    if col_location:
        print(f"\nDisengagement locations (normalized):")
        print(combined["location_clean"].value_counts().to_string())

    print(f"\nReport type breakdown:")
    print(combined["_report_type"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 5. Build cross-tabulation matrices
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING CROSS-TABULATION MATRICES")
    print("=" * 70)

    # 5a. Manufacturer x Disengagement Initiator
    if col_mfr and col_initiator:
        ct_initiator = pd.crosstab(
            combined[col_mfr],
            combined["initiator_clean"],
            margins=True,
            margins_name="TOTAL",
        )
        out_path = PROCESSED_DIR / "manufacturer_x_initiator.csv"
        ct_initiator.to_csv(out_path)
        print(f"\n  Manufacturer x Initiator: {ct_initiator.shape}")
        print(f"  Saved to: {out_path}")
        print(ct_initiator.to_string())

    # 5b. Manufacturer x Location
    if col_mfr and col_location:
        ct_location = pd.crosstab(
            combined[col_mfr],
            combined["location_clean"],
            margins=True,
            margins_name="TOTAL",
        )
        out_path = PROCESSED_DIR / "manufacturer_x_location.csv"
        ct_location.to_csv(out_path)
        print(f"\n  Manufacturer x Location: {ct_location.shape}")
        print(f"  Saved to: {out_path}")
        print(ct_location.to_string())

    # 5c. Binary response matrix: manufacturer x location (1 if any disengagement)
    if col_mfr and col_location:
        ct_raw = pd.crosstab(combined[col_mfr], combined["location_clean"])
        binary = (ct_raw > 0).astype(int)
        out_path = PROCESSED_DIR / "response_matrix_binary.csv"
        binary.to_csv(out_path)
        print(f"\n  Binary response matrix (mfr x location): {binary.shape}")
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 6. Per-manufacturer summary
    # ------------------------------------------------------------------
    if col_mfr:
        mfr_stats = []
        for mfr, grp in combined.groupby(col_mfr):
            row = {"manufacturer": mfr, "total_disengagements": len(grp)}
            if col_date:
                dates = pd.to_datetime(grp[col_date], errors="coerce", format="mixed")
                row["earliest_date"] = dates.min()
                row["latest_date"] = dates.max()
            if col_initiator:
                row["most_common_initiator"] = grp["initiator_clean"].mode().iloc[0] if len(grp) > 0 else np.nan
            if col_location:
                row["most_common_location"] = grp["location_clean"].mode().iloc[0] if len(grp) > 0 else np.nan
            row["report_type_general"] = (grp["_report_type"] == "general").sum()
            row["report_type_driverless"] = (grp["_report_type"] == "driverless").sum()
            mfr_stats.append(row)

        mfr_df = pd.DataFrame(mfr_stats).sort_values("total_disengagements", ascending=False)
        out_path = PROCESSED_DIR / "manufacturer_summary.csv"
        mfr_df.to_csv(out_path, index=False)
        print(f"\n  Per-manufacturer summary: {mfr_df.shape}")
        print(f"  Saved to: {out_path}")
        print(mfr_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Save clean combined CSV
    # ------------------------------------------------------------------
    keep_cols = [c for c in [
        col_mfr, col_date, "initiator_clean", "location_clean",
        col_driverless, col_driver_present, col_desc, "_report_type",
    ] if c is not None and c in combined.columns]

    clean = combined[keep_cols].copy()
    out_path = PROCESSED_DIR / "disengagements_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"\n  Clean combined CSV: {clean.shape}")
    print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("CA DMV disengagement processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
