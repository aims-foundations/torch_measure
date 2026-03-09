#!/usr/bin/env python3
"""Create per-domain and combined item_content CSV files for TAU-bench.

Fixes two issues:
1. Combined RM uses composite IDs (e.g., airline_v1_0) that don't match
   task_metadata.csv bare IDs (e.g., 0). Creates item_content_combined.csv
   with composite IDs as item_id.
2. Per-domain RMs share task_metadata.csv which has duplicate task_ids across
   domains/benchmarks, causing cross-contamination via dict(zip()). Creates
   separate item_content files filtered to the correct domain.

Domain mapping (RM domain -> metadata benchmark + domain):
  airline_v1      -> tau-bench-v1 / airline
  airline_v2      -> tau2-bench   / airline
  airline_v1_hal  -> tau-bench-v1 / airline  (same tasks as v1, different agents)
  retail          -> tau-bench-v1 / retail
  telecom         -> tau2-bench   / telecom
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "taubench_data" / "processed"

# ---------------------------------------------------------------------------
# Mapping: (benchmark, domain) -> per-domain RM file suffix
# ---------------------------------------------------------------------------
# Combined RM uses domain prefixes: airline_v1, airline_v2, retail, telecom
# Per-domain RMs: v1_airline, v2_airline, hal_airline, retail, telecom

DOMAIN_MAP = {
    # key: (per-domain RM suffix, RM domain column value)
    # value: (metadata benchmark, metadata domain)
    "v1_airline":  ("tau-bench-v1", "airline"),
    "v2_airline":  ("tau2-bench",   "airline"),
    "hal_airline": ("tau-bench-v1", "airline"),  # HAL uses v1 airline tasks
    "retail":      ("tau-bench-v1", "retail"),
    "telecom":     ("tau2-bench",   "telecom"),
}

# Combined RM domain -> (metadata benchmark, metadata domain)
COMBINED_DOMAIN_MAP = {
    "airline_v1": ("tau-bench-v1", "airline"),
    "airline_v2": ("tau2-bench",   "airline"),
    "retail":     ("tau-bench-v1", "retail"),
    "telecom":    ("tau2-bench",   "telecom"),
}


def main() -> None:
    # Load metadata
    meta = pd.read_csv(PROCESSED_DIR / "task_metadata.csv")
    meta["task_id"] = meta["task_id"].astype(str)
    print(f"Loaded task_metadata.csv: {len(meta)} rows")
    print(f"  Benchmarks: {meta['benchmark'].unique().tolist()}")
    print(f"  Domains: {meta['domain'].unique().tolist()}")
    print()

    # ── Fix 1: Create item_content_combined.csv ──────────────────────
    combined_rm = pd.read_csv(PROCESSED_DIR / "response_matrix_combined.csv")
    combined_item_ids = combined_rm["task_id"].astype(str).tolist()
    combined_domains = combined_rm["domain"].astype(str).tolist()

    print(f"Combined RM: {len(combined_item_ids)} items")

    rows_combined = []
    for item_id, rm_domain in zip(combined_item_ids, combined_domains):
        bench, meta_domain = COMBINED_DOMAIN_MAP[rm_domain]
        # Extract the bare task_id by stripping the domain prefix
        # e.g., "airline_v1_0" -> strip "airline_v1_" -> "0"
        # e.g., "retail_0" -> strip "retail_" -> "0"
        # e.g., "telecom_[mms_issue]..." -> strip "telecom_" -> "[mms_issue]..."
        prefix = rm_domain + "_"
        bare_id = item_id[len(prefix):]

        # Look up in metadata
        match = meta[(meta["benchmark"] == bench) & (meta["domain"] == meta_domain) & (meta["task_id"] == bare_id)]
        if len(match) == 1:
            content = str(match.iloc[0]["instruction"]) if pd.notna(match.iloc[0]["instruction"]) else ""
        elif len(match) > 1:
            content = str(match.iloc[0]["instruction"]) if pd.notna(match.iloc[0]["instruction"]) else ""
            print(f"  WARNING: Multiple matches for {item_id} (bench={bench}, domain={meta_domain}, task_id={bare_id})")
        else:
            content = ""

        rows_combined.append({"item_id": item_id, "content": content})

    df_combined = pd.DataFrame(rows_combined)
    n_matched = sum(1 for c in df_combined["content"] if c)
    print(f"  Combined content: {n_matched}/{len(df_combined)} items matched")
    df_combined.to_csv(PROCESSED_DIR / "item_content_combined.csv", index=False)
    print(f"  Saved: item_content_combined.csv")
    print()

    # ── Fix 2: Create per-domain item_content files ──────────────────
    for rm_suffix, (bench, meta_domain) in DOMAIN_MAP.items():
        rm_path = PROCESSED_DIR / f"response_matrix_{rm_suffix}.csv"
        if not rm_path.exists():
            print(f"  WARNING: {rm_path.name} not found, skipping")
            continue

        rm = pd.read_csv(rm_path)
        item_ids = rm["task_id"].astype(str).tolist()

        # Filter metadata to the correct benchmark+domain
        filtered_meta = meta[(meta["benchmark"] == bench) & (meta["domain"] == meta_domain)].copy()
        content_map = dict(zip(filtered_meta["task_id"], filtered_meta["instruction"].fillna("")))

        rows = []
        matched = 0
        for iid in item_ids:
            content = str(content_map.get(iid, ""))
            if content:
                matched += 1
            rows.append({"item_id": iid, "content": content})

        df = pd.DataFrame(rows)
        out_name = f"item_content_{rm_suffix}.csv"
        df.to_csv(PROCESSED_DIR / out_name, index=False)
        print(f"  {rm_suffix}: {matched}/{len(item_ids)} items matched -> {out_name}")

    print()
    print("Done! All item_content files created.")


if __name__ == "__main__":
    main()
