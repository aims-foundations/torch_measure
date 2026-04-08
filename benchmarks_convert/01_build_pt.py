"""
Build long-format .pt files from wide-format response matrices.

Reads benchmark_registry.yaml and, for each benchmark, converts the wide
response_matrix.csv into a long-format DataFrame saved as a .pt file.

Usage:
    python benchmarks/build_benchmark_pt.py                 # all benchmarks
    python benchmarks/build_benchmark_pt.py swebench bbq    # specific ones
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = SCRIPT_DIR / "benchmark_registry.yaml"
OUTPUT_DIR = SCRIPT_DIR / "pt"

COLUMNS = [
    "dataset_name",
    "test_taker",
    "item",
    "item_text",
    "test_condition",
    "benchmark_url",
    "response",
    "category",
]


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def read_response_matrix(csv_path: Path, model_column: str) -> pd.DataFrame:
    """Read wide-format CSV. First column is the test_taker identifier."""
    df = pd.read_csv(csv_path, dtype=str)
    # Normalise the first column name
    first_col = df.columns[0]
    if first_col != model_column and model_column:
        # Registry might say "" for blank headers
        pass
    df = df.rename(columns={first_col: "__test_taker__"})
    return df


def melt_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide → long: one row per (test_taker, item)."""
    item_cols = [c for c in df.columns if c != "__test_taker__"]
    long = df.melt(
        id_vars="__test_taker__",
        value_vars=item_cols,
        var_name="item",
        value_name="response",
    )
    long = long.rename(columns={"__test_taker__": "test_taker"})
    long["response"] = pd.to_numeric(long["response"], errors="coerce")
    long = long.dropna(subset=["response"])
    return long


def transpose_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose: items-as-rows → models-as-rows."""
    item_col = df.columns[0]
    items = df[item_col].values
    model_names = df.columns[1:]
    data = df.iloc[:, 1:].T
    data.columns = items
    data.insert(0, "__test_taker__", model_names)
    return data


def extract_category_from_pattern(items: pd.Series, pattern: str) -> pd.Series:
    """Apply regex to item names to extract category."""
    compiled = re.compile(pattern)

    def _extract(item_id):
        m = compiled.match(str(item_id))
        return m.group(1) if m else ""

    return items.map(_extract)


def join_metadata(
    long: pd.DataFrame, meta_cfg: dict, data_root: Path, item_order: list[str] | None = None,
) -> pd.DataFrame:
    """Left-join metadata onto the long DataFrame.

    If ``item_order`` is provided and ``positional_join`` is set in meta_cfg,
    metadata rows are matched positionally to the item column order instead
    of by key.
    """
    meta_path = data_root / meta_cfg["file"]
    if not meta_path.exists():
        print(f"  WARNING: metadata file not found: {meta_path}")
        return long

    meta = pd.read_csv(meta_path, dtype=str)
    item_key = meta_cfg["item_key"]

    # Positional join: metadata rows correspond 1:1 to response matrix columns
    if meta_cfg.get("positional_join") and item_order is not None:
        if len(meta) == len(item_order):
            meta["__item_name__"] = item_order
            item_key = "__item_name__"
        else:
            print(f"  WARNING: positional join size mismatch: {len(meta)} metadata vs {len(item_order)} items")

    if item_key not in meta.columns:
        print(f"  WARNING: item_key '{item_key}' not in metadata columns: {list(meta.columns)}")
        return long

    # Ensure join key types match (both as string)
    meta[item_key] = meta[item_key].astype(str)
    long["item"] = long["item"].astype(str)

    # Build the columns we want from metadata, renaming to avoid collisions
    category_col = meta_cfg.get("category_column")
    condition_col = meta_cfg.get("test_condition_column")
    text_cols = meta_cfg.get("item_text_columns", [])

    rename_map = {}
    cols_to_join = [item_key]
    if category_col and category_col in meta.columns:
        rename_map[category_col] = "__meta_category__"
        cols_to_join.append(category_col)
    if condition_col and condition_col in meta.columns:
        rename_map[condition_col] = "__meta_condition__"
        cols_to_join.append(condition_col)
    valid_text_cols = [c for c in text_cols if c in meta.columns]
    for c in valid_text_cols:
        if c not in cols_to_join:
            cols_to_join.append(c)

    if len(cols_to_join) <= 1:
        return long

    meta_subset = meta[cols_to_join].drop_duplicates(subset=[item_key])
    meta_subset = meta_subset.rename(columns=rename_map)
    join_key = rename_map.get(item_key, item_key)

    merged = long.merge(meta_subset, left_on="item", right_on=join_key, how="left")

    if "__meta_category__" in merged.columns:
        merged["category"] = merged["__meta_category__"].fillna("")
        merged = merged.drop(columns=["__meta_category__"])

    if "__meta_condition__" in merged.columns:
        merged["test_condition"] = merged["__meta_condition__"].fillna("")
        merged = merged.drop(columns=["__meta_condition__"])

    # Combine text columns into item_text
    if valid_text_cols:
        merged["item_text"] = merged[valid_text_cols].fillna("").agg(" ".join, axis=1).str.strip()
        drop_cols = [c for c in valid_text_cols if c != "item_text"]
        merged = merged.drop(columns=drop_cols, errors="ignore")

    # Drop the metadata key column if it duplicates item
    if item_key in merged.columns and item_key != "item":
        merged = merged.drop(columns=[item_key])

    return merged


def build_one(benchmark: dict, data_root: Path) -> pd.DataFrame:
    """Process a single benchmark into a long-format DataFrame."""
    name = benchmark["name"]
    csv_path = data_root / benchmark["response_matrix"]

    if not csv_path.exists():
        print(f"  SKIP: response matrix not found: {csv_path}")
        return pd.DataFrame()

    print(f"  Reading {csv_path.name} ...", end=" ")
    df = read_response_matrix(csv_path, benchmark.get("model_column", ""))

    if benchmark.get("transposed", False):
        print("(transposing) ...", end=" ")
        df = transpose_matrix(df)

    # Capture item column order before melting (for positional metadata joins)
    item_order = [c for c in df.columns if c != "__test_taker__"]

    long = melt_matrix(df)
    print(f"{len(long):,} rows")

    # Add fixed columns
    long["dataset_name"] = name
    long["benchmark_url"] = benchmark.get("url", "")
    if "test_condition" not in long.columns:
        long["test_condition"] = ""
    if "category" not in long.columns:
        long["category"] = ""

    # Extract category from item name pattern
    pattern = benchmark.get("category_pattern")
    if pattern:
        extracted = extract_category_from_pattern(long["item"], pattern)
        # Only overwrite where category is still empty
        mask = long["category"] == ""
        long.loc[mask, "category"] = extracted[mask]

    # Join metadata
    meta_cfg = benchmark.get("metadata")
    if meta_cfg:
        long = join_metadata(long, meta_cfg, data_root, item_order=item_order)

    # Keep only the target columns in order
    for col in COLUMNS:
        if col not in long.columns:
            long[col] = ""

    long = long[COLUMNS]

    # Cast string columns to categorical
    for col in COLUMNS:
        if col == "response":
            long[col] = long[col].astype("float32")
        else:
            long[col] = long[col].astype("category")

    return long


def main():
    parser = argparse.ArgumentParser(description="Build long-format .pt from response matrices")
    parser.add_argument("benchmarks", nargs="*", help="Specific benchmark names (default: all)")
    args = parser.parse_args()

    registry = load_registry()
    data_root = PROJECT_ROOT / registry["data_root"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_benchmarks = {b["name"]: b for b in registry["benchmarks"]}

    if args.benchmarks:
        names = args.benchmarks
        missing = [n for n in names if n not in all_benchmarks]
        if missing:
            print(f"ERROR: unknown benchmarks: {missing}")
            print(f"Available: {sorted(all_benchmarks.keys())}")
            sys.exit(1)
    else:
        names = list(all_benchmarks.keys())

    for name in names:
        bm = all_benchmarks[name]
        print(f"\n[{name}]")
        long = build_one(bm, data_root)
        if long.empty:
            continue

        out_path = OUTPUT_DIR / f"{name}.pt"
        torch.save(long, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {out_path.name} ({size_mb:.1f} MB, {len(long):,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
