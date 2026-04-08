"""
Convert individual benchmark .pt files into CSV folders for compile_benchmark_db.py.

For each .pt file in benchmarks/pt/, creates a folder under benchmarks/<name>/
containing:
    items.csv         — item_id, item_text, benchmark, category
    ground_truth.csv  — model_id, item_id, label
    embeddings.npy    — (N_items, 128) float32, L2-normalized, same row order as items.csv

Usage:
    python benchmarks/pt_to_csv_folders.py                    # all benchmarks
    python benchmarks/pt_to_csv_folders.py --no-embeddings    # skip embedding generation
    python benchmarks/pt_to_csv_folders.py swebench bbq       # specific benchmarks
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PT_DIR = SCRIPT_DIR / "pt"
OUTPUT_DIR = SCRIPT_DIR / "_compiled"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 128
BATCH_SIZE = 256

# Skip these — merged file or excluded benchmarks with leftover .pt files
SKIP = {"all_benchmarks"}


def compute_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape[1] > EMBEDDING_DIM:
        embeddings = embeddings[:, :EMBEDDING_DIM]
    elif embeddings.shape[1] < EMBEDDING_DIM:
        padded = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        padded[:, :embeddings.shape[1]] = embeddings
        embeddings = padded

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embeddings = embeddings / norms
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", help="Specific benchmark names (default: all)")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation entirely")
    parser.add_argument("--force-embeddings", action="store_true", help="Regenerate embeddings even if they exist")
    args = parser.parse_args()

    pt_files = sorted(PT_DIR.glob("*.pt"))
    pt_files = [f for f in pt_files if f.stem not in SKIP]

    if args.benchmarks:
        pt_files = [f for f in pt_files if f.stem in args.benchmarks]

    if not pt_files:
        print("No .pt files to convert.")
        return

    # Load embedding model once if needed
    embed_model = None
    if not args.no_embeddings:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nConverting {len(pt_files)} benchmark .pt files to CSV folders:\n")

    for pt_file in pt_files:
        name = pt_file.stem
        print(f"  [{name}]")

        df = torch.load(pt_file, weights_only=False)

        # Convert categoricals to plain strings
        for col in df.columns:
            if df[col].dtype.name == "category":
                df[col] = df[col].astype(str)

        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # items.csv — unique items with metadata
        item_cols = ["item", "dataset_name", "category"]
        if "item_text" in df.columns:
            item_cols.append("item_text")
        items = df[item_cols].drop_duplicates(subset=["item"])
        items = items.rename(columns={
            "item": "item_id",
            "dataset_name": "benchmark",
        })
        if "item_text" not in items.columns:
            items["item_text"] = ""
        items = items[["item_id", "item_text", "benchmark", "category"]]
        items.to_csv(out_dir / "items.csv", index=False)

        # ground_truth.csv — all (model, item, label) triplets
        gt = df[["test_taker", "item", "response"]].copy()
        gt = gt.rename(columns={
            "test_taker": "model_id",
            "item": "item_id",
            "response": "label",
        })
        gt["label"] = gt["label"].astype(int)
        gt.to_csv(out_dir / "ground_truth.csv", index=False)

        # embeddings.npy — one row per item, same order as items.csv
        if embed_model is not None:
            emb_path = out_dir / "embeddings.npy"
            if emb_path.exists() and not args.force_embeddings:
                print(f"    embeddings.npy exists, skipping (use --force-embeddings to regenerate)")
                print(f"    {len(items)} items, {len(gt):,} pairs")
                continue
            texts = items["item_text"].fillna("").tolist()
            # Use fallback for empty texts
            for i, (t, row) in enumerate(zip(texts, items.itertuples())):
                if not t.strip():
                    texts[i] = f"{name}: {row.item_id}"

            print(f"    Encoding {len(texts)} items ...", end=" ")
            emb = embed_model.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.shape[1] > EMBEDDING_DIM:
                emb = emb[:, :EMBEDDING_DIM]
            elif emb.shape[1] < EMBEDDING_DIM:
                padded = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
                padded[:, :emb.shape[1]] = emb
                emb = padded
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            emb = emb / norms
            np.save(emb_path, emb)
            print(f"saved embeddings.npy ({emb.shape})")

        print(f"    {len(items)} items, {len(gt):,} pairs")

    print("\nDone. Run scripts/compile_benchmark_db.py to rebuild benchmarks.sqlite")


if __name__ == "__main__":
    main()
