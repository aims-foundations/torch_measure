# Copyright (c) 2026 AIMS Foundations. MIT License.

import argparse
import glob
import os

import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from torch_measure.models import NCF


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate an NCF model.")
    parser.add_argument(
        "--encoder",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for encoding",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=384,
        help="Embedding dimension of the encoder",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Batch size used when encoding subjects/items",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the training DataLoader",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--embeddings-checkpoint",
        type=str,
        default="ncf_embeddings.pt",
        help="Path to save/load encoded subject and item tensors",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained NCF head state dict to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ncf_head.pt",
        help="Path to save the trained NCF head state dict",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Download data snapshot and load lookup tables
    snapshot_path = snapshot_download(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
    )
    subjects_df = pd.read_parquet(os.path.join(snapshot_path, "subjects.parquet"))
    items_df = pd.read_parquet(os.path.join(snapshot_path, "items.parquet"))

    # Load trial files (skip _traces, subjects, items, benchmarks)
    skip = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    trial_files = [
        f
        for f in glob.glob(os.path.join(snapshot_path, "*.parquet"))
        if not os.path.basename(f).endswith("_traces.parquet") and os.path.basename(f) not in skip
    ]
    print(f"\nLoading {len(trial_files)} trial files")
    trials = pd.concat(
        [pd.read_parquet(f, columns=["subject_id", "item_id", "response"]) for f in trial_files],
        ignore_index=True,
    ).dropna(subset=["response"])
    # Keep only binary pass/fail labels
    trials = trials[trials["response"].isin([0.0, 1.0])]
    trials = trials.merge(subjects_df, on="subject_id", how="inner").merge(items_df, on="item_id", how="inner")
    print(f"Total training samples: {len(trials)}")

    labels = torch.tensor(trials["response"].values, dtype=torch.float32)

    encoder = SentenceTransformer(args.encoder)
    model = NCF(
        encoder=encoder,
        embedding_dim=args.embed_dim,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
    )

    # Load or initialize NCF head
    if args.model_checkpoint is not None:
        print(f"Loading pre-trained NCF head from {args.model_checkpoint}")
        model.load_head(args.model_checkpoint)
    else:
        # Load or compute embeddings for training
        if os.path.exists(args.embeddings_checkpoint):
            U, V = model.load_embeddings(args.embeddings_checkpoint)
        else:
            subjects = trials["display_name"].tolist()
            items = trials["content"].tolist()
            print("Encoding subjects and items")
            U, V = model.encode_batch(subjects, items)
            print(f"Saving embeddings to {args.embeddings_checkpoint}")
            torch.save(
                {"subject_embeddings": U, "item_embeddings": V},
                args.embeddings_checkpoint,
            )
        X = torch.cat([U, V], dim=-1)
        dataset = TensorDataset(X, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = AdamW(model.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        print("Training NCF head")
        for epoch in range(args.epochs):
            total_loss = 0.0
            model.net.train()
            for xb, yb in loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                optimizer.zero_grad()
                logits = model.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
            print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss / len(dataset):.4f}")

        torch.save(model.net.state_dict(), args.output)
        print(f"Saved {args.output}")
