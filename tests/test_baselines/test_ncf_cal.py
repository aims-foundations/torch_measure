# Copyright (c) 2026 AIMS Foundations. MIT License.

import argparse
import glob
import os

import numpy as np
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
    parser.add_argument(
        "--cal-size",
        type=int,
        default=50,
        help="Number of labeled samples used for calibration",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=0,
        help="Number of test samples to evaluate NLL on; 0 = use all",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of data held out as the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and calibration sampling",
    )
    return parser.parse_args()


def compute_nll(probs: list[float], labels: list[float]) -> float:
    """Binary negative log-likelihood."""
    probs = np.clip(np.array(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    labels = np.array(labels, dtype=np.float64)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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
    print(f"Total samples: {len(trials)}")

    encoder = SentenceTransformer(args.encoder)
    model = NCF(
        encoder=encoder,
        embedding_dim=args.embed_dim,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
    )

    # Load model head, or load/compute embeddings and train
    if args.model_checkpoint is not None:
        print(f"Loading pre-trained NCF head from {args.model_checkpoint}")
        model.load_head(args.model_checkpoint)
        U, V = None, None
    else:
        if os.path.exists(args.embeddings_checkpoint):
            U, V = model.load_embeddings(args.embeddings_checkpoint)
        else:
            print("Encoding subjects and items")
            U, V = model.encode_batch(trials["display_name"].tolist(), trials["content"].tolist())
            print(f"Saving embeddings to {args.embeddings_checkpoint}")
            torch.save({"subject_embeddings": U, "item_embeddings": V}, args.embeddings_checkpoint)

    # Train / test split
    idx = rng.permutation(len(trials))
    n_test = int(len(trials) * args.test_frac)
    train_idx = idx[n_test:]
    train_trials = trials.iloc[train_idx].reset_index(drop=True)
    test_trials = trials.iloc[idx[:n_test]].reset_index(drop=True)
    print(f"Train: {len(train_trials)}  Test: {len(test_trials)}")

    if U is not None:
        U = U[train_idx]
        V = V[train_idx]
        labels = torch.tensor(train_trials["response"].values, dtype=torch.float32)
        dataset = TensorDataset(torch.cat([U, V], dim=-1), labels)
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


    # Sample calibration set from test trials
    assert args.cal_size < len(test_trials), (
        f"--cal-size ({args.cal_size}) must be smaller than the test set ({len(test_trials)})"
    )
    cal_mask = np.zeros(len(test_trials), dtype=bool)
    cal_mask[rng.choice(len(test_trials), size=args.cal_size, replace=False)] = True
    cal_trials = test_trials[cal_mask].reset_index(drop=True)
    eval_trials = test_trials[~cal_mask].reset_index(drop=True)
    if args.eval_size > 0:
        eval_trials = eval_trials[:args.eval_size]

    cal_labeled = [
        {
            "subject_content": row["display_name"],
            "item_content": row["content"],
            "label": float(row["response"]),
        }
        for _, row in cal_trials.iterrows()
    ]

    # NLL before calibration
    model._round_calibrated = False
    probs_before = [
        model.predict(
            {"subject_content": row["display_name"], "item_content": row["content"]},
            labeled=[],
        )
        for _, row in eval_trials.iterrows()
    ]
    nll_before = compute_nll(probs_before, eval_trials["response"].tolist())
    print(f"\nNLL before calibration: {nll_before:.4f}")

    # NLL after Platt scaling
    model._round_calibrated = False
    probs_after = [
        model.predict(
            {"subject_content": row["display_name"], "item_content": row["content"]},
            labeled=cal_labeled,
        )
        for _, row in eval_trials.iterrows()
    ]
    nll_after = compute_nll(probs_after, eval_trials["response"].tolist())
    print(f"NLL after calibration: {nll_after:.4f}")
    print(f"Platt params — a={model._platt_a:.4f}  b={model._platt_b:.4f}")

    assert nll_after < nll_before, (
        f"Calibration did not improve NLL: {nll_before:.4f} → {nll_after:.4f}"
    )
    print("\nCalibration test passed: NLL improved after Platt scaling.")

