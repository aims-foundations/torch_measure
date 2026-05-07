# Copyright (c) 2026 AIMS Foundation. MIT License.

"""
Neural Collaborative Filter (NCF) that predicts response matrix entries.

Architecture:
  - Sentence embeddings for both subject and item content
  - Small MLP head trained offline on training data

Training:
  python ncf.py
  -> writes ncf_head.pt into this directory
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

class NCFHead(nn.Module):
    """Neural Collaborative Filter Multi-Layer Perceptron Head.
    
    Maps sentence embeddings encoded by `ENCODER` to a unidimensional output.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# Initialize MLP head for the NCF and load the saved checkpoint
NCF = NCFHead(in_dim=EMBED_DIM * 2).to(DEVICE)
NCF.load_state_dict(torch.load("ncf_head.pt", map_location=DEVICE))
NCF.eval()

def _encode_pair(subject: str, item: str) -> torch.Tensor:
    """Encode a subject-item pair into a single concatenated embedding."""
    u = ENCODER.encode(subject, convert_to_tensor=True, device=DEVICE)
    v = ENCODER.encode(item, convert_to_tensor=True, device=DEVICE)
    return torch.cat([u, v], dim=-1)


def _raw_prob(subject: str, item: str) -> float:
    """Forward pass through the NCF, returns probability in [0, 1]."""
    with torch.no_grad():
        x = _encode_pair(subject, item).unsqueeze(0)
        logit = NCF(x).item()
    return float(1.0 / (1.0 + math.exp(-logit)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict P(subject passes item).
    """
    raw_p = _raw_prob(input["subject_content"], input["item_content"])
    raw_p = float(np.clip(raw_p, 1e-7, 1 - 1e-7))
    return raw_p


# ── Offline training script ───────────────────────────────────────────────────
# Run this once to train the NCF head.
if __name__ == "__main__":
    """
    Usage: python ncf.py
    """
    import glob
    import os
    import pandas as pd
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW

    print("Downloading dataset snapshot...")
    snapshot_path = snapshot_download(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
    )

    # Load lookup tables
    subjects_df = pd.read_parquet(os.path.join(snapshot_path, "subjects.parquet"))
    items_df = pd.read_parquet(os.path.join(snapshot_path, "items.parquet"))
    
    # Load trial files (skip _traces, subjects, items, benchmarks)
    skip = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    trial_files = [
        f for f in glob.glob(os.path.join(snapshot_path, "*.parquet"))
        if not os.path.basename(f).endswith("_traces.parquet")
        and os.path.basename(f) not in skip
    ]
    print(f"Loading {len(trial_files)} trial files...")
    trials = pd.concat(
        [pd.read_parquet(f, columns=["subject_id", "item_id", "response"]) for f in trial_files],
        ignore_index=True,
    ).dropna(subset=["response"])
    # Keep only binary pass/fail labels
    trials = trials[trials["response"].isin([0.0, 1.0])]

    # Join to get text content
    trials = (
        trials
        .merge(subjects_df, on="subject_id", how="inner")
        .merge(items_df, on="item_id", how="inner")
    )
    print(f"Total training samples: {len(trials)}")

    subjects = trials["display_name"].tolist()
    items = trials["content"].tolist()
    labels = torch.tensor(trials["response"].values, dtype=torch.float32)

    print("Encoding subjects and items (this may take a while)...")
    U = torch.tensor(ENCODER.encode(subjects, batch_size=256, show_progress_bar=True))
    V = torch.tensor(ENCODER.encode(items, batch_size=256, show_progress_bar=True))
    X = torch.cat([U, V], dim=-1)

    dataset = TensorDataset(X, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = NCFHead(in_dim=EMBED_DIM * 2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Training NCF head...")
    for epoch in range(10):
        total_loss = 0.0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), "ncf_head.pt")
    print("Saved ncf_head.pt")