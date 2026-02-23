# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Data loaders for common benchmark datasets (requires huggingface_hub)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_measure.data.response_matrix import ResponseMatrix


def from_csv(path: str, **kwargs) -> ResponseMatrix:
    """Load a response matrix from a CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file. First column is treated as subject IDs,
        remaining columns as items.

    Returns
    -------
    ResponseMatrix
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError("Loading from CSV requires pandas. Install with: pip install pandas") from err

    from torch_measure.data.response_matrix import ResponseMatrix

    df = pd.read_csv(path, index_col=0, **kwargs)
    return ResponseMatrix.from_dataframe(df)


def from_huggingface(repo_id: str, filename: str | None = None, **kwargs) -> ResponseMatrix:
    """Load a response matrix from a HuggingFace dataset.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g., "stair-lab/reeval_matrices").
    filename : str | None
        Specific file to load from the repo. If None, loads the default.

    Returns
    -------
    ResponseMatrix
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError("Loading from HuggingFace requires huggingface_hub. Install with: pip install torch-measure[data]") from err

    from torch_measure.data.response_matrix import ResponseMatrix

    if filename is not None:
        path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        if filename.endswith(".pt"):
            data = torch.load(path, weights_only=True)
            return ResponseMatrix(data)
        elif filename.endswith(".csv"):
            return from_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    # Try common filenames
    for name in ["response_matrix.pt", "response_matrix.csv", "data.pt", "data.csv"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=name, **kwargs)
            if name.endswith(".pt"):
                data = torch.load(path, weights_only=True)
                return ResponseMatrix(data)
            else:
                return from_csv(path)
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not find a response matrix in {repo_id}. "
        f"Specify the filename explicitly."
    )
