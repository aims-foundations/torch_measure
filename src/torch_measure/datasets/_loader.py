# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Dataset download and loading pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from torch_measure.datasets._registry import info as _info

if TYPE_CHECKING:
    from torch_measure.data.pairwise import PairwiseComparisons
    from torch_measure.data.response_matrix import ResponseMatrix


def load(
    name: str,
    *,
    force_download: bool = False,
    local_dir: str | Path | None = None,
) -> ResponseMatrix | PairwiseComparisons:
    """Load a dataset by name, downloading from HuggingFace Hub if needed.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"swebench"`` or ``"chatbot_arena"``).
        Use :func:`list_datasets` to see available names.
    force_download : bool
        If ``True``, re-download even if cached locally.
    local_dir : str | Path | None
        If provided, look for ``.pt`` files in this directory before
        downloading from HuggingFace Hub.  Useful for offline use or
        loading data produced by ``data/scripts/upload_to_hf.py --no-upload``.
        The file is expected at ``<local_dir>/<filename>`` where *filename*
        is the ``filename`` field from the dataset registry (e.g.,
        ``swebench.pt``).

    Returns
    -------
    ResponseMatrix | PairwiseComparisons
        A :class:`~torch_measure.data.ResponseMatrix` for binary/continuous
        datasets, or a :class:`~torch_measure.data.PairwiseComparisons` for
        pairwise preference datasets.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    ImportError
        If ``huggingface_hub`` is not installed and no local file is found.
    """
    dataset_info = _info(name)

    # Determine filename — default convention: ``family/benchmark.pt``
    filename = dataset_info.filename or f"{name}.pt"

    # ── Try local directory first ──────────────────────────────────────
    if local_dir is not None:
        local_path = Path(local_dir) / filename
        if local_path.exists():
            payload = torch.load(local_path, weights_only=True)
            if dataset_info.response_type == "pairwise":
                return _load_pairwise(payload, filename)
            return _load_response_matrix(payload, filename)

    # ── Download from HuggingFace Hub ──────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            "Loading datasets requires huggingface_hub. Install with: pip install torch_measure[data]"
        ) from err

    path = hf_hub_download(
        repo_id=dataset_info.repo_id,
        filename=filename,
        repo_type="dataset",
        force_download=force_download,
    )

    payload = torch.load(path, weights_only=True)

    if dataset_info.response_type == "pairwise":
        return _load_pairwise(payload, filename)
    return _load_response_matrix(payload, filename)


def load_csv(
    path: str | Path,
    *,
    index_col: int | str = 0,
) -> ResponseMatrix:
    """Load a CSV response matrix into a :class:`ResponseMatrix`.

    This is a convenience function for loading CSV files produced by the
    benchmark processing scripts (``data/<benchmark>_data/processed/response_matrix.csv``).

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.  Expects rows = subjects (models),
        columns = items (tasks), with the first column or a named column
        as the subject index.
    index_col : int | str
        Column to use as subject IDs (passed to ``pd.read_csv``).

    Returns
    -------
    ResponseMatrix

    Examples
    --------
    >>> from torch_measure.datasets import load_csv
    >>> rm = load_csv("data/bfcl_data/processed/response_matrix.csv")
    >>> rm.shape
    (93, 4751)
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError("load_csv requires pandas. Install with: pip install pandas") from err

    from torch_measure.data.response_matrix import ResponseMatrix

    df = pd.read_csv(path, index_col=index_col)
    data = torch.tensor(df.values, dtype=torch.float32)

    return ResponseMatrix(
        data=data,
        subject_ids=list(df.index.astype(str)),
        item_ids=list(df.columns.astype(str)),
    )


def _load_response_matrix(payload: dict | torch.Tensor, filename: str) -> ResponseMatrix:
    """Deserialize a response matrix payload."""
    from torch_measure.data.response_matrix import ResponseMatrix

    if isinstance(payload, dict):
        data = payload["data"]
        subject_ids = payload.get("subject_ids")
        item_ids = payload.get("item_ids")
        item_contents = payload.get("item_contents")
        subject_metadata = payload.get("subject_metadata")
        info = payload.get("info")
    elif isinstance(payload, torch.Tensor):
        data = payload
        subject_ids = None
        item_ids = None
        item_contents = None
        subject_metadata = None
        info = None
    else:
        raise TypeError(f"Unexpected payload type in {filename}: {type(payload)}")

    return ResponseMatrix(
        data=data,
        subject_ids=subject_ids,
        item_ids=item_ids,
        item_contents=item_contents,
        subject_metadata=subject_metadata,
        info=info,
    )


def _load_pairwise(payload: dict | torch.Tensor, filename: str) -> PairwiseComparisons:
    """Deserialize a pairwise comparisons payload."""
    from torch_measure.data.pairwise import PairwiseComparisons

    if not isinstance(payload, dict):
        raise TypeError(f"Pairwise dataset {filename} must be a dict payload, got {type(payload)}")

    return PairwiseComparisons(
        subject_a=payload["subject_a"],
        subject_b=payload["subject_b"],
        outcome=payload["outcome"],
        subject_ids=payload["subject_ids"],
        item_ids=payload.get("item_ids"),
        item_contents=payload.get("item_contents"),
        item_idx=payload.get("item_idx"),
        subject_metadata=payload.get("subject_metadata"),
        comparison_metadata=payload.get("comparison_metadata"),
    )
