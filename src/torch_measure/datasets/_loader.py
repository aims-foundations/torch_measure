# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Dataset download and loading pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_measure.datasets._registry import info as _info

if TYPE_CHECKING:
    from torch_measure.data.response_matrix import ResponseMatrix


def load(name: str, *, force_download: bool = False) -> ResponseMatrix:
    """Load a dataset by name, downloading from HuggingFace Hub if needed.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"helm/mmlu"``).
        Use :func:`list_datasets` to see available names.
    force_download : bool
        If ``True``, re-download even if cached locally.

    Returns
    -------
    ResponseMatrix
        Response matrix with ``subject_ids`` and ``item_ids`` when available.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            "Loading datasets requires huggingface_hub. Install with: pip install torch-measure[data]"
        ) from err

    from torch_measure.data.response_matrix import ResponseMatrix

    dataset_info = _info(name)

    # Determine filename — default convention: ``family/benchmark.pt``
    filename = dataset_info.filename or f"{name}.pt"

    path = hf_hub_download(
        repo_id=dataset_info.repo_id,
        filename=filename,
        repo_type="dataset",
        force_download=force_download,
    )

    payload = torch.load(path, weights_only=True)

    if isinstance(payload, dict):
        data = payload["data"]
        subject_ids = payload.get("subject_ids")
        item_ids = payload.get("item_ids")
        item_contents = payload.get("item_contents")
        subject_contents = payload.get("subject_contents")
    elif isinstance(payload, torch.Tensor):
        data = payload
        subject_ids = None
        item_ids = None
        item_contents = None
        subject_contents = None
    else:
        raise TypeError(f"Unexpected payload type in {filename}: {type(payload)}")

    return ResponseMatrix(
        data=data,
        subject_ids=subject_ids,
        item_ids=item_ids,
        item_contents=item_contents,
        subject_contents=subject_contents,
    )
