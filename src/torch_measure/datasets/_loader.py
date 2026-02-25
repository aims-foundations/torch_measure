# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Dataset download and loading pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_measure.datasets._registry import info as _info

if TYPE_CHECKING:
    from torch_measure.data.pairwise import PairwiseComparisons
    from torch_measure.data.response_matrix import ResponseMatrix


def load(name: str, *, force_download: bool = False) -> ResponseMatrix | PairwiseComparisons:
    """Load a dataset by name, downloading from HuggingFace Hub if needed.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"helm/mmlu"`` or ``"arena/chatbot_arena"``).
        Use :func:`list_datasets` to see available names.
    force_download : bool
        If ``True``, re-download even if cached locally.

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
        If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            "Loading datasets requires huggingface_hub. Install with: pip install torch-measure[data]"
        ) from err

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

    if dataset_info.response_type == "pairwise":
        return _load_pairwise(payload, filename)
    return _load_response_matrix(payload, filename)


def _load_response_matrix(payload: dict | torch.Tensor, filename: str) -> ResponseMatrix:
    """Deserialize a response matrix payload."""
    from torch_measure.data.response_matrix import ResponseMatrix

    if isinstance(payload, dict):
        data = payload["data"]
        subject_ids = payload.get("subject_ids")
        item_ids = payload.get("item_ids")
        item_contents = payload.get("item_contents")
        subject_metadata = payload.get("subject_metadata")
    elif isinstance(payload, torch.Tensor):
        data = payload
        subject_ids = None
        item_ids = None
        item_contents = None
        subject_metadata = None
    else:
        raise TypeError(f"Unexpected payload type in {filename}: {type(payload)}")

    return ResponseMatrix(
        data=data,
        subject_ids=subject_ids,
        item_ids=item_ids,
        item_contents=item_contents,
        subject_metadata=subject_metadata,
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
