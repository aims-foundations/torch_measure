# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Built-in benchmark datasets for torch_measure.

Usage::

    from torch_measure.datasets import load, list_datasets, info

    list_datasets()              # see all available datasets
    info("helm/mmlu")            # inspect metadata without downloading
    rm = load("helm/mmlu")       # download and return a ResponseMatrix
"""

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._loader import load
from torch_measure.datasets._registry import info, list_datasets

__all__ = [
    "DatasetInfo",
    "info",
    "list_datasets",
    "load",
]
