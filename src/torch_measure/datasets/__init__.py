# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Built-in benchmark datasets for torch_measure.

Usage::

    from torch_measure.datasets import load, list_datasets, info, load_csv

    list_datasets()              # see all available datasets
    info("helm/mmlu")            # inspect metadata without downloading
    rm = load("helm/mmlu")       # download and return a ResponseMatrix
    rm = load_csv("data/bfcl_data/processed/response_matrix.csv")  # from local CSV
"""

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._loader import load, load_csv
from torch_measure.datasets._registry import info, list_datasets

__all__ = [
    "DatasetInfo",
    "info",
    "list_datasets",
    "load",
    "load_csv",
]
