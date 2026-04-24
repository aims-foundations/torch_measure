# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Built-in benchmark datasets for torch_measure.

Usage::

    from torch_measure.datasets import load, list_datasets, info

    list_datasets()                 # see all available datasets
    info("mtbench")                 # inspect metadata without downloading
    data = load("mtbench")          # long-form tables (LongFormData)
    rm = data.to_response_matrix()  # opt-in pivot to wide-form ResponseMatrix
"""

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._loader import load
from torch_measure.datasets._long_form import LongFormData
from torch_measure.datasets._registry import info, list_datasets

__all__ = ["DatasetInfo", "LongFormData", "info", "list_datasets", "load"]
