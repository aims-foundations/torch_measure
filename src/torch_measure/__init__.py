# Copyright (c) 2026 AIMS Foundation. MIT License.

"""torch_measure: PyTorch-native measurement science toolkit for AI evaluation."""

from torch_measure import cat, data, fitting, metrics, models

try:
    from torch_measure._version import version as __version__
except Exception:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "cat",
    "data",
    "fitting",
    "metrics",
    "models",
]
