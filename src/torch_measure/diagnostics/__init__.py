# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Item-level diagnostics for AI benchmark response matrices.

Implements the ensemble flagging procedure from
arXiv:2511.16842 ("Fantastic Bugs and Where to Find Them in AI
Benchmarks"). Operates on a binary response matrix and surfaces items
whose per-item statistics violate Rasch-model implications, optionally
routing flagged items through a pluggable LLM-judge second pass.
"""

from torch_measure.diagnostics._ensemble import flag_items, gaussian_rank
from torch_measure.diagnostics._judge import ItemJudge
from torch_measure.diagnostics._signals import (
    average_tetrachoric_correlation,
    item_scalability,
    item_total_correlation,
    tetrachoric_correlation,
)

__all__ = [
    "ItemJudge",
    "average_tetrachoric_correlation",
    "flag_items",
    "gaussian_rank",
    "item_scalability",
    "item_total_correlation",
    "tetrachoric_correlation",
]
