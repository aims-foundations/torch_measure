from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DIMENSION_ORDER: tuple[str, ...] = (
    "AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl", "QLq", "SNs",
    "KNa", "KNc", "KNf", "KNn", "KNs", "AT", "VO", "UG",
)

DEMAND_DIMENSIONS: tuple[str, ...] = DIMENSION_ORDER[:18]
N_DIMENSIONS: int = 19


@dataclass
class Rubric:
    acronym: str
    dimension_name: str
    content: str       # verbatim file text after the # Title line
    rubric_hash: str   # sha256(content)[:16]


@dataclass
class AnnotationJob:
    item_id: str
    content: str
    reference_answer: str


@dataclass
class DemandAnnotation:
    item_id: str
    demand: str        # rubric acronym
    level: float       # 0-5 or math.nan
    finish_reason: str
    model_response: str


@dataclass
class UGAnnotation:
    item_id: str
    raw_output: str
    ug_score: float    # 0-100 or math.nan
    finish_reason: str
    model_response: str


@dataclass
class ItemAnnotation:
    item_id: str
    demands: dict[str, DemandAnnotation]  # acronym -> DemandAnnotation
    ug: UGAnnotation

    def to_feature_vector(self) -> list[float]:
        result: list[float] = []
        for dim in DEMAND_DIMENSIONS:
            ann = self.demands.get(dim)
            result.append(ann.level if ann is not None else math.nan)
        result.append(self.ug.ug_score)
        return result


@dataclass
class DemandVector:
    item_ids: list[str]
    tensor: Any  # torch.Tensor at runtime; torch not imported here


@dataclass
class CacheEntry:
    key: str
    item_id: str
    demand: str        # rubric acronym or "UG"
    level: float       # demand level 0-5 or UG score 0-100; math.nan on parse failure
    finish_reason: str
    model_response: str
    rubric_hash: str
    model_id: str
    content_hash: str
    timestamp: str
    raw_output: str = ""  # UG only: the raw model token ("3", "open", etc.)
