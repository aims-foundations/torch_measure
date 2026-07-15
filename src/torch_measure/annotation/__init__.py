"""ADeLe demand annotation pipeline (Gemini re-implementation).

Reproduces the annotation methodology from:
  Zhou et al. (2026) "General scales unlock AI evaluation with explanatory
  and predictive power." Nature.

``DemandAnnotator`` accepts any client that implements
``generate(prompt: str) -> tuple[str, str]`` (response text, finish reason).
``GeminiClient`` is the bundled implementation, but any LLM provider
(OpenAI, Anthropic, Azure, etc.) can be used by wrapping it in a class with
that single method.

Public API
----------
DemandAnnotator   — main entry point: annotates one item or a full dataset
GeminiClient      — Gemini API wrapper (caller supplies pinned model string)
RubricsCatalog    — loads the 19 bundled rubric files
AnnotationCache   — append-only JSONL result cache

Data types
----------
AnnotationJob     — input: item_id, content, reference_answer
DemandAnnotation  — one (item, rubric) result with CoT response
UGAnnotation      — UG classification result
ItemAnnotation    — all 19 annotations for one item (.to_feature_vector())
DemandVector      — full-dataset tensor (n_items × 19) for DemandAssessor
CacheEntry        — one persisted cache record

Constants
---------
DIMENSION_ORDER   — canonical ordering of all 19 dimensions
DEMAND_DIMENSIONS — the first 18 (excludes UG)
"""
from ._annotator import DemandAnnotator
from ._cache import AnnotationCache
from ._claude_client import ClaudeClient
from ._client import GeminiClient
from ._openai_client import OpenAIClient
from ._rubrics import RubricsCatalog
from ._types import (
    DEMAND_DIMENSIONS,
    DIMENSION_ORDER,
    N_DIMENSIONS,
    AnnotationJob,
    CacheEntry,
    DemandAnnotation,
    DemandVector,
    ItemAnnotation,
    Rubric,
    UGAnnotation,
)
from ._ug import UGAnnotator

__all__ = [
    "DemandAnnotator",
    "ClaudeClient",
    "GeminiClient",
    "OpenAIClient",
    "RubricsCatalog",
    "AnnotationCache",
    "UGAnnotator",
    "AnnotationJob",
    "DemandAnnotation",
    "UGAnnotation",
    "ItemAnnotation",
    "DemandVector",
    "CacheEntry",
    "Rubric",
    "DIMENSION_ORDER",
    "DEMAND_DIMENSIONS",
    "N_DIMENSIONS",
]
