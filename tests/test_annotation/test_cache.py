# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Unit tests for _cache.py — no API calls required.

Verifies: cache key format, NaN serialization, put/get round-trip,
rubric-hash-based cache invalidation, JSON RFC compliance.
"""

import json
import math
from pathlib import Path

import pytest

from torch_measure.annotation._cache import AnnotationCache, make_cache_key
from torch_measure.annotation._types import CacheEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    content="test item content",
    acronym="AS",
    model_id="model-001",
    rubric_hash="abc12345",
    level=3.0,
    raw_output="",
) -> CacheEntry:
    key = make_cache_key(content, acronym, model_id, rubric_hash)
    return CacheEntry(
        key=key,
        item_id="item_001",
        demand=acronym,
        level=level,
        finish_reason="stop",
        model_response="Thus, the level is 3",
        rubric_hash=rubric_hash,
        model_id=model_id,
        content_hash=key.split(":")[0],
        timestamp="2026-01-01T00:00:00+00:00",
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

class TestMakeCacheKey:

    def test_content_hash_is_16_hex_chars(self):
        key = make_cache_key("content", "AS", "model-001", "rubrichash")
        content_hash = key.split(":")[0]
        assert len(content_hash) == 16
        assert all(c in "0123456789abcdef" for c in content_hash)

    def test_key_ends_with_rubric_hash(self):
        key = make_cache_key("content", "AS", "model-001", "rubrichash")
        assert key.endswith(":rubrichash")

    def test_key_contains_acronym(self):
        key = make_cache_key("content", "AS", "model-001", "rubrichash")
        assert ":AS:" in key

    def test_different_content_different_key(self):
        k1 = make_cache_key("content A", "AS", "model", "hash")
        k2 = make_cache_key("content B", "AS", "model", "hash")
        assert k1 != k2

    def test_different_acronym_different_key(self):
        k1 = make_cache_key("content", "AS", "model", "hash")
        k2 = make_cache_key("content", "CL", "model", "hash")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = make_cache_key("content", "AS", "model-v1", "hash")
        k2 = make_cache_key("content", "AS", "model-v2", "hash")
        assert k1 != k2

    def test_different_rubric_hash_different_key(self):
        """Editing a rubric file changes its hash, which changes the cache key.
        This is the rubric-change cache-invalidation mechanism.
        """
        k1 = make_cache_key("content", "AS", "model", "hash_before_edit")
        k2 = make_cache_key("content", "AS", "model", "hash_after_edit")
        assert k1 != k2

    def test_identical_inputs_produce_identical_key(self):
        k1 = make_cache_key("content", "AS", "model", "hash")
        k2 = make_cache_key("content", "AS", "model", "hash")
        assert k1 == k2

    def test_ug_key_uses_ug_acronym(self):
        key = make_cache_key("content", "UG", "model", "ug_hash")
        assert ":UG:" in key


# ---------------------------------------------------------------------------
# AnnotationCache — put / get
# ---------------------------------------------------------------------------

class TestAnnotationCachePutGet:

    def test_put_and_get_roundtrip(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e = _entry(level=3.0)
        cache.put(e)
        retrieved = cache.get(e.key)
        assert retrieved is not None
        assert retrieved.level == 3.0
        assert retrieved.demand == "AS"
        assert retrieved.item_id == "item_001"

    def test_get_missing_key_returns_none(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        assert cache.get("nonexistent::key::here") is None

    def test_multiple_entries_retrievable(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e1 = _entry(content="item A", acronym="AS", level=1.0)
        e2 = _entry(content="item B", acronym="CL", level=4.0)
        cache.put(e1)
        cache.put(e2)
        assert cache.get(e1.key).level == 1.0
        assert cache.get(e2.key).level == 4.0

    def test_most_recent_put_wins_in_memory(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e1 = _entry(level=1.0)
        e2 = _entry(level=5.0)  # same key (same inputs)
        cache.put(e1)
        cache.put(e2)
        assert cache.get(e1.key).level == 5.0

    def test_raw_output_stored_and_retrieved(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e = _entry(acronym="UG", raw_output="4")
        cache.put(e)
        assert cache.get(e.key).raw_output == "4"

    def test_finish_reason_preserved(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e = _entry()
        cache.put(e)
        assert cache.get(e.key).finish_reason == "stop"

    def test_model_response_preserved(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e = _entry()
        cache.put(e)
        assert "level is 3" in cache.get(e.key).model_response


# ---------------------------------------------------------------------------
# NaN serialization (RFC compliance)
# ---------------------------------------------------------------------------

class TestNaNSerialization:

    def test_nan_written_as_null_not_bare_nan(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        cache.put(_entry(level=math.nan))
        raw = cache_path.read_text()
        # Must NOT contain bare NaN (non-RFC JSON)
        assert "NaN" not in raw
        # Must contain null
        assert "null" in raw

    def test_nan_survives_put_get_roundtrip_in_memory(self, tmp_path):
        cache = AnnotationCache(tmp_path / "cache.jsonl")
        e = _entry(level=math.nan)
        cache.put(e)
        result = cache.get(e.key)
        assert math.isnan(result.level)

    def test_nan_survives_disk_reload(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache1 = AnnotationCache(cache_path)
        cache1.put(_entry(level=math.nan))

        cache2 = AnnotationCache(cache_path)
        result = cache2.get(_entry(level=math.nan).key)
        assert result is not None
        assert math.isnan(result.level)

    def test_all_lines_are_valid_rfc_json(self, tmp_path):
        """Every JSONL line must be parseable by strict json.loads (no allow_nan)."""
        cache_path = tmp_path / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        for level in [0.0, 1.0, 3.0, 5.0, math.nan]:
            cache.put(_entry(level=level))
        for line in cache_path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # raises if NaN/Infinity present

    def test_valid_levels_not_converted_to_null(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        cache.put(_entry(level=3.0))
        data = json.loads(cache_path.read_text().strip())
        assert data["level"] == 3.0
        assert data["level"] is not None


# ---------------------------------------------------------------------------
# Persistence and reload
# ---------------------------------------------------------------------------

class TestAnnotationCachePersistence:

    def test_entry_survives_cache_reload(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        e = _entry(level=2.0)

        cache1 = AnnotationCache(cache_path)
        cache1.put(e)

        cache2 = AnnotationCache(cache_path)
        result = cache2.get(e.key)
        assert result is not None
        assert result.level == 2.0

    def test_last_write_wins_on_reload(self, tmp_path):
        """Append-only: same key written twice; on reload the last value is active."""
        cache_path = tmp_path / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        e1 = _entry(level=1.0)
        e2 = _entry(level=4.0)
        cache.put(e1)
        cache.put(e2)

        cache2 = AnnotationCache(cache_path)
        assert cache2.get(e1.key).level == 4.0

    def test_file_not_created_until_put(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        _ = AnnotationCache(cache_path)
        assert not cache_path.exists()

    def test_parent_directories_created_on_put(self, tmp_path):
        cache_path = tmp_path / "a" / "b" / "c" / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        cache.put(_entry())
        assert cache_path.exists()

    def test_empty_file_loads_without_error(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache_path.write_text("")
        cache = AnnotationCache(cache_path)
        assert cache.get("any_key") is None

    def test_blank_lines_in_file_are_skipped(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        e = _entry(level=2.0)
        cache = AnnotationCache(cache_path)
        cache.put(e)
        original = cache_path.read_text()
        cache_path.write_text("\n\n" + original + "\n\n")
        cache2 = AnnotationCache(cache_path)
        assert cache2.get(e.key) is not None

    def test_raw_output_default_empty_for_demand_entries(self, tmp_path):
        """Demand annotation entries have raw_output='' (not UG)."""
        cache_path = tmp_path / "cache.jsonl"
        cache = AnnotationCache(cache_path)
        cache.put(_entry(raw_output=""))
        cache2 = AnnotationCache(cache_path)
        assert cache2.get(_entry().key).raw_output == ""
