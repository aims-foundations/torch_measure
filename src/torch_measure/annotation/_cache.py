from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._types import CacheEntry


def make_cache_key(content: str, acronym: str, model_id: str, rubric_hash: str) -> str:
    """sha256(content)[:16] : acronym : model_id : rubric_hash"""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{content_hash}:{acronym}:{model_id}:{rubric_hash}"


class AnnotationCache:
    """Append-only JSONL cache keyed by sha256(content)[:16]:acronym:model_id:rubric_hash."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._index: dict[str, CacheEntry] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # NaN is serialised as null (RFC-compliant); restore here.
                level = data.get("level")
                if level is None:
                    data["level"] = math.nan
                entry = CacheEntry(**data)
                self._index[entry.key] = entry

    def get(self, key: str) -> Optional[CacheEntry]:
        return self._index.get(key)

    def put(self, entry: CacheEntry) -> None:
        self._index[entry.key] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = dataclasses.asdict(entry)
        if isinstance(record["level"], float) and math.isnan(record["level"]):
            record["level"] = None
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
