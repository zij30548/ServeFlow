"""Bounded structural retrieval checks, not a semantic relevance judge.

The model can select no supporting source, but that decision remains fallible.
No lexical threshold is used: literal overlap alone both misses paraphrases and
mistakes shared vocabulary for useful evidence.
"""
from __future__ import annotations

import json
import re


RETRIEVAL_POLICY_VERSION = "original-query-full-evidence-v1"
MAX_QUERY_CHARS = 2000
MAX_RESULT_ROWS = 20
MAX_CONTENT_CHARS = 2000
MAX_BATCH_BYTES = 65536
_SOURCE_FIELDS = frozenset({"title", "content", "score", "document_id", "version", "chunk"})
# Sentence boundaries only. Commas, but/except/or and their conditions stay in
# the same unit; the model may select a unit, never paraphrase its facts.
_SENTENCES = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def query_units(query: str) -> tuple[str, ...]:
    """Complete extractive search units, always in source order."""
    return tuple(dict.fromkeys(m.group().strip() for m in _SENTENCES.finditer(query) if m.group().strip()))


def checked_rewrites(query: str, value, maximum: int) -> list[str]:
    if (type(value) is not list or len(value) > maximum
            or any(type(v) is not str or not v.strip() or len(v) > MAX_QUERY_CHARS for v in value)):
        raise ValueError("invalid_extractive_queries")
    allowed = {query, *query_units(query)}
    # Reject the whole response rather than silently removing an invented
    # entity, temporal qualifier or causal claim and calling it successful.
    if any(v not in allowed for v in value):
        raise ValueError("non_extractive_query")
    return list(dict.fromkeys([query, *value]))


def checked_sources(items) -> list[dict]:
    if type(items) is not list or len(items) > MAX_RESULT_ROWS:
        raise ValueError("retrieval_source_count")
    for item in items:
        if (type(item) is not dict or set(item) - _SOURCE_FIELDS or type(item.get("title")) is not str
                or not 1 <= len(item["title"]) <= 160
                or type(item.get("content")) is not str
                or not 1 <= len(item["content"]) <= MAX_CONTENT_CHARS):
            raise ValueError("retrieval_source_shape")
    encoded = json.dumps(items, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BATCH_BYTES:
        raise ValueError("retrieval_source_batch_limit")
    return items


def checked_permutation(value, size: int) -> list[int]:
    if (type(value) is not list or len(value) != size or any(type(v) is not int for v in value)
            or set(value) != set(range(size))):
        raise ValueError("rerank_complete_unique_indices_required")
    return value


def json_array(text: str):
    """Accept a whole JSON response (or one fence), not the first embedded list."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4].strip()
    return json.loads(text)
