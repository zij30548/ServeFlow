"""Bounded Beijing text embeddings for intent templates only; no startup sends.

The protected demo shares its existing durable send-permit journal with Messages.
Bodies and keys never enter diagnostics or the journal. There are no retries,
redirects, environment proxies, alternative models or automatic index migrations.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from urllib.parse import urlsplit

import httpx

from core import protected_demo_quota as quota

MODEL = "qwen3.7-text-embedding"
DIMENSIONS = 1024
BATCH_SIZE = 20
TIMEOUT = 6.0
REQUEST_LIMIT = 65536
RESPONSE_LIMIT = 1024 * 1024
MAX_TEXT_CHARS = 4000
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z")


class EmbeddingError(ValueError):
    """Only fixed codes are allowed across this boundary."""
    _CODES = frozenset({"embedding_beijing_endpoint_required", "embedding_api_key_required",
        "embedding_input_invalid", "embedding_request_limit", "embedding_response_invalid",
        "embedding_response_limit", "embedding_http_error", "embedding_timeout",
        "embedding_network_error", "embedding_unavailable"})

    def __init__(self, code="embedding_unavailable"):
        self.code = code if code in self._CODES else "embedding_unavailable"
        super().__init__(self.code)


def canonical_base_url(value):
    try:
        if (type(value) is not str or not 1 <= len(value) <= 512
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
                or any(c in value for c in "?#\\")):
            raise ValueError()
        u = urlsplit(value)
        if (u.scheme != "https" or u.hostname is None or u.netloc != u.hostname
                or u.port is not None or u.username is not None or u.password is not None
                or u.path != "/compatible-mode/v1"
                or not (u.hostname == "dashscope.aliyuncs.com" or _WORKSPACE.fullmatch(u.hostname))):
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise EmbeddingError("embedding_beijing_endpoint_required") from None


def validate_key(value):
    if (type(value) is not str or not 16 <= len(value) <= 512
            or any(ord(c) <= 32 or ord(c) >= 127 for c in value)):
        raise EmbeddingError("embedding_api_key_required")
    return value


def _payload(texts):
    if (type(texts) is not list or not 1 <= len(texts) <= BATCH_SIZE
            or any(type(t) is not str or not t.strip() or len(t) > MAX_TEXT_CHARS for t in texts)):
        raise EmbeddingError("embedding_input_invalid")
    raw = json.dumps({"model": MODEL, "input": texts, "dimensions": DIMENSIONS,
                      "encoding_format": "float"}, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > REQUEST_LIMIT:
        raise EmbeddingError("embedding_request_limit")
    return raw


def _pairs(values):
    result = {}
    for k, v in values:
        if k in result:
            raise ValueError()
        result[k] = v
    return result


def parse_vectors(raw, count):
    try:
        if type(count) is not int or not 1 <= count <= BATCH_SIZE:
            raise ValueError()
        body = json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if (type(body) is not dict or body.get("model", MODEL) != MODEL
                or type(body.get("data")) is not list or len(body["data"]) != count):
            raise ValueError()
        vectors = [None] * count
        for row in body["data"]:
            if type(row) is not dict:
                raise ValueError()
            i, vector = row.get("index"), row.get("embedding")
            if (type(i) is not int or not 0 <= i < count or vectors[i] is not None
                    or type(vector) is not list or len(vector) != DIMENSIONS
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise ValueError()
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError()
            vectors[i] = [x / norm for x in vector]
        return vectors
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        raise EmbeddingError("embedding_response_invalid") from None


def _delegate():
    return httpx.AsyncHTTPTransport(retries=0, trust_env=False,
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=1))


async def request_embeddings(texts, *, base_url, api_key):
    """One bounded physical request; reserve before send, never refund or retry."""
    base_url, api_key = canonical_base_url(base_url), validate_key(api_key)
    raw = _payload(texts)
    operation, store, response, client = None, None, None, None
    entered = False
    status = None
    try:
        if quota.enabled():
            # Explicit server opt-in; direct callers cannot enable another URL.
            import os
            if (os.getenv("INTENT_EMBEDDING_PROVIDER") != "qwen"
                    or base_url != canonical_base_url(os.getenv("QWEN_EMBEDDING_BASE_URL", ""))
                    or api_key != os.getenv("QWEN_EMBEDDING_API_KEY")):
                raise quota.ProtectedDemoQuotaError("demo_model_request_rejected")
            quota.raise_if_request_blocked()
        client = httpx.AsyncClient(transport=_delegate(), timeout=TIMEOUT,
                                  follow_redirects=False, trust_env=False)
        request = client.build_request("POST", base_url + "/embeddings", content=raw,
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
                     "Accept-Encoding": "identity"})
        current = asyncio.current_task()
        if current is None or current.cancelling():
            raise asyncio.CancelledError()
        if quota.enabled():
            store = quota.QuotaStore.from_env()
            operation = store.reserve(model=MODEL, phase=quota.EMBEDDING_PHASE,
                                      payload_sha256=hashlib.sha256(raw).hexdigest())
        if current.cancelling():
            raise asyncio.CancelledError()
        async with asyncio.timeout(TIMEOUT):
            entered = True
            response = await client.send(request, stream=True)
            status = response.status_code
            if (response.headers.get("content-encoding", "identity").lower() != "identity"
                    or "content-length" in response.headers and
                       (not response.headers["content-length"].isdigit()
                        or int(response.headers["content-length"]) > RESPONSE_LIMIT)):
                raise EmbeddingError("embedding_response_limit")
            if response.is_stream_consumed:
                body = response.content
            else:
                chunks, size = [], 0
                async for part in response.aiter_raw():
                    size += len(part)
                    if size > RESPONSE_LIMIT:
                        raise EmbeddingError("embedding_response_limit")
                    chunks.append(part)
                body = b"".join(chunks)
            if len(body) > RESPONSE_LIMIT:
                raise EmbeddingError("embedding_response_limit")
        if operation is not None:
            completed, operation = operation, None
            store.finish(completed, "received" if 200 <= status < 300 else "http_error", status)
        if not 200 <= status < 300:
            raise EmbeddingError("embedding_http_error")
        return parse_vectors(body, len(texts))
    except BaseException as error:
        cancelled = isinstance(error, asyncio.CancelledError)
        if operation is not None:
            outcome = ("not_sent" if not entered else
                       "cancelled" if isinstance(error, asyncio.CancelledError) else
                       "timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else
                       "network_error" if isinstance(error, httpx.RequestError) else
                       "response_limit" if isinstance(error, EmbeddingError) else "unknown")
            try:
                completed, operation = operation, None
                store.finish(completed, outcome, status, delegate_entered=entered)
            except Exception:
                error = quota.ProtectedDemoQuotaError()
        if isinstance(error, quota.ProtectedDemoQuotaError):
            quota.block_request(error)
        if cancelled:
            raise asyncio.CancelledError() from None
        if isinstance(error, quota.ProtectedDemoQuotaError):
            raise error from None
        code = ("embedding_timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else
                "embedding_network_error" if isinstance(error, httpx.RequestError) else
                error.code if isinstance(error, EmbeddingError) else "embedding_unavailable")
        raise EmbeddingError(code) from None
    finally:
        for resource in (response, client):
            if resource is not None:
                try:
                    await asyncio.wait_for(resource.aclose(), .25)
                except BaseException:
                    pass
