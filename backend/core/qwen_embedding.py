"""Bounded Beijing text embeddings for intent templates only; no startup sends.

The protected demo shares its existing durable send-permit journal with Messages.
Bodies and keys never enter diagnostics or the journal. There are no retries,
redirects, environment proxies, alternative models or automatic index migrations.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
import ssl
import struct
import time
from urllib.parse import urlsplit

import httpx
import httpcore

from core import protected_demo_quota as quota

MODEL = "qwen3.7-text-embedding"
DIMENSIONS = 1024
BATCH_SIZE = 20
TIMEOUT = 6.0
PREPARE_TIMEOUT = 30.0
ENCODING_FORMAT = "base64"
REQUEST_LIMIT = 65536
RESPONSE_LIMIT = 1024 * 1024
MAX_TEXT_CHARS = 4000
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z")
logger = logging.getLogger(__name__)


class EmbeddingError(ValueError):
    """Only fixed codes are allowed across this boundary."""
    _CODES = frozenset({"embedding_beijing_endpoint_required", "embedding_api_key_required",
        "embedding_input_invalid", "embedding_request_limit", "embedding_response_invalid",
        "embedding_response_limit", "embedding_http_error", "embedding_timeout",
        "embedding_network_error", "embedding_unavailable", "embedding_provider_required"})

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


def _payload(texts, *, encoding_format=ENCODING_FORMAT):
    if (type(texts) is not list or not 1 <= len(texts) <= BATCH_SIZE
            or any(type(t) is not str or not t.strip() or len(t) > MAX_TEXT_CHARS for t in texts)
            or type(encoding_format) is not str or encoding_format not in {"float", "base64"}):
        raise EmbeddingError("embedding_input_invalid")
    raw = json.dumps({"model": MODEL, "input": texts, "dimensions": DIMENSIONS,
                      "encoding_format": encoding_format}, ensure_ascii=True, allow_nan=False,
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
        response_format = None
        for row in body["data"]:
            if type(row) is not dict:
                raise ValueError()
            i, vector = row.get("index"), row.get("embedding")
            kind = type(vector)
            if kind not in (str, list) or response_format not in (None, kind):
                raise ValueError()
            response_format = kind
            if kind is str:
                # OpenAI-compatible base64 vectors are little-endian float32.
                # Old official gateways may still return float arrays even
                # when base64 was requested; both paths remain strict.
                if len(vector) != ((DIMENSIONS * 4 + 2) // 3) * 4:
                    raise ValueError()
                binary = base64.b64decode(vector, validate=True)
                if len(binary) != DIMENSIONS * 4 or base64.b64encode(binary).decode('ascii') != vector:
                    raise ValueError()
                vector = list(struct.unpack('<' + str(DIMENSIONS) + 'f', binary))
            if (type(i) is not int or not 0 <= i < count or vectors[i] is not None
                    or type(vector) is not list or len(vector) != DIMENSIONS
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise ValueError()
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError()
            vectors[i] = [x / norm for x in vector]
        return vectors
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError, struct.error):
        raise EmbeddingError("embedding_response_invalid") from None


def _delegate():
    return httpx.AsyncHTTPTransport(retries=0, trust_env=False,
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=1))


def _failure_codes(error):
    """Only fixed classifications, never exception messages/args or type names."""
    allowed = (
        (httpx.ConnectTimeout, "connect_timeout"), (httpx.ReadTimeout, "read_timeout"),
        (httpx.WriteTimeout, "write_timeout"), (httpx.PoolTimeout, "pool_timeout"),
        (httpx.TimeoutException, "timeout"),
        (httpx.RemoteProtocolError, "remote_protocol_error"),
        (httpx.LocalProtocolError, "local_protocol_error"),
        (httpx.ConnectError, "connect_error"), (httpx.ReadError, "read_error"),
        (httpx.WriteError, "write_error"), (httpx.CloseError, "close_error"),
        (httpx.RequestError, "request_error"),
        (httpcore.ConnectTimeout, "core_connect_timeout"),
        (httpcore.ReadTimeout, "core_read_timeout"),
        (httpcore.RemoteProtocolError, "core_remote_protocol_error"),
        (httpcore.LocalProtocolError, "core_local_protocol_error"),
        (httpcore.ConnectError, "core_connect_error"), (httpcore.ReadError, "core_read_error"),
        (httpcore.WriteError, "core_write_error"),
        (ssl.SSLCertVerificationError, "tls_certificate_error"),
        (ssl.SSLEOFError, "tls_unexpected_eof"), (ssl.SSLError, "tls_error"),
        (ConnectionResetError, "connection_reset"), (BrokenPipeError, "broken_pipe"),
        (ConnectionAbortedError, "connection_aborted"),
        (ConnectionRefusedError, "connection_refused"),
        (asyncio.CancelledError, "cancelled"), (TimeoutError, "timeout"),
    )
    codes, seen = [], set()
    current = error
    for _ in range(6):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        code = next((code for kind, code in allowed if isinstance(current, kind)), None)
        if code is not None and code not in codes:
            codes.append(code)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return codes


def _response_details(response, *, api_key=None):
    """Bounded protocol metadata, independent of whether the body finishes."""
    length = response.headers.get("content-length", "")
    expected = int(length) if length.isascii() and length.isdigit() and len(length) <= 7 else None
    if expected is not None and expected > RESPONSE_LIMIT:
        expected = None
    framing = ("content_length" if expected is not None else
               "chunked" if response.headers.get("transfer-encoding", "").lower() == "chunked" else "unknown")
    version = response.extensions.get("http_version")
    version = version.decode("ascii") if version in (b"HTTP/1.0", b"HTTP/1.1", b"HTTP/2", b"HTTP/3") else "unknown"
    request_id = response.headers.get("x-request-id", "") or response.headers.get("x-dashscope-request-id", "")
    if (not re.fullmatch(r"[A-Za-z0-9_.:-]{1,150}", request_id)
            or api_key and api_key in request_id):
        request_id = "unknown"
    return {"expected_response_bytes": expected, "response_framing": framing,
            "http_version": version, "request_id": request_id}


async def request_embeddings(texts, *, base_url, api_key, purpose="query", diagnostic=None):
    """One bounded physical request; reserve before send, never refund or retry."""
    base_url, api_key = canonical_base_url(base_url), validate_key(api_key)
    raw = _payload(texts)
    if purpose not in {"query", "prepare"} or diagnostic is not None and type(diagnostic) is not dict:
        raise EmbeddingError("embedding_input_invalid")
    timeout = PREPARE_TIMEOUT if purpose == "prepare" else TIMEOUT
    started = time.monotonic()
    elapsed = lambda: round(max(0., time.monotonic() - started) * 1000, 2)
    info = {"purpose": purpose, "phase": "admission", "deadline_s": timeout,
        "admission_ms": None,
        "headers_ms": None, "first_byte_ms": None, "body_complete_ms": None,
        "response_bytes": 0, "http_status": None, "error_code": None,
        "transport_failure_codes": [], "expected_response_bytes": None,
        "response_framing": "unknown", "http_version": "unknown", "request_id": "unknown"}
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
        client = httpx.AsyncClient(transport=_delegate(), timeout=timeout,
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
        info["admission_ms"] = elapsed()
        async with asyncio.timeout(timeout):
            info["phase"] = "waiting_headers"
            entered = True
            response = await client.send(request, stream=True)
            status = response.status_code
            info.update(headers_ms=elapsed(), http_status=status, phase="reading_body")
            info.update(_response_details(response, api_key=api_key))
            if (response.headers.get("content-encoding", "identity").lower() != "identity"
                    or "content-length" in response.headers and
                       (not response.headers["content-length"].isdigit()
                        or int(response.headers["content-length"]) > RESPONSE_LIMIT)):
                raise EmbeddingError("embedding_response_limit")
            if response.is_stream_consumed:
                body = response.content
                info.update(first_byte_ms=elapsed() if body else None, response_bytes=len(body))
            else:
                chunks, size = [], 0
                async for part in response.aiter_raw():
                    if part and info["first_byte_ms"] is None: info["first_byte_ms"] = elapsed()
                    size += len(part)
                    info["response_bytes"] = size
                    if size > RESPONSE_LIMIT:
                        raise EmbeddingError("embedding_response_limit")
                    chunks.append(part)
                body = b"".join(chunks)
            if len(body) > RESPONSE_LIMIT:
                raise EmbeddingError("embedding_response_limit")
            info["body_complete_ms"] = elapsed()
            info["phase"] = "accounting"
        if operation is not None:
            completed, operation = operation, None
            store.finish(completed, "received" if 200 <= status < 300 else "http_error", status)
        if not 200 <= status < 300:
            raise EmbeddingError("embedding_http_error")
        info["phase"] = "validating_vectors"
        vectors = parse_vectors(body, len(texts))
        info["phase"] = "complete"
        return vectors
    except BaseException as error:
        info["transport_failure_codes"] = _failure_codes(error)
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
            info["error_code"] = error.code
        if cancelled:
            if info["error_code"] is None: info["error_code"] = "embedding_cancelled"
            raise asyncio.CancelledError() from None
        if isinstance(error, quota.ProtectedDemoQuotaError):
            raise error from None
        code = ("embedding_timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else
                "embedding_network_error" if isinstance(error, httpx.RequestError) else
                error.code if isinstance(error, EmbeddingError) else "embedding_unavailable")
        info["error_code"] = code
        raise EmbeddingError(code) from None
    finally:
        for resource in (response, client):
            if resource is not None:
                try:
                    await asyncio.wait_for(resource.aclose(), .25)
                except BaseException:
                    pass
        info["elapsed_ms"] = elapsed()
        info["delegate_entered"] = entered
        if diagnostic is not None:
            diagnostic.clear()
            diagnostic.update(info)
        if info["error_code"]:
            logger.warning("embedding_failure purpose=%s phase=%s code=%s elapsed_ms=%s headers_ms=%s first_byte_ms=%s response_bytes=%s transport_codes=%s framing=%s expected_bytes=%s",
                info["purpose"], info["phase"], info["error_code"], info["elapsed_ms"],
                info["headers_ms"], info["first_byte_ms"], info["response_bytes"],
                info["transport_failure_codes"], info["response_framing"], info["expected_response_bytes"])
