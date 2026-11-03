"""One-send transport for an explicitly admitted intent trial.

This module does not read credentials, create a ledger, or grant live-call
permission. The caller supplies the frozen payload hash and a synchronous,
durable reservation callback. Error messages contain fixed codes only.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx


_WORKSPACE_HOST = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z"
)
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,150}\Z")
_REQUEST_LIMIT = 65_536
_CLEANUP_TIMEOUT_S = 0.1


class TrialTransportError(RuntimeError):
    """A safe fixed-code failure, without upstream exception text or bodies."""

    def __init__(self, code: str, http_status: int | None = None):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


def validated_endpoint(base_url: str) -> str:
    """Return the canonical supported Beijing base, rejecting all explicit ports."""
    if (type(base_url) is not str or len(base_url) > 512
            or any(ord(char) <= 32 or ord(char) >= 127 for char in base_url)
            or any(char in base_url for char in "?#\\")):
        raise TrialTransportError("invalid_endpoint")
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        if (parsed.scheme != "https" or host is None or parsed.netloc != host
                or parsed.port is not None or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"/apps/anthropic", "/apps/anthropic/"}
                or not (host == "dashscope.aliyuncs.com" or _WORKSPACE_HOST.fullmatch(host))):
            raise ValueError
    except (TypeError, ValueError, UnicodeError):
        raise TrialTransportError("invalid_endpoint") from None
    return f"https://{host}/apps/anthropic"


def messages_url(base_url: str) -> str:
    return validated_endpoint(base_url) + "/v1/messages"


def _strict_json(value: Any) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _strict_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError
            _strict_json(item)
        return
    raise ValueError


def canonical_payload_hash(payload: dict) -> str:
    """Hash strict JSON using sorted keys and compact UTF-8 serialization."""
    try:
        if type(payload) is not dict:
            raise ValueError
        _strict_json(payload)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise TrialTransportError("invalid_payload") from None
    return hashlib.sha256(raw).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError


class GuardedIntentTransport(httpx.AsyncBaseTransport):
    """A single physical send, with admission before any network delegate call.

    ``sent`` means the delegate was entered, not that upstream received or billed
    the request. The caller must pause on any ``failure_code`` and never retry
    an unknown send. Each cleanup step is best effort and bounded to 100 ms;
    cleanup never replaces an already observed primary failure.
    """

    def _timeout_ceiling(self) -> float:
        """Protected opt-in boundary; historical intent callers stay at 8s."""
        return 8.0

    def _prepare_network_request(self, request: httpx.Request) -> None:
        """Trusted subclasses may tighten request-local transport settings."""
        return None

    def __init__(self, base_url: str, expected_payload_sha256: str,
                 before_send: Callable[[], Any], transport: httpx.AsyncBaseTransport | None = None,
                 timeout_s: float = 8, response_limit: int = 262_144):
        self.base_url = validated_endpoint(base_url)
        self._url = messages_url(self.base_url)
        if type(expected_payload_sha256) is not str or not _SHA256.fullmatch(expected_payload_sha256):
            raise TrialTransportError("invalid_payload")
        if (not callable(before_send) or inspect.iscoroutinefunction(before_send)
                or type(timeout_s) not in {int, float} or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= self._timeout_ceiling() or type(response_limit) is not int
                or not 1 <= response_limit <= 262_144):
            raise TrialTransportError("invalid_request")
        self._expected_hash = expected_payload_sha256
        self._before_send = before_send
        self._transport = transport if transport is not None else httpx.AsyncHTTPTransport(
            retries=0, trust_env=False,
        )
        self._timeout_s = float(timeout_s)
        self._response_limit = response_limit
        self._used = False
        self._closed = False
        self.sent = False
        self.failure_code: str | None = None
        self.http_status: int | None = None
        self.request_id: str | None = None
        self.request_bytes_sha256: str | None = None

    def _failure(self, code: str, status: int | None = None) -> TrialTransportError:
        self.failure_code = code
        if status is not None:
            self.http_status = status
        return TrialTransportError(code, self.http_status)

    def _validate_request(self, request: httpx.Request) -> None:
        if request.method != "POST" or str(request.url) != self._url:
            raise self._failure("invalid_request")
        try:
            body = request.content
            if len(body) > _REQUEST_LIMIT:
                raise ValueError
            payload = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_pairs,
                                 parse_constant=_reject_constant)
            digest = canonical_payload_hash(payload)
        except (ValueError, TypeError, UnicodeError, RecursionError, httpx.RequestNotRead):
            raise self._failure("invalid_payload") from None
        except TrialTransportError:
            raise self._failure("invalid_payload") from None
        if digest != self._expected_hash:
            raise self._failure("payload_mismatch")
        self.request_bytes_sha256 = hashlib.sha256(body).hexdigest()
        # Set this before admission; reject compressed responses rather than
        # permitting unbounded decompression in HTTPX/SDK response processing.
        request.headers["accept-encoding"] = "identity"

    @staticmethod
    def _status_code(status: int) -> str:
        if 300 <= status <= 399:
            return "redirect"
        if status in {401, 403}:
            return "authentication_error"
        if status == 402:
            return "payment_required"
        if status == 429:
            return "rate_limited"
        if status >= 500:
            return "upstream_error"
        return "http_error"

    async def _response_bytes(self, response: httpx.Response) -> bytes:
        # Mock transports may return a response whose body is already buffered;
        # the real HTTP transport returns an unconsumed async byte stream.
        if response.is_stream_consumed:
            raw = response.content
            if len(raw) > self._response_limit:
                raise self._failure("response_limit")
            return raw
        parts = []
        size = 0
        async for chunk in response.aiter_raw():
            size += len(chunk)
            if size > self._response_limit:
                raise self._failure("response_limit")
            parts.append(chunk)
        return b"".join(parts)

    async def _cleanup(self, response: httpx.Response | None) -> bool:
        failed = False
        if response is not None:
            try:
                await asyncio.wait_for(response.aclose(), _CLEANUP_TIMEOUT_S)
            except (Exception, asyncio.CancelledError):
                failed = True
        try:
            await asyncio.wait_for(self.aclose(), _CLEANUP_TIMEOUT_S)
        except (Exception, asyncio.CancelledError):
            failed = True
        return failed

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._used:
            raise self._failure("already_sent")
        if self._closed:
            raise self._failure("transport_closed")
        self._used = True
        response = None
        primary_failure = False
        try:
            self._validate_request(request)
            try:
                result = self._before_send()
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise ValueError
            except Exception:
                raise self._failure("before_send_failed") from None
            self._prepare_network_request(request)
            async with asyncio.timeout(self._timeout_s):
                self.sent = True
                response = await self._transport.handle_async_request(request)
                self.http_status = response.status_code
                request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
                if isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id):
                    self.request_id = request_id
                if not 200 <= response.status_code <= 299:
                    raise self._failure(self._status_code(response.status_code), response.status_code)
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                if encoding not in {"", "identity"}:
                    raise self._failure("response_encoding")
                raw = await self._response_bytes(response)
                headers = {}
                content_type = response.headers.get("content-type")
                if (isinstance(content_type, str) and len(content_type) <= 128
                        and all(32 <= ord(char) < 127 for char in content_type)):
                    headers["content-type"] = content_type
                if self.request_id is not None:
                    headers["request-id"] = self.request_id
                return httpx.Response(response.status_code, content=raw, headers=headers, request=request)
        except TrialTransportError:
            primary_failure = True
            raise
        except (TimeoutError, httpx.TimeoutException):
            primary_failure = True
            raise self._failure("timeout") from None
        except asyncio.CancelledError:
            primary_failure = True
            raise self._failure("cancelled") from None
        except Exception:
            primary_failure = True
            raise self._failure("network_error") from None
        except BaseException:
            primary_failure = True
            raise
        finally:
            cleanup_failed = await self._cleanup(response)
            if cleanup_failed and not primary_failure:
                raise self._failure("network_error") from None

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                await self._transport.aclose()
            except (Exception, asyncio.CancelledError):
                if self.failure_code is None:
                    self.failure_code = "network_error"
                raise TrialTransportError("network_error", self.http_status) from None
