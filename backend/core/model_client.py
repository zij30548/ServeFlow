"""Original SDK by default; bounded, counted sends in protected-demo only."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
from urllib.parse import urlsplit

import anthropic
import httpx

from core.protected_demo_quota import (
    MODEL, ProtectedDemoQuotaError, QuotaStore, block_request, enabled, raise_if_request_blocked,
)


REQUEST_LIMIT = 65536
RESPONSE_LIMIT = 262144
TIMEOUT = 30.0
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com\Z")


def canonical_endpoint(value):
    try:
        if (type(value) is not str or not 1 <= len(value) <= 512
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value) or any(c in value for c in "?#\\")):
            raise ValueError()
        u = urlsplit(value)
        if (u.scheme != "https" or u.hostname is None or u.netloc != u.hostname or u.port is not None
                or u.username is not None or u.password is not None or u.path not in {"/apps/anthropic", "/apps/anthropic/"}
                or not (u.hostname == "dashscope.aliyuncs.com" or _WORKSPACE.fullmatch(u.hostname))):
            raise ValueError()
        return f"https://{u.hostname}/apps/anthropic"
    except Exception:
        raise ProtectedDemoQuotaError("demo_model_request_rejected") from None


def _pairs(values):
    result = {}
    for k, v in values:
        if k in result:
            raise ValueError()
        result[k] = v
    return result


def _payload(raw):
    try:
        p = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if (type(p) is not dict or p.get("model") != MODEL or p.get("stream", False) is not False
                or type(p.get("max_tokens")) is not int or not 1 <= p["max_tokens"] <= 4096
                or "temperature" in p and (type(p["temperature"]) not in {float, int}
                    or not math.isfinite(p["temperature"]) or not 0 <= p["temperature"] <= 1)):
            raise ValueError()
        if "thinking" in p and p["thinking"] != {"type": "disabled"}:
            raise ValueError()
        def finite(value, depth=0):
            if depth > 30:
                raise ValueError()
            if type(value) is float and not math.isfinite(value):
                raise ValueError()
            if type(value) is dict:
                for v in value.values(): finite(v, depth + 1)
            elif type(value) is list:
                for v in value: finite(v, depth + 1)
        finite(p)
        return p
    except Exception:
        raise ProtectedDemoQuotaError("demo_model_request_rejected") from None


def _delegate():
    """Fixed internal factory; tests replace it, requests cannot select it."""
    return httpx.AsyncHTTPTransport(retries=0, trust_env=False,
        limits=httpx.Limits(max_connections=6, max_keepalive_connections=3))


class ProtectedDemoTransport(httpx.AsyncBaseTransport):
    def __init__(self, base_url, phase):
        self.base_url = canonical_endpoint(base_url)
        if type(phase) is not str or phase not in {"intent", "agent", "retrieval", "memory", "judge", "provider"}:
            raise ProtectedDemoQuotaError("demo_model_request_rejected")
        self.phase, self._transport, self.closed = phase, _delegate(), False

    async def handle_async_request(self, request):
        operation, response, store = None, None, None
        entered = False
        deadline = asyncio.get_running_loop().time() + TIMEOUT
        try:
            raise_if_request_blocked()
            if self.closed or request.method != "POST" or str(request.url) != self.base_url + "/v1/messages":
                raise ProtectedDemoQuotaError("demo_model_request_rejected")
            if request.headers.get("host") != urlsplit(self.base_url).netloc:
                raise ProtectedDemoQuotaError("demo_model_request_rejected")
            chunks, size = [], 0
            async with asyncio.timeout_at(deadline):
                async for part in request.stream:
                    chunks.append(part)
                    size += len(part)
                    if size > REQUEST_LIMIT:
                        raise ProtectedDemoQuotaError("demo_model_request_rejected")
            raw = b"".join(chunks)
            _payload(raw)
            current = asyncio.current_task()
            if current is None or current.cancelling():
                raise asyncio.CancelledError()
            store = QuotaStore.from_env()
            operation = store.reserve(model=MODEL, phase=self.phase, payload_sha256=hashlib.sha256(raw).hexdigest())
            # Reservation is durable before entering the delegate. A crash in
            # between remains unresolved and charged; never presume no billing.
            bounded = httpx.Request(request.method, request.url, headers=request.headers, content=raw,
                extensions={**request.extensions, "timeout": {k: TIMEOUT for k in ("connect", "read", "write", "pool")}})
            if current.cancelling() or asyncio.get_running_loop().time() >= deadline:
                raise asyncio.CancelledError() if current.cancelling() else TimeoutError()
            async with asyncio.timeout_at(deadline):
                entered = True
                response = await self._transport.handle_async_request(bounded)
                if (response.headers.get("content-encoding", "identity").lower() != "identity"
                        or "content-length" in response.headers and
                           (not response.headers["content-length"].isdigit()
                            or int(response.headers["content-length"]) > RESPONSE_LIMIT)):
                    raise ProtectedDemoQuotaError("demo_model_request_rejected")
                if response.is_stream_consumed:
                    body = response.content
                else:
                    pieces, total = [], 0
                    async for part in response.aiter_raw():
                        pieces.append(part)
                        total += len(part)
                        if total > RESPONSE_LIMIT:
                            raise ProtectedDemoQuotaError("demo_model_request_rejected")
                    body = b"".join(pieces)
                if len(body) > RESPONSE_LIMIT:
                    raise ProtectedDemoQuotaError("demo_model_request_rejected")
            status = response.status_code
            completed = operation
            operation = None
            # A finish whose persistence is uncertain is not retried by the
            # error path; the original permit remains occupied.
            store.finish(completed, "received" if 200 <= status < 300 else "http_error", status)
            if not 200 <= status < 300:
                # No free-form error body is allowed to trigger further paid
                # fallback attempts within the same user request.
                raise ProtectedDemoQuotaError("demo_model_provider_unavailable")
            return httpx.Response(status, headers=response.headers, content=body, request=request)
        except BaseException as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            if operation is not None and store is not None:
                outcome = ("not_sent" if not entered else
                           "cancelled" if isinstance(error, asyncio.CancelledError) else
                           "timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else
                           "network_error" if isinstance(error, httpx.RequestError) else
                           "response_limit" if isinstance(error, ProtectedDemoQuotaError) else "unknown")
                try:
                    completed, operation = operation, None
                    store.finish(completed, outcome, delegate_entered=entered)
                except Exception:
                    error = ProtectedDemoQuotaError()
            if cancelled:
                block_request(error if isinstance(error, ProtectedDemoQuotaError) else
                              ProtectedDemoQuotaError("demo_model_provider_unavailable"))
                raise asyncio.CancelledError() from None
            safe = error if isinstance(error, ProtectedDemoQuotaError) else ProtectedDemoQuotaError("demo_model_provider_unavailable")
            block_request(safe)
            raise safe from None
        finally:
            if response is not None:
                try:
                    await asyncio.wait_for(response.aclose(), .25)
                except BaseException:
                    pass

    async def aclose(self):
        if not self.closed:
            self.closed = True
            await asyncio.wait_for(self._transport.aclose(), .25)


def AsyncAnthropic(*args, **kwargs):
    """Keep ordinary callers on the unchanged SDK and opt in server-side only."""
    if not enabled():
        return anthropic.AsyncAnthropic(*args, **kwargs)
    try:
        key = kwargs.get("api_key")
        if (args or set(kwargs) - {"api_key", "base_url", "max_retries", "timeout"}
                or kwargs.get("max_retries", 0) != 0 or kwargs.get("timeout", TIMEOUT) != TIMEOUT
                or type(key) is not str or not 1 <= len(key) <= 4096
                or any(ord(c) <= 32 or ord(c) >= 127 for c in key)
                or os.getenv("ANTHROPIC_MODEL") != MODEL):
            raise ProtectedDemoQuotaError("demo_model_request_rejected")
        base_url = canonical_endpoint(str(kwargs.get("base_url", "")))
        if base_url != canonical_endpoint(os.getenv("ANTHROPIC_BASE_URL", "")):
            raise ProtectedDemoQuotaError("demo_model_request_rejected")
        # Constructing a client verifies but does not consume quota. Zero
        # remaining is valid for read-only startup/monitoring.
        QuotaStore.from_env().verify()
        module = inspect.currentframe().f_back.f_globals.get("__name__", "")
        phase = {"core.intent_recognizer": "intent", "agents.agent_orchestrator": "agent",
                 "tooling.tool_manager": "retrieval", "memory.conversation_memory": "memory",
                 "evaluation.evaluator": "judge"}.get(module, "provider")
        transport = ProtectedDemoTransport(base_url, phase)
        http = httpx.AsyncClient(transport=transport, timeout=TIMEOUT, follow_redirects=False,
                                trust_env=False, headers={"Accept-Encoding": "identity"})
        return anthropic.AsyncAnthropic(**{**kwargs, "base_url": base_url, "max_retries": 0,
            "timeout": TIMEOUT, "http_client": http})
    except ProtectedDemoQuotaError as error:
        block_request(error)
        raise
