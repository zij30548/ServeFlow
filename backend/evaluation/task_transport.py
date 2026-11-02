"""Dynamic native-tool request admission for the fixed task trial only.

No clients are created at import time. Follow-up payloads cannot be frozen before
model/tool outputs exist; their exact canonical and wire hashes are persisted at
the actual send boundary. The existing one-send HTTP guard remains unchanged.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
from types import SimpleNamespace

import httpx

from evaluation import intent_trial as base
from evaluation.intent_transport import (
    GuardedIntentTransport, TrialTransportError, canonical_payload_hash,
    validated_endpoint, messages_url, _unique_pairs, _reject_constant,
)


@dataclass(frozen=True)
class Frame:
    case_id: str
    turn_id: str
    phase: str | None = None
    agent: str | None = None


CURRENT: ContextVar[Frame | None] = ContextVar("serveflow_task_trial", default=None)


@contextmanager
def context(frame=None, *, phase=None, agent=None):
    value = frame or CURRENT.get()
    if value is None:
        raise TrialTransportError("protocol")
    if phase is not None:
        value = replace(value, phase=phase)
    if agent is not None:
        value = replace(value, agent=agent)
    token = CURRENT.set(value)
    try:
        yield value
    finally:
        CURRENT.reset(token)


class BoundClient:
    """SDK facade that assigns immutable, task-local call provenance."""

    def __init__(self, sdk, phase=None, agent=None):
        self.base_url = sdk.base_url
        self.messages = SimpleNamespace(create=self.create)
        self._sdk, self._phase, self._agent = sdk, phase, agent

    async def create(self, **kwargs):
        frame = CURRENT.get()
        if frame is None:
            raise TrialTransportError("protocol")
        if self._phase is not None:
            # Explicit non-agent stages never inherit a stale actor tag.
            frame = replace(frame, phase=self._phase, agent=self._agent)
        with context(frame):
            return await self._sdk.messages.create(**kwargs)


PHASE_PARAMETERS = {
    "intent": (256, 0.1), "composer": (1000, 0.1),
    "rag_rewrite": (512, 0.3), "rag_rerank": (2048, 0.0),
    "judge": (4096, 0.0),
}
AGENT_PARAMETERS = {"general": (900, 0.3), "technical": (1200, 0.1), "billing": (1100, 0.0)}
HTTP_LIMITS = {"request_bytes": 65536, "response_bytes": 262144,
               "request_deadline_s": 8, "cleanup_step_s": 0.1, "concurrency": 3,
               "sdk_retries": 0, "http_retries": 0, "redirects": False}


def _response(raw):
    """Retain bounded response data, never HTTP headers or hidden reasoning."""
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    canonical_payload_hash(value)
    if not isinstance(value.get("content"), list):
        raise ValueError
    blocks = []
    for item in value["content"]:
        if not isinstance(item, dict):
            raise ValueError
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            blocks.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "tool_use" and set(item) >= {"id", "name", "input"}:
            blocks.append({key: item[key] for key in ("type", "id", "name", "input")})
        else:
            raise ValueError
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    tokens = {key: usage.get(key) if type(usage.get(key)) is int and usage[key] >= 0 else None
              for key in ("input_tokens", "output_tokens")}
    return {"content": blocks, "upstream_model": value.get("model") if isinstance(value.get("model"), str) and len(value["model"]) <= 150 else None,
            "stop_reason": value.get("stop_reason") if isinstance(value.get("stop_reason"), str) and len(value["stop_reason"]) <= 80 else None,
            "usage": tokens}


class TaskHTTPTransport(httpx.AsyncBaseTransport):
    """All stage facades share this physical-send gate and durable ledger."""

    def __init__(self, base_url, root, budget, lease, verify_admission, *, delegate_factory=None):
        self.base_url = validated_endpoint(base_url)
        self.root, self.budget, self.lease = root, budget, lease
        self.verify_admission = verify_admission
        self.factory = delegate_factory
        self._slots = asyncio.Semaphore(3)
        self.fault = None
        self.kind = "controlled-transport" if delegate_factory is not None else "live-model"

    def _stop(self, code):
        self.fault = self.fault or code
        path = base._path(self.root, "stopped.json")
        if not path.exists():
            try:
                base._write(self.root, "stopped.json", {"schema_version": "serveflow-task-stop-v1",
                    "error_code": self.fault, "recorded_at": base._now()})
            except Exception:
                pass

    def _validate(self, request, frame):
        if self.fault or frame is None or request.method != "POST" or str(request.url) != messages_url(self.base_url):
            raise TrialTransportError("protocol")
        raw = request.content
        if len(raw) > HTTP_LIMITS["request_bytes"]:
            raise TrialTransportError("protocol")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
        expected = AGENT_PARAMETERS.get(frame.agent) if frame.phase == "agent" else PHASE_PARAMETERS.get(frame.phase)
        if (expected is None or set(payload) - {"model", "messages", "max_tokens", "temperature", "thinking", "system", "tools", "stream"}
                or payload.get("model") != base.MODEL or type(payload.get("max_tokens")) is not int
                or type(payload.get("temperature")) not in {int, float} or not math.isfinite(payload["temperature"])
                or (payload["max_tokens"], payload.get("temperature")) != expected
                or payload.get("thinking") != {"type": "disabled"} or payload.get("stream", False) is not False
                or not isinstance(payload.get("messages"), list)):
            raise TrialTransportError("protocol")
        if frame.phase != "agent" and ("system" in payload or "tools" in payload):
            raise TrialTransportError("protocol")
        return canonical_payload_hash(payload), hashlib.sha256(raw).hexdigest()

    async def handle_async_request(self, request):
        async with self._slots:
            frame, operation, guard, started = CURRENT.get(), None, None, time.perf_counter()
            admission_error = None
            try:
                payload_hash, wire_hash = self._validate(request, frame)
                def admit():
                    nonlocal operation, admission_error
                    try:
                        if self.fault or (self.root / "stopped.json").exists():
                            raise TrialTransportError("protocol")
                        self.verify_admission()
                        operation = self.budget.reserve(frame.case_id, frame.turn_id, frame.phase, frame.agent,
                            payload_hash, wire_hash, lease=self.lease)
                    except Exception as error:
                        admission_error = getattr(error, "code", "storage_failure")
                        raise
                delegate = self.factory(frame) if self.factory else None
                if self.factory is not None and not isinstance(delegate, httpx.MockTransport):
                    raise TrialTransportError("protocol")
                # Hashes here are dynamic admission evidence, not a claim that
                # unknown future native-tool messages were frozen in advance.
                guard = GuardedIntentTransport(self.base_url, payload_hash, admit, transport=delegate)
                response = await guard.handle_async_request(request)
                raw = response.content
                try:
                    parsed = _response(raw)
                    outcome = "received"
                except (ValueError, TypeError, KeyError, RecursionError, OverflowError, TrialTransportError):
                    parsed = {"content": [], "upstream_model": None, "stop_reason": None,
                              "usage": {"input_tokens": None, "output_tokens": None}}
                    outcome = "structure_failed"
                self._seal(operation, frame, guard, parsed, outcome, None,
                           (time.perf_counter() - started) * 1000, hashlib.sha256(raw).hexdigest())
                return response
            except BaseException as error:
                code = admission_error
                if code is None:
                    code = ("cancelled" if isinstance(error, asyncio.CancelledError)
                            else base._error(guard.failure_code) if guard and guard.failure_code else "protocol")
                if code not in {"auth", "quota", "rate_limit", "network", "timeout", "cancelled", "response_limit", "protocol", "storage_failure"}:
                    code = "protocol"
                self._stop(code)
                if operation is not None:
                    try:
                        self._seal(operation, frame, guard, {"content": [], "upstream_model": None,
                            "stop_reason": None, "usage": {"input_tokens": None, "output_tokens": None}},
                            "cancelled" if code == "cancelled" else "execution_error", code,
                            (time.perf_counter() - started) * 1000, None)
                    except Exception:
                        pass  # Reservation remains occupied/unknown, never refunded.
                raise TrialTransportError(code) from None

    def _seal(self, operation, frame, guard, response, outcome, error_code, latency_ms, response_hash):
        if operation is None:
            raise TrialTransportError("protocol")
        name = f"physical-{operation['attempt']:03d}.json"
        value = {"schema_version": "serveflow-task-physical-v1", "evidence_kind": self.kind,
            "operation_id": operation["operation_id"], "attempt": operation["attempt"],
            "case_id": frame.case_id, "turn_id": frame.turn_id, "phase": frame.phase, "agent": frame.agent,
            "payload_sha256": operation["payload_sha256"], "request_bytes_sha256": operation["request_bytes_sha256"],
            "request_model": base.MODEL, "request_id": guard.request_id if guard else None,
            "http_status": guard.http_status if guard else None, "delegate_entered": bool(guard and guard.sent),
            "error_code": error_code, "outcome": outcome, "latency_ms": latency_ms,
            "response_sha256": response_hash, "response": response}
        base._write(self.root, name, value)
        from evaluation import benchmark as bench
        self.budget.finish(operation["operation_id"], outcome, bench.sha256(self.root / name), error_code,
                           lease=self.lease)
