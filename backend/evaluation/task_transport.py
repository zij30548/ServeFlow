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
import inspect
import json
import math
import time
from types import SimpleNamespace
from typing import Callable

import httpx

from evaluation import intent_trial as base
from evaluation.task_evidence import EvidenceError, FATAL_EVIDENCE_CODES
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
    evidence_guard: Callable[[], None] | None = None


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
        if frame.evidence_guard is not None:
            frame.evidence_guard()
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


TIMING_FIELDS = frozenset({"queue_ms", "admission_ms", "network_ms", "cleanup_ms", "wall_ms"})


class _ObservedIntentGuard(GuardedIntentTransport):
    """Observe the unchanged one-send guard without changing its deadlines."""

    def __init__(self, base_url, expected_payload_sha256, before_send, **kwargs):
        self.network_ms = self.cleanup_ms = 0.0
        self._network_started = None

        def observed_admission():
            value = before_send()
            # The inherited guard still rejects an awaitable before_send. The
            # runtime's actual admission/reservation callback is synchronous.
            self._network_started = time.perf_counter()
            return value

        super().__init__(base_url, expected_payload_sha256, observed_admission, **kwargs)

    async def _cleanup(self, response):
        started = time.perf_counter()
        if self._network_started is not None:
            self.network_ms = max(0.0, (started - self._network_started) * 1000)
        try:
            return await super()._cleanup(response)
        finally:
            self.cleanup_ms = max(0.0, (time.perf_counter() - started) * 1000)


class TaskHTTPTransport(httpx.AsyncBaseTransport):
    """All stage facades share this physical-send gate and durable ledger.

    The default keeps its original inline verification inside before_send.
    Only the explicit subclass offloads read-only admission verification.
    """

    _guard_type = None

    def __init__(self, base_url, root, budget, lease, verify_admission, *, delegate_factory=None):
        self.base_url = validated_endpoint(base_url)
        self.root, self.budget, self.lease = root, budget, lease
        self.verify_admission = verify_admission
        self.factory = delegate_factory
        self._slots = asyncio.Semaphore(3)
        self.fault = None
        self.kind = "controlled-transport" if delegate_factory is not None else "live-model"

    def _stop(self, code, *, evidence_error=None):
        self.fault = self.fault or code
        path = base._path(self.root, "stopped.json")
        if not path.exists():
            try:
                value = {"schema_version": "serveflow-task-stop-v1",
                         "error_code": self.fault, "recorded_at": base._now()}
                if evidence_error in FATAL_EVIDENCE_CODES:
                    value["evidence_error"] = evidence_error
                base._write(self.root, "stopped.json", value)
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
        if (expected is None or set(payload) - {"model", "messages", "max_tokens", "temperature", "thinking", "system", "tools", "stream", "tool_choice"}
                or payload.get("model") != base.MODEL or type(payload.get("max_tokens")) is not int
                or type(payload.get("temperature")) not in {int, float} or not math.isfinite(payload["temperature"])
                or (payload["max_tokens"], payload.get("temperature")) != expected
                or payload.get("thinking") != {"type": "disabled"} or payload.get("stream", False) is not False
                or not isinstance(payload.get("messages"), list)):
            raise TrialTransportError("protocol")
        if frame.phase != "agent" and ("system" in payload or "tools" in payload or "tool_choice" in payload):
            raise TrialTransportError("protocol")
        if "tool_choice" in payload:
            choice = payload["tool_choice"]
            named_roles = {
                "search_knowledge_base": {"general", "technical", "billing", "escalation"},
                "read_demo_order": {"general", "billing"},
                "build_diagnostic_plan": {"technical"},
                "lookup_error_code": {"technical"},
            }
            if (frame.phase != "agent" or type(choice) is not dict
                    or set(choice) != {"type", "name"} or choice.get("type") != "tool"
                    or type(choice.get("name")) is not str
                    or choice["name"] not in named_roles or frame.agent not in named_roles[choice["name"]]
                    or type(payload.get("tools")) is not list
                    or sum(type(item) is dict and item.get("name") == choice["name"]
                           for item in payload["tools"]) != 1):
                raise TrialTransportError("protocol")
        return canonical_payload_hash(payload), hashlib.sha256(raw).hexdigest()

    async def _prepare_admission(self):
        """Optional async preflight; False preserves original inline checking.

        An override returning True must have completed the exact configured
        verify_admission callback, without reserving or sending anything.
        """
        return False

    def _physical_timing(self, queued_at, queue_ms, admission_ms, guard):
        return None

    async def _seal_physical(self, *args, timing_started_at=None, **kwargs):
        self._seal(*args, **kwargs)

    async def _seal_after_error(self, *args, **kwargs):
        await self._seal_physical(*args, **kwargs)

    async def handle_async_request(self, request):
        queued_at = time.perf_counter()
        async with self._slots:
            frame, operation, guard, started = CURRENT.get(), None, None, time.perf_counter()
            queue_ms = max(0.0, (started - queued_at) * 1000)
            admission_ms = 0.0
            received_response = None
            received_hash = None
            admission_error = None
            evidence_error = None
            try:
                payload_hash, wire_hash = self._validate(request, frame)
                admission_started = time.perf_counter()
                try:
                    preverified = await self._prepare_admission()
                    if type(preverified) is not bool:
                        raise TrialTransportError("protocol")
                except Exception as error:
                    admission_error = getattr(error, "code", "storage_failure")
                    raise
                finally:
                    admission_ms = max(0.0, (time.perf_counter() - admission_started) * 1000)
                if preverified:
                    # A caller can mutate an httpx.Request while verification
                    # awaits a worker thread. Canonical equality alone is not
                    # enough: the exact wire bytes must still match as well.
                    if self._validate(request, frame) != (payload_hash, wire_hash):
                        admission_error = "protocol"
                        raise TrialTransportError("protocol")
                def admit():
                    nonlocal operation, admission_error, evidence_error
                    try:
                        if self.fault or (self.root / "stopped.json").exists():
                            raise TrialTransportError("protocol")
                        if not preverified:
                            self.verify_admission()
                        elif self._validate(request, frame) != (payload_hash, wire_hash):
                            raise TrialTransportError("protocol")
                        # Recheck after waiting for a slot and after complete
                        # protocol checks. Another role may have latched a fault
                        # after this request's facade was entered.
                        if frame.evidence_guard is not None:
                            frame.evidence_guard()
                        operation = self.budget.reserve(frame.case_id, frame.turn_id, frame.phase, frame.agent,
                            payload_hash, wire_hash, lease=self.lease)
                    except Exception as error:
                        admission_error = getattr(error, "code", "storage_failure")
                        if isinstance(error, EvidenceError) and error.reason in FATAL_EVIDENCE_CODES:
                            evidence_error = error.reason
                        raise
                delegate = self.factory(frame) if self.factory else None
                if self.factory is not None and not isinstance(delegate, httpx.MockTransport):
                    raise TrialTransportError("protocol")
                # Hashes here are dynamic admission evidence, not a claim that
                # unknown future native-tool messages were frozen in advance.
                guard_factory = self._guard_type or GuardedIntentTransport
                guard = guard_factory(self.base_url, payload_hash, admit, transport=delegate)
                response = await guard.handle_async_request(request)
                raw = response.content
                try:
                    parsed = _response(raw)
                    outcome = "received"
                except (ValueError, TypeError, KeyError, RecursionError, OverflowError, TrialTransportError):
                    parsed = {"content": [], "upstream_model": None, "stop_reason": None,
                              "usage": {"input_tokens": None, "output_tokens": None}}
                    outcome = "structure_failed"
                received_response = parsed
                received_hash = hashlib.sha256(raw).hexdigest()
                await self._seal_physical(operation, frame, guard, parsed, outcome, None,
                           (time.perf_counter() - started) * 1000, received_hash,
                           timing=self._physical_timing(queued_at, queue_ms, admission_ms, guard),
                           timing_started_at=queued_at)
                return response
            except BaseException as error:
                code = admission_error
                if code is None:
                    code = ("cancelled" if isinstance(error, asyncio.CancelledError)
                            else base._error(guard.failure_code) if guard and guard.failure_code else "protocol")
                if code not in {"auth", "quota", "rate_limit", "network", "timeout", "cancelled", "response_limit", "protocol", "storage_failure"}:
                    code = "protocol"
                self._stop(code, evidence_error=evidence_error)
                if operation is not None:
                    try:
                        await self._seal_after_error(operation, frame, guard,
                            received_response if received_response is not None else {
                                "content": [], "upstream_model": None, "stop_reason": None,
                                "usage": {"input_tokens": None, "output_tokens": None}},
                            "cancelled" if code == "cancelled" else "execution_error", code,
                            (time.perf_counter() - started) * 1000, received_hash,
                            timing=self._physical_timing(queued_at, queue_ms, admission_ms, guard),
                            timing_started_at=queued_at)
                    except Exception:
                        pass  # Reservation remains occupied/unknown, never refunded.
                raise TrialTransportError(code) from None

    def _seal(self, operation, frame, guard, response, outcome, error_code, latency_ms, response_hash, *, timing=None):
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
        if timing is not None:
            if (type(timing) is not dict or set(timing) != TIMING_FIELDS
                    or any(type(number) not in (int, float) or not math.isfinite(number) or number < 0
                           for number in timing.values())):
                raise TrialTransportError("protocol")
            value["timing"] = {key: round(timing[key], 3) for key in sorted(TIMING_FIELDS)}
        base._write(self.root, name, value)
        from evaluation import benchmark as bench
        self.budget.finish(operation["operation_id"], outcome, bench.sha256(self.root / name), error_code,
                           lease=self.lease)


class ThreadedTaskHTTPTransport(TaskHTTPTransport):
    """Read-only full verification off-loop; reserve and send remain on-loop.

    Cancellation of to_thread does not terminate its read-only callback. Its
    eventual completion cannot resume this canceled coroutine, reserve quota,
    or send. No callback is skipped, cached or replaced with a partial check.

    Optional physical timing is local elapsed observation: queue is the shared
    request-slot wait; admission includes thread scheduling and full callback;
    network runs after atomic admission through response read/error; cleanup
    measures the unchanged bounded close; wall starts before the slot wait.
    Parsing/scheduling/reservation overhead need not equal the phase sum. These
    observations cannot establish a provider fault or remote processing time.
    """

    _guard_type = _ObservedIntentGuard

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(self.verify_admission) or inspect.iscoroutinefunction(self.verify_admission):
            raise TrialTransportError("protocol")
        # Full verification reads this same ledger with nonblocking flock.
        # Serializing it with on-loop sealing avoids falsely calling a local
        # read/write race a storage failure. Waiting is asynchronous; network
        # response reads and their original deadlines remain independent.
        self._metadata_gate = asyncio.Lock()

    async def _prepare_admission(self):
        async with self._metadata_gate:
            worker = asyncio.create_task(asyncio.to_thread(self.verify_admission))
            try:
                verified = await asyncio.shield(worker)
                if inspect.isawaitable(verified):
                    if inspect.iscoroutine(verified):
                        verified.close()
                    raise TrialTransportError("protocol")
            except asyncio.CancelledError:
                # to_thread cannot stop the actual reader. Reap it before
                # releasing the metadata gate; never continue to reserve/send.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not worker.cancelled():
                    worker.exception()  # Consume a late validation exception.
                raise
        # No await separates this return, payload recheck and atomic reserve.
        return True

    async def _seal_physical(self, *args, timing_started_at=None, **kwargs):
        async with self._metadata_gate:
            if kwargs.get("timing") is not None and timing_started_at is not None:
                # Include asynchronous metadata-gate wait, but never claim to
                # measure this immutable record's own fsync/hash/ledger finish.
                kwargs["timing"] = {**kwargs["timing"],
                    "wall_ms": max(0.0, (time.perf_counter() - timing_started_at) * 1000)}
            self._seal(*args, **kwargs)

    async def _seal_after_error(self, *args, **kwargs):
        # The body may be fully received before cancellation while waiting to
        # seal. Preserve that known response with a canceled outcome. Repeated
        # cancellation cannot turn it into a blank response or duplicate send.
        finalizer = asyncio.create_task(self._seal_physical(*args, **kwargs))
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                continue
        return finalizer.result()

    async def handle_async_request(self, request):
        try:
            return await super().handle_async_request(request)
        except asyncio.CancelledError:
            # The parent's historical queue wait is outside its try block.
            # Only this explicit subclass adds the sticky queue-cancel stop.
            self._stop("cancelled")
            raise

    def _physical_timing(self, queued_at, queue_ms, admission_ms, guard):
        return {
            "queue_ms": queue_ms,
            "admission_ms": admission_ms,
            "network_ms": guard.network_ms if guard is not None else 0.0,
            "cleanup_ms": guard.cleanup_ms if guard is not None else 0.0,
            "wall_ms": max(0.0, (time.perf_counter() - queued_at) * 1000),
        }
