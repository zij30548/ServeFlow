"""Second fixed continuation identity and typed secondary-error provenance.

No business/runtime source is edited. The observer preserves the original
capture's permissions, argument checks, handler and result, and records only a
small fixed diagnostic vocabulary after a real physical failure was sealed.
An arbitrary tool_handler_failed is never recoverable merely by its name.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import threading

from anthropic import APIConnectionError
import httpx

from core.model_deadline import ModelDeadlineExceeded
from evaluation import benchmark as bench, intent_trial as base
from evaluation.coverage_continuation_transport import (
    CoverageContinuationTransport, CaseExecutionFailure,
)
from evaluation.intent_transport import messages_url, TrialTransportError
from evaluation.task_evidence import CaptureTools, EvidenceError, FATAL_EVIDENCE_CODES, TOOL_NAMES
from evaluation.task_transport import CURRENT


MANIFEST = "continuation-v2.json"
STOP_FILE = "continuation-v2-stop.json"
SCHEMA = "serveflow-coverage-continuation-v2"
_PATCH_LOCK = threading.Lock()
_SOFT = frozenset({"timeout", "network"})
_CODES = frozenset({"auth", "quota", "rate_limit", "network", "timeout", "cancelled",
                    "response_limit", "protocol", "storage_failure"})


class CoverageContinuation2Transport(CoverageContinuationTransport):
    def __init__(self, *args, **kwargs):
        self._sealed_soft_errors = {}
        super().__init__(*args, **kwargs)

    def _manifest_guard(self):
        path = base._path(self.root, MANIFEST)
        if (bench.sha256(path) != self.continuation_sha256
                or bench.read_json(path).get("schema_version") != SCHEMA
                or (self.root / STOP_FILE).exists()):
            raise TrialTransportError("protocol")

    def _stop(self, code, *, evidence_error=None):
        self.fault = self.fault or (code if code in _CODES else "protocol")
        self._case_open = False
        path = base._path(self.root, STOP_FILE)
        if not path.exists():
            value = {"schema_version": SCHEMA + "-stop", "continuation_sha256": self.continuation_sha256,
                     "error_code": self.fault, "recorded_at": base._now(), "case_id": self._case_id}
            if evidence_error in FATAL_EVIDENCE_CODES:
                value["evidence_error"] = evidence_error
            try:
                base._write(self.root, STOP_FILE, value)
            except Exception:
                self.fault = "storage_failure"

    def _seal(self, operation, frame, guard, response, outcome, error_code, latency_ms, response_hash, *, timing=None):
        # super writes the immutable physical record AND durably finishes the
        # occupied ledger claim. A failed/uncertain seal never creates a proof.
        super()._seal(operation, frame, guard, response, outcome, error_code, latency_ms, response_hash, timing=timing)
        if (outcome == "execution_error" and error_code in _SOFT and guard is not None and guard.sent):
            physical = bench.read_json(self.root / f"physical-{operation['attempt']:03d}.json")
            self._sealed_soft_errors[operation["operation_id"]] = {
                **{key: operation[key] for key in ("operation_id", "attempt", "case_id", "turn_id", "phase", "agent", "request_bytes_sha256")},
                "outcome": outcome, "error_code": error_code, "delegate_entered": True,
                "output_sha256": bench.sha256(self.root / f"physical-{operation['attempt']:03d}.json"),
                "continuation_sha256": self.continuation_sha256,
                "stage_deadline_reason": physical.get("stage_deadline_reason"),
            }

    def sealed_case_error_proofs(self):
        return [deepcopy(row) for row in self._sealed_soft_errors.values() if row["case_id"] == self._case_id]


def _typed_proofs(error, transport, *, frame=None):
    failure = transport.case_failure
    if (transport.fault or type(failure) is not dict or failure.get("sent") is not True
            or failure.get("case_id") != transport._case_id or failure.get("error_code") not in _SOFT):
        return []
    proofs = transport.sealed_case_error_proofs()
    if isinstance(error, CaseExecutionFailure):
        return proofs if error.case_id == failure["case_id"] and error.code == failure["error_code"] else []
    if isinstance(error, APIConnectionError):
        request = getattr(error, "request", None)
        if (not isinstance(request, httpx.Request) or request.method != "POST"
                or str(request.url) != messages_url(transport.base_url)):
            return []
        wire = hashlib.sha256(request.content).hexdigest()
        return [row for row in proofs if row["request_bytes_sha256"] == wire
                and (frame is None or row["case_id"] == frame.case_id and row["turn_id"] == frame.turn_id
                     and row["agent"] == frame.agent)]
    if isinstance(error, ModelDeadlineExceeded) and error.reason == "active" and frame is not None:
        # Unlike SDK connection errors this exception has no HTTP request.
        # Require exact live stage identity; an unbound local timer is unknown.
        return [row for row in proofs if row["case_id"] == frame.case_id and row["turn_id"] == frame.turn_id
                and row["phase"] == frame.phase and row["agent"] == frame.agent and row["error_code"] == "timeout"]
    return []


class CaseErrorObserver:
    """Request-local evidence, containing no inputs, bodies or error strings."""
    def __init__(self, transport):
        self.transport = transport
        self.case_id = transport._case_id
        self.events, self.captures = [], []
        self.closed, self.invalid = False, False

    def _record(self, capture, error, *, site, tool=None):
        try:
            frame = CURRENT.get()
            proofs = _typed_proofs(error, self.transport, frame=frame)
            if (capture not in self.captures or frame is None or frame.case_id != self.case_id or len(self.events) >= 64
                    or site not in {"before_call", "handler"} or tool is not None and tool not in TOOL_NAMES):
                self.invalid = True
                return
            if isinstance(error, CaseExecutionFailure): kind = "CaseExecutionFailure"
            elif isinstance(error, APIConnectionError): kind = "APIConnectionError"
            elif isinstance(error, ModelDeadlineExceeded): kind = "ModelDeadlineExceeded"
            elif isinstance(error, asyncio.CancelledError): kind = "CancelledError"
            elif isinstance(error, EvidenceError): kind = "EvidenceError"
            else: kind = "other"
            self.events.append({"capture": self.captures.index(capture),
                "site": site, "tool": tool, "error_type": kind,
                "case_id": frame.case_id, "turn_id": frame.turn_id, "handler_phase": frame.phase, "agent": frame.agent,
                "typed_sealed_origin": bool(proofs), "proofs": proofs})
        except BaseException:
            # Diagnostics cannot replace the real exception or grant sends.
            self.invalid = True

    def summary(self):
        return {"schema_version": "serveflow-case-error-provenance-v1", "case_id": self.case_id,
                "closed": self.closed, "invalid": self.invalid, "events": deepcopy(self.events),
                "capture_faults": [capture.fault for capture in self.captures]}


@contextmanager
def observe_case_errors(transport):
    """Observe original handler exceptions before the capture wraps them.

    Only the isolated evaluation module references are replaced, never the
    application's tools, SDK, business handlers or global CaptureTools class.
    The factory is restored even after cancellation or failed construction.
    """
    from evaluation import coverage_runtime as shared, coverage_conversation_runtime as memory
    if not isinstance(transport, CoverageContinuation2Transport):
        raise TrialTransportError("protocol")
    observer = CaseErrorObserver(transport)
    if not _PATCH_LOCK.acquire(blocking=False):
        raise TrialTransportError("protocol")
    originals = shared.CaptureTools, memory.CaptureTools
    try:
        if any(value is not CaptureTools for value in originals):
            raise TrialTransportError("protocol")

        class ObservedCapture(CaptureTools):
            def __init__(self, *args, **kwargs):
                before = kwargs.get("before_call")
                if before is not None:
                    def observed_before(meta):
                        try:
                            value = before(meta)
                        except BaseException as error:
                            observer._record(self, error, site="before_call")
                            raise
                        if not inspect.isawaitable(value):
                            return value
                        async def wait():
                            try: return await value
                            except BaseException as error:
                                observer._record(self, error, site="before_call")
                                raise
                        return wait()
                    kwargs["before_call"] = observed_before
                super().__init__(*args, **kwargs)
                observer.captures.append(self)

            def _handler(self, role, scope, spec):
                async def original_handler(req, args):
                    try:
                        value = spec.handler(req, args)
                        return await value if inspect.isawaitable(value) else value
                    except BaseException as error:
                        observer._record(self, error, site="handler", tool=spec.name)
                        raise
                return super()._handler(role, scope, replace(spec, handler=original_handler))

        shared.CaptureTools = memory.CaptureTools = ObservedCapture
        yield observer
    finally:
        shared.CaptureTools, memory.CaptureTools = originals
        observer.closed = True
        _PATCH_LOCK.release()


def _reconcile_proofs(transport, proofs):
    state = transport.budget.status()
    if (state["pending"] or transport._tasks or not transport._case_closed or transport.fault):
        return False
    operations = transport.budget.results()["operations"]
    for proof in proofs:
        op = operations.get(proof["operation_id"])
        path = transport.root / f"physical-{proof['attempt']:03d}.json"
        if (type(op) is not dict or op.get("output_sha256") != proof["output_sha256"]
                or any(op.get(key) != proof[key] for key in ("case_id", "turn_id", "phase", "agent", "outcome", "error_code", "request_bytes_sha256"))
                or bench.sha256(path) != proof["output_sha256"]):
            return False
        physical = bench.read_json(path)
        if (physical.get("delegate_entered") is not True
                or physical.get("continuation_sha256") != transport.continuation_sha256
                or any(physical.get(key) != proof[key] for key in ("case_id", "turn_id", "phase", "agent", "outcome", "error_code", "request_bytes_sha256"))):
            return False
    return bool(proofs)


def is_recoverable_case_error(error, transport, observer):
    """Recognize a sealed transport failure, never a generic evidence failure."""
    try:
        if (type(observer) is not CaseErrorObserver or observer.transport is not transport
                or not observer.closed or observer.invalid or observer.case_id != transport._case_id
                or transport.fault or any(c.fault not in {None, "tool_handler_failed"} for c in observer.captures)):
            return False
        proofs = transport.sealed_case_error_proofs()
        if not _reconcile_proofs(transport, proofs):
            return False
        if any(not event["typed_sealed_origin"] for event in observer.events):
            return False
        if any(c.fault == "tool_handler_failed" and not any(
                event["capture"] == index and event["typed_sealed_origin"] for event in observer.events)
               for index, c in enumerate(observer.captures)):
            return False
        if error is None:
            return True
        if isinstance(error, EvidenceError):
            return (error.reason == "tool_handler_failed" and bool(observer.events)
                    and any(c.fault == "tool_handler_failed" for c in observer.captures))
        if isinstance(error, ModelDeadlineExceeded):
            return error.reason == "active" and (any(p["stage_deadline_reason"] == "active" for p in proofs)
                or any(e["error_type"] == "ModelDeadlineExceeded" and e["typed_sealed_origin"] for e in observer.events))
        return bool(_typed_proofs(error, transport))
    except BaseException:
        return False
