"""Case-scoped failure isolation for the explicitly approved coverage continuation.

The original protocol, stop receipts and twenty occupied attempts are immutable.
Only this fixed continuation may finish a failed independent case and advance;
it never resends a case, retries a request, changes a model deadline or refunds a
claim. Existing sent siblings drain under their original deadlines. Cancellation
or unprovable cleanup is always a permanent stop.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import time

import anyio
import httpx

from evaluation import benchmark as bench, intent_trial as base
from evaluation.coverage_budget import ROOT
from evaluation.coverage_execution import CoverageTransport
from evaluation.intent_transport import TrialTransportError
from evaluation.phase_deadline_transport import _PhaseObservedGuard, _phase_limit
from evaluation.task_evidence import EvidenceError, FATAL_EVIDENCE_CODES
from evaluation.task_transport import CURRENT, TIMING_FIELDS, _response


MANIFEST = "continuation-v1.json"
STOP_FILE = "continuation-v1-stop.json"
SCHEMA = "serveflow-coverage-continuation-v1"
MAX_DRAIN_S = 31.0
FORCE_REAP_S = 1.0
_SOFT = frozenset({"timeout", "network"})
_CODES = frozenset({"auth", "quota", "rate_limit", "network", "timeout", "cancelled",
                    "response_limit", "protocol", "storage_failure"})


class CaseExecutionFailure(TrialTransportError):
    """Expected abort of an already failed case, not a reusable global bypass."""

    def __init__(self, case_id, code):
        self.case_id = case_id
        super().__init__(code)


class _ContinuationGuard(_PhaseObservedGuard):
    def __init__(self, *args, continuation_ready, **kwargs):
        self._continuation_ready = continuation_ready
        super().__init__(*args, **kwargs)

    def _prepare_network_request(self, request):
        super()._prepare_network_request(request)
        self._continuation_ready()


class CoverageContinuationTransport(CoverageTransport):
    """Unchanged request checks, clocks and seals, with explicit case lifecycle.

    ``begin_case`` follows the ledger's atomic case start. The runner must call
    ``finish_case`` even when a model/runtime raises, and cannot advance until
    all occupied physical requests are sealed and the current callers drained.
    Neither method changes the ledger's plan or its total 480-request limit.
    """

    def __init__(self, *args, continuation_sha256, **kwargs):
        if type(continuation_sha256) is not str or not re.fullmatch(r"[a-f0-9]{64}", continuation_sha256):
            raise TrialTransportError("protocol")
        super().__init__(*args, **kwargs)
        if self.root != ROOT and self.factory is None:
            raise TrialTransportError("protocol")
        self.continuation_sha256 = continuation_sha256
        self._case_id = None
        self._case_open = False
        self._case_closed = True
        self._closed_cases = set()
        self._tasks = set()
        self._case_failure = None
        self._manifest_guard()

    def _manifest_guard(self):
        path = base._path(self.root, MANIFEST)
        if (bench.sha256(path) != self.continuation_sha256
                or bench.read_json(path).get("schema_version") != SCHEMA
                or (self.root / STOP_FILE).exists()):
            raise TrialTransportError("protocol")

    @property
    def case_failure(self):
        return deepcopy(self._case_failure)

    def begin_case(self, case_id):
        self._manifest_guard()
        if (self.fault or not self._case_closed or self._tasks or case_id in self._closed_cases
                or case_id not in self._jobs):
            raise TrialTransportError("protocol")
        state = self.budget.status()
        if state["active_case"] != case_id or state["pending"] or state["paused"]:
            raise TrialTransportError("protocol")
        self._case_id, self._case_open, self._case_closed = case_id, True, False
        self._case_failure = None

    def assert_case_healthy(self):
        if self.fault:
            raise TrialTransportError(self.fault)
        if self._case_failure is not None:
            raise CaseExecutionFailure(self._case_id, self._case_failure["error_code"])
        if not self._case_open or self._case_closed:
            raise TrialTransportError("protocol")

    def assert_case_active(self):
        self.assert_case_healthy()
        frame = CURRENT.get()
        if (not self._case_open or self._case_closed or frame is None
                or frame.case_id != self._case_id or frame.case_id in self._closed_cases):
            raise TrialTransportError("protocol")

    def _validate(self, request, frame):
        self._manifest_guard()
        self.assert_case_active()
        return super()._validate(request, frame)

    def _stop(self, code, *, evidence_error=None):
        # This is deliberately permanent. Only the sent-request exception
        # boundary below can establish an isolated timeout/network failure.
        self.fault = self.fault or (code if code in _CODES else "protocol")
        self._case_open = False
        path = base._path(self.root, STOP_FILE)
        if not path.exists():
            value = {"schema_version": "serveflow-coverage-continuation-stop-v1",
                     "continuation_sha256": self.continuation_sha256,
                     "error_code": self.fault, "recorded_at": base._now(), "case_id": self._case_id}
            if evidence_error in FATAL_EVIDENCE_CODES:
                value["evidence_error"] = evidence_error
            try:
                base._write(self.root, STOP_FILE, value)
            except Exception:
                self.fault = "storage_failure"

    def _case_stop(self, code, frame):
        if self._case_failure is None:
            self._case_failure = {"case_id": self._case_id, "error_code": code,
                                  "turn_id": frame.turn_id, "phase": frame.phase,
                                  "agent": frame.agent, "sent": True}
        self._case_open = False

    async def _cancel_and_reap(self, tasks):
        for task in tasks:
            if not task.done():
                task.cancel()
        if not tasks:
            return
        # Finish this bounded cleanup even if the operator cancels repeatedly.
        # It never reopens admission or converts cancellation into timeout.
        with anyio.CancelScope(shield=True):
            cleanup = asyncio.create_task(asyncio.wait(tasks, timeout=FORCE_REAP_S))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            done, _ = cleanup.result()
            for task in done:
                if not task.cancelled():
                    task.exception()

    async def finish_case(self):
        """Close admission and drain already-sent siblings, never retry them."""
        if self._case_id is None or self._case_closed:
            raise TrialTransportError("protocol")
        self._case_open = False
        tasks = set(self._tasks) - {asyncio.current_task()}
        try:
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=MAX_DRAIN_S)
                if pending:
                    self._stop("storage_failure")
                    await self._cancel_and_reap(pending)
                    raise TrialTransportError("storage_failure")
                for task in tasks:
                    if not task.cancelled():
                        task.exception()
            state = self.budget.status()
            if self._tasks or state["pending"]:
                self._stop("storage_failure")
                raise TrialTransportError("storage_failure")
            self._case_closed = True
            self._closed_cases.add(self._case_id)
            return self.case_failure
        except asyncio.CancelledError:
            self._stop("cancelled")
            await self._cancel_and_reap(tasks)
            raise

    async def handle_async_request(self, request):
        """Keep the parent's send algorithm, replacing only stop ownership.

        The historical implementation hardcodes its immutable stopped.json in
        the admission closure. This explicit version uses the separately hashed
        continuation receipt; no old source or stop file is edited or ignored.
        """
        frame, task = CURRENT.get(), asyncio.current_task()
        self.assert_case_active()
        self._tasks.add(task)
        stage_token = self._stage_reason.set(None)
        limits_token = None
        try:
            limit = _phase_limit(frame)
            limits_token = self._configured_limits.set({"network_limit_s": limit,
                "sdk_limit_s": 30.0, "active_stage_limit_s": limit})
            return await self._handle_current(request)
        except CaseExecutionFailure:
            raise
        except asyncio.CancelledError:
            self._stop("cancelled")
            raise
        except TrialTransportError:
            raise
        except TimeoutError:
            self._stop("timeout")
            raise TrialTransportError("timeout") from None
        except BaseException:
            self._stop("protocol")
            raise
        finally:
            if limits_token is not None:
                self._configured_limits.reset(limits_token)
            self._stage_reason.reset(stage_token)
            self._tasks.discard(task)

    async def _handle_current(self, request):
        queued_at = time.perf_counter()
        async with self._request_slot():
            frame, operation, guard, started = CURRENT.get(), None, None, time.perf_counter()
            queue_ms = max(0.0, (started - queued_at) * 1000)
            admission_ms = 0.0
            received_response = received_hash = admission_error = evidence_error = None
            try:
                payload_hash, wire_hash = self._validate(request, frame)
                admission_started = time.perf_counter()
                try:
                    preverified = await self._prepare_admission()
                    if type(preverified) is not bool:
                        raise TrialTransportError("protocol")
                except CaseExecutionFailure:
                    raise
                except Exception as error:
                    admission_error = getattr(error, "code", "storage_failure")
                    raise
                finally:
                    admission_ms = max(0.0, (time.perf_counter() - admission_started) * 1000)
                if preverified and self._validate(request, frame) != (payload_hash, wire_hash):
                    admission_error = "protocol"
                    raise TrialTransportError("protocol")

                def admit():
                    nonlocal operation, admission_error, evidence_error
                    try:
                        self._manifest_guard()
                        self.assert_case_active()
                        if not preverified:
                            self.verify_admission()
                        elif self._validate(request, frame) != (payload_hash, wire_hash):
                            raise TrialTransportError("protocol")
                        if frame.evidence_guard is not None:
                            frame.evidence_guard()
                        operation = self.budget.reserve(frame.case_id, frame.turn_id, frame.phase, frame.agent,
                            payload_hash, wire_hash, lease=self.lease)
                    except CaseExecutionFailure:
                        raise
                    except Exception as error:
                        admission_error = getattr(error, "code", "storage_failure")
                        if isinstance(error, EvidenceError) and error.reason in FATAL_EVIDENCE_CODES:
                            evidence_error = error.reason
                        raise

                delegate = self.factory(frame) if self.factory else None
                if self.factory is not None and not isinstance(delegate, httpx.MockTransport):
                    raise TrialTransportError("protocol")
                guard = _ContinuationGuard(self.base_url, payload_hash, admit, transport=delegate,
                    continuation_ready=lambda: (self._manifest_guard(), self.assert_case_active()))
                response = await guard.handle_async_request(request)
                raw = response.content
                try:
                    parsed, outcome = _response(raw), "received"
                except (ValueError, TypeError, KeyError, RecursionError, OverflowError, TrialTransportError):
                    parsed, outcome = {"content": [], "upstream_model": None, "stop_reason": None,
                                       "usage": {"input_tokens": None, "output_tokens": None}}, "structure_failed"
                received_response, received_hash = parsed, hashlib.sha256(raw).hexdigest()
                await self._seal_physical(operation, frame, guard, parsed, outcome, None,
                    (time.perf_counter() - started) * 1000, received_hash,
                    timing=self._physical_timing(queued_at, queue_ms, admission_ms, guard), timing_started_at=queued_at)
                return response
            except CaseExecutionFailure:
                if operation is not None:
                    self._stop("protocol")
                    raise TrialTransportError("protocol") from None
                raise
            except BaseException as error:
                code = self._failure_code(error, admission_error, guard)
                code = code if code in _CODES else "protocol"
                if code in _SOFT and operation is not None and guard is not None and guard.sent:
                    self._case_stop(code, frame)
                else:
                    self._stop(code, evidence_error=evidence_error)
                if operation is not None:
                    try:
                        await self._seal_after_error(operation, frame, guard,
                            received_response if received_response is not None else {
                                "content": [], "upstream_model": None, "stop_reason": None,
                                "usage": {"input_tokens": None, "output_tokens": None}},
                            "cancelled" if code == "cancelled" else "execution_error", code,
                            (time.perf_counter() - started) * 1000, received_hash,
                            timing=self._physical_timing(queued_at, queue_ms, admission_ms, guard), timing_started_at=queued_at)
                    except BaseException:
                        self._stop("storage_failure")
                        raise
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
            "response_sha256": response_hash, "response": response,
            "continuation_sha256": self.continuation_sha256}
        if timing is not None:
            if (type(timing) is not dict or set(timing) != TIMING_FIELDS
                    or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in timing.values())):
                raise TrialTransportError("protocol")
            value["timing"] = {key: round(timing[key], 3) for key in sorted(TIMING_FIELDS)}
        diagnostics = self._physical_diagnostics()
        numeric = {"network_limit_s", "sdk_limit_s", "active_stage_limit_s"}
        if (type(diagnostics) is not dict or set(diagnostics) - ({"stage_deadline_reason"} | numeric)
                or "stage_deadline_reason" in diagnostics and diagnostics["stage_deadline_reason"] not in {"active", "overhead", "wall"}
                or any(type(diagnostics[k]) not in (int, float) or not math.isfinite(diagnostics[k])
                       or not 0 < diagnostics[k] <= 30 for k in numeric & set(diagnostics))):
            raise TrialTransportError("protocol")
        value.update(diagnostics)
        base._write(self.root, name, value)
        self.budget.finish(operation["operation_id"], outcome, bench.sha256(self.root / name), error_code, lease=self.lease)
