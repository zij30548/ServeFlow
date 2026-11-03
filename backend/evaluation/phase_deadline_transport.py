"""Closed phase-following deadlines for one newly frozen quality experiment.

Application calls and historical trial transports do not import/select this
class. Model-stage budgets are unchanged. The SDK's 30s ceiling is tightened
per request to that stage's configured budget; local audit work stays bounded
separately and cannot hide network work. No URL/model/timeout request option is
introduced, no retry is enabled, and importing this module performs no I/O.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import math

import httpx

from core.model_deadline import assert_model_active
from evaluation.intent_transport import TrialTransportError
from evaluation.quality_trial import QualityHTTPTransport
from evaluation.task_transport import CURRENT, Frame, HTTP_LIMITS, _ObservedIntentGuard


PHASE_DEADLINES = {
    "intent": 8, "agent": 15, "rag_rewrite": 10,
    "rag_rerank": 30, "composer": 8, "judge": 8,
}
SDK_TIMEOUT_S = 30
PHASE_HTTP_LIMITS = {**HTTP_LIMITS, "request_deadline_s": SDK_TIMEOUT_S}
_TIMEOUT_FIELDS = frozenset({"connect", "read", "write", "pool"})


def _phase_limit(frame):
    if type(frame) is not Frame or type(frame.phase) is not str or frame.phase not in PHASE_DEADLINES:
        raise TrialTransportError("invalid_request")
    value = PHASE_DEADLINES[frame.phase]
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= SDK_TIMEOUT_S:
        raise TrialTransportError("invalid_request")
    return float(value)


def _sdk_timeout(request):
    values = request.extensions.get("timeout")
    if (type(values) is not dict or set(values) != _TIMEOUT_FIELDS
            or any(type(value) not in {int, float} or not math.isfinite(value) or value != SDK_TIMEOUT_S
                   for value in values.values())):
        # A new runner accidentally retaining the SDK's old 8s/default values
        # must fail before a quota claim, rather than silently clipping a role.
        raise TrialTransportError("invalid_request")


def _uncancelled():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        raise TrialTransportError("invalid_request") from None
    if task is None:
        raise TrialTransportError("invalid_request")
    if task.cancelling():
        # Synchronous admission may observe cancellation before the loop next
        # delivers it. Never enter a new delegate during that interval.
        raise asyncio.CancelledError


class _PhaseObservedGuard(_ObservedIntentGuard):
    """No caller-supplied timeout: only the trusted current stage selects it."""

    def __init__(self, base_url, expected_payload_sha256, before_send, *, transport=None,
                 response_limit=262_144):
        self._phase_frame = CURRENT.get()
        self._phase_limit_s = _phase_limit(self._phase_frame)
        super().__init__(base_url, expected_payload_sha256, before_send, transport=transport,
                         timeout_s=self._phase_limit_s, response_limit=response_limit)

    def _timeout_ceiling(self):
        return float(SDK_TIMEOUT_S)

    def _prepare_network_request(self, request):
        # Recheck after atomic admission and before delegate entry. An expired
        # stage, stale frame or a change to SDK settings cannot start a send.
        assert_model_active()
        _uncancelled()
        if CURRENT.get() != self._phase_frame:
            raise TrialTransportError("invalid_request")
        _sdk_timeout(request)
        request.extensions["timeout"] = {key: self._phase_limit_s for key in _TIMEOUT_FIELDS}


class PhaseDeadlineTaskHTTPTransport(QualityHTTPTransport):
    """Complete quality payload audit plus a per-phase one-send guard.

    Diagnostics are configured limits, not measured network/provider processing
    times. Their context is request-local and copied into cleanup-only finalizers.
    """

    _guard_type = _PhaseObservedGuard

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configured_limits = ContextVar("serveflow_phase_configured_limits", default=None)

    def _validate(self, request, frame):
        _uncancelled()
        _phase_limit(frame)
        _sdk_timeout(request)
        return super()._validate(request, frame)

    def _physical_diagnostics(self):
        value = super()._physical_diagnostics()
        limits = self._configured_limits.get()
        return value if limits is None else {**value, **limits}

    async def handle_async_request(self, request):
        frame = CURRENT.get()
        try:
            limit = _phase_limit(frame)
        except TrialTransportError:
            self._stop("protocol")
            raise TrialTransportError("protocol") from None
        token = self._configured_limits.set({"network_limit_s": limit,
            "sdk_limit_s": float(SDK_TIMEOUT_S), "active_stage_limit_s": limit})
        try:
            return await super().handle_async_request(request)
        finally:
            self._configured_limits.reset(token)
