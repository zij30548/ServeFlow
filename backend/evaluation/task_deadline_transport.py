"""Evaluation-only accounting overhead, separate from model execution time.

No model deadline or network limit is raised. The existing complete verifier,
wire recheck, atomic reservation and one-send guard run unchanged. Only trusted
queue/admission/seal management sections pause the active model-stage clock;
their independent cumulative/wall limits belong to ``model_deadline``.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

from core.model_deadline import (
    ModelDeadlineExceeded, assert_model_active, current_deadline_failure_reason,
    overhead,
)
from evaluation.intent_transport import TrialTransportError
from evaluation.task_transport import ThreadedTaskHTTPTransport


class DeadlineTaskHTTPTransport(ThreadedTaskHTTPTransport):
    """Keep audit management outside the active stage timer, never the network.

    This class requires an explicitly enabled evaluation clock and a live
    ``await_model`` frame. Ordinary application clients never select it. Outside
    that trusted context it fails closed rather than obtaining unbounded time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One shared transport serves several roles. Request-specific reasons
        # belong to task context, not a mutable shared "last error" field.
        self._stage_reason = ContextVar("serveflow_task_stage_deadline_reason", default=None)

    def _failure_code(self, error, admission_error, guard):
        reason = current_deadline_failure_reason()
        if reason is None and isinstance(error, ModelDeadlineExceeded):
            reason = error.reason
        if reason is not None:
            self._stage_reason.set(reason)
            return "timeout"
        return super()._failure_code(error, admission_error, guard)

    def _physical_diagnostics(self):
        reason = self._stage_reason.get()
        return {} if reason is None else {"stage_deadline_reason": reason}

    def _validate(self, request, frame):
        # A trusted management scope cannot be used to hide network work. The
        # inherited send callback calls this again immediately before reserve.
        assert_model_active()
        return super()._validate(request, frame)

    @asynccontextmanager
    async def _request_slot(self):
        acquired = False
        try:
            async with overhead("queue"):
                await self._slots.acquire()
                acquired = True
            # The yield/body contains the real network and is deliberately
            # outside the excluded scope.
            yield
        finally:
            if acquired:
                self._slots.release()

    async def _prepare_admission(self):
        async with overhead("admission"):
            return await super()._prepare_admission()

    async def _seal_physical(self, *args, **kwargs):
        async with overhead("seal"):
            return await super()._seal_physical(*args, **kwargs)

    async def _seal_after_error(self, *args, **kwargs):
        # The request is already failed/cancelled. This child only preserves
        # observed response/usage and finishes its occupied claim; it cannot
        # reserve, send or receive extra model time. Calling the parent method
        # directly also avoids borrowing another task's active-clock identity.
        finalizer = asyncio.create_task(
            ThreadedTaskHTTPTransport._seal_physical(self, *args, **kwargs)
        )
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                continue
        return finalizer.result()

    async def handle_async_request(self, request):
        token = self._stage_reason.set(None)
        try:
            return await super().handle_async_request(request)
        except TrialTransportError:
            raise
        except asyncio.CancelledError:
            self._stop("cancelled")
            raise
        except TimeoutError:
            # A management or active-clock failure may originate at the queue
            # hook, before the historical parent's try/physical-send boundary.
            self._stop("timeout")
            raise TrialTransportError("timeout") from None
        except Exception:
            self._stop("protocol")
            raise TrialTransportError("protocol") from None
        finally:
            self._stage_reason.reset(token)
