"""Model-stage deadlines with an explicit, server-only evaluation clock.

Ordinary calls use the original ``asyncio.wait_for`` semantics. A new evaluation
runner may explicitly opt in to a separate clock: full local verification,
request-slot waiting and durable sealing do not spend the model-stage budget.
They have a separate cumulative limit and an unpausable wall cancellation timer.
The HTTP guard's network deadline is independent and must never be paused.

These are cooperative cancellation deadlines, not process/thread kill timers.
An uncooperative awaitable or an existing read-only thread that must be reaped
can delay return after cancellation; the process supervisor remains the hard
containment boundary. This module creates no clients and performs no I/O.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import math
import threading
from typing import Any, Awaitable, TypeVar


T = TypeVar("T")
SCOPES = frozenset({"queue", "admission", "seal"})
MAX_SCOPE_ENTRIES = 64


class ModelDeadlineExceeded(TimeoutError):
    """Fixed safe reason; no request, response, credentials or exception input."""

    code = "timeout"

    def __init__(self, reason: str):
        if reason not in {"active", "overhead", "wall"}:
            raise ValueError("model_deadline_reason_invalid")
        self.reason = reason
        super().__init__("model_deadline_" + reason)


class ModelDeadlineContextError(RuntimeError):
    code = "protocol"

    def __init__(self):
        super().__init__("model_deadline_context_invalid")


def _number(value, *, positive=True):
    return (type(value) in {int, float} and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


@dataclass(frozen=True)
class EvaluationDeadlinePolicy:
    overhead_limit_s: float = 30.0
    cleanup_grace_s: float = 0.25

    def __post_init__(self):
        if (not _number(self.overhead_limit_s) or self.overhead_limit_s > 30
                or not _number(self.cleanup_grace_s, positive=False) or self.cleanup_grace_s > 1):
            raise ValueError("evaluation_deadline_policy_invalid")


@dataclass
class _Session:
    policy: EvaluationDeadlinePolicy
    thread_id: int = field(default_factory=threading.get_ident)
    active: bool = True
    loop: Any = None


@dataclass
class _Clock:
    session: _Session
    task: asyncio.Task
    loop: asyncio.AbstractEventLoop
    active_timer: asyncio.Timeout
    wall_timer: asyncio.Timeout
    baseline_cancels: int
    alive: bool = True
    used_overhead_s: float = 0.0
    stack: list[str] = field(default_factory=list)
    administrative_timers: list[asyncio.Timeout] = field(default_factory=list)
    exited_timers: set[int] = field(default_factory=set)
    aborted_overhead: bool = False
    invalid: bool = False
    entries: int = 0


_SESSION: ContextVar[_Session | None] = ContextVar("serveflow_evaluation_deadline_session", default=None)
_CLOCK: ContextVar[_Clock | None] = ContextVar("serveflow_model_stage_deadline", default=None)


def _dispose(awaitable):
    # Invalid admission must not leave an unawaited SDK coroutine behind. Do not
    # execute or cancel someone else's Future/Task as part of validation.
    if inspect.iscoroutine(awaitable):
        awaitable.close()


@contextmanager
def eval_deadlines(policy: EvaluationDeadlinePolicy):
    """Opt in from trusted runner code, never a request/model parameter.

    Child tasks may each start an independent model call. They cannot pause a
    parent's clock, and copies that outlive this context become stale.
    """
    previous = _SESSION.get()
    if type(policy) is not EvaluationDeadlinePolicy or previous is not None and previous.active:
        raise ModelDeadlineContextError()
    session = _Session(policy)
    token = _SESSION.set(session)
    try:
        yield policy
    finally:
        session.active = False
        _SESSION.reset(token)


def _owned_clock() -> _Clock:
    clock = _CLOCK.get()
    try:
        task, loop = asyncio.current_task(), asyncio.get_running_loop()
    except RuntimeError:
        raise ModelDeadlineContextError() from None
    if (clock is None or not clock.alive or not clock.session.active
            or _SESSION.get() is not clock.session
            or threading.get_ident() != clock.session.thread_id
            or clock.task is not task or clock.loop is not loop):
        raise ModelDeadlineContextError()
    return clock


def _expired_reason(clock):
    timers = [(clock.active_timer, "active"), (clock.wall_timer, "wall")]
    timers += [(timer, "overhead") for timer in clock.administrative_timers]
    expired = [(timer.when(), reason) for timer, reason in timers if timer.expired()]
    return min(expired, key=lambda item: item[0])[1] if expired else None


@asynccontextmanager
async def _tracked_timer(clock, timer):
    try:
        async with timer:
            yield
    finally:
        # asyncio.Timeout removes its own cancellation once its __aexit__ has
        # completed, even when it raises TimeoutError. This marker lets the
        # transport distinguish those past timers from pending own cancels.
        clock.exited_timers.add(id(timer))


def current_deadline_failure_reason():
    """Safe same-task observation for the actual transport failure boundary.

    An independent/external cancellation takes priority over an own expired
    timer. No active frame, stale frame or a child/thread gets a reason. This
    function changes no deadline and authorizes no request or retry.
    """
    try:
        clock = _owned_clock()
    except ModelDeadlineContextError:
        return None
    timers = [clock.active_timer, clock.wall_timer, *clock.administrative_timers]
    pending_own = sum(timer.expired() and id(timer) not in clock.exited_timers for timer in timers)
    if clock.task.cancelling() - clock.baseline_cancels > pending_own:
        return None
    return _expired_reason(clock)


def assert_model_active():
    """A new evaluation transport must call this before reserve/delegate entry.

    This prevents a mistakenly broad overhead scope from hiding actual network
    waiting. It authorizes no tool, URL or model and creates no quota claim.
    """
    clock = _owned_clock()
    if clock.stack or clock.invalid or clock.aborted_overhead:
        clock.invalid = True
        raise ModelDeadlineContextError()
    reason = _expired_reason(clock)
    if reason:
        raise ModelDeadlineExceeded(reason)
    when = clock.active_timer.when()
    if when is None or when <= clock.loop.time():
        raise ModelDeadlineExceeded("active")


@asynccontextmanager
async def overhead(scope: str):
    """Exclude one bounded management interval from the current stage only.

    Nested allowed scopes consume the outer interval once. Only the owner task
    can enter/leave; thread and stale/inherited-clock operations fail closed.
    Cancellation or a failed interval never re-enables the stage for a send.
    """
    clock = _owned_clock()
    if type(scope) is not str or scope not in SCOPES or clock.invalid or clock.aborted_overhead:
        clock.invalid = True
        raise ModelDeadlineContextError()
    clock.entries += 1
    if clock.entries > MAX_SCOPE_ENTRIES:
        clock.invalid = True
        raise ModelDeadlineContextError()
    if clock.stack:
        clock.stack.append(scope)
        try:
            yield
        finally:
            clock.stack.pop()
        return
    reason = _expired_reason(clock)
    if reason:
        raise ModelDeadlineExceeded(reason)
    when = clock.active_timer.when()
    remaining = None if when is None else when - clock.loop.time()
    if remaining is None or remaining <= 0:
        raise ModelDeadlineExceeded("active")
    admin_remaining = clock.session.policy.overhead_limit_s - clock.used_overhead_s
    if admin_remaining <= 0:
        raise ModelDeadlineExceeded("overhead")
    timer = asyncio.timeout(admin_remaining)
    clock.administrative_timers.append(timer)
    clock.stack.append(scope)
    clock.active_timer.reschedule(None)
    started = clock.loop.time()
    completed = False
    try:
        try:
            async with _tracked_timer(clock, timer):
                yield
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if timer.expired():
                raise ModelDeadlineExceeded("overhead") from None
            raise
        if timer.expired() or clock.used_overhead_s + clock.loop.time() - started >= clock.session.policy.overhead_limit_s:
            raise ModelDeadlineExceeded("overhead")
        completed = True
    finally:
        clock.used_overhead_s += max(0.0, clock.loop.time() - started)
        clock.stack.pop()
        if completed and clock.alive and clock.session.active:
            # Never resurrect a deadline already cancelled by the independent
            # wall clock or an external cancellation while finishing the block.
            if not clock.wall_timer.expired() and clock.task.cancelling() <= clock.baseline_cancels:
                clock.active_timer.reschedule(clock.loop.time() + remaining)
            else:
                clock.aborted_overhead = True
        else:
            clock.aborted_overhead = True


async def await_model(awaitable: Awaitable[T], timeout_s: float) -> T:
    """Await one existing SDK coroutine with unchanged production defaults."""
    if not _number(timeout_s):
        _dispose(awaitable)
        raise ValueError("model_deadline_timeout_invalid")
    session = _SESSION.get()
    if session is None:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    task, loop = asyncio.current_task(), asyncio.get_running_loop()
    parent_clock = _CLOCK.get()
    if (not session.active or session.thread_id != threading.get_ident()
            or session.loop is not None and session.loop is not loop
            or parent_clock is not None and parent_clock.alive and parent_clock.task is task):
        _dispose(awaitable)
        raise ModelDeadlineContextError()
    session.loop = loop
    active_timer = asyncio.timeout(timeout_s)
    wall_timer = asyncio.timeout(timeout_s + session.policy.overhead_limit_s + session.policy.cleanup_grace_s)
    clock = _Clock(session, task, loop, active_timer, wall_timer, task.cancelling())
    token = _CLOCK.set(clock)
    try:
        try:
            async with _tracked_timer(clock, wall_timer):
                async with _tracked_timer(clock, active_timer):
                    result = await awaitable
        except asyncio.CancelledError:
            raise
        except BaseException:
            # A transport can convert its received cancellation into a typed
            # error. Preserve an external cancel; otherwise a locally expired
            # deadline cannot be turned into an ordinary successful response.
            if task.cancelling() > clock.baseline_cancels:
                raise asyncio.CancelledError from None
            reason = _expired_reason(clock)
            if reason:
                raise ModelDeadlineExceeded(reason) from None
            raise
        if task.cancelling() > clock.baseline_cancels:
            raise asyncio.CancelledError
        reason = _expired_reason(clock)
        if reason:
            raise ModelDeadlineExceeded(reason)
        if not session.active or clock.invalid or clock.aborted_overhead or clock.stack:
            raise ModelDeadlineContextError()
        return result
    finally:
        clock.alive = False
        _CLOCK.reset(token)
