"""Scaled local-clock tests; no provider/model, store or business API calls."""
import asyncio
import math
import socket
import time

import pytest

from core.model_deadline import (
    EvaluationDeadlinePolicy, ModelDeadlineContextError, ModelDeadlineExceeded,
    assert_model_active, await_model, current_deadline_failure_reason, eval_deadlines, overhead,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("clock tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def policy(limit=.15, grace=.01):
    return EvaluationDeadlinePolicy(overhead_limit_s=limit, cleanup_grace_s=grace)


def run(coro):
    return asyncio.run(coro)


def test_default_without_opt_in_preserves_wait_for_success_and_timeout():
    async def example():
        assert await await_model(asyncio.sleep(.001, result="ok"), timeout_s=.05) == "ok"
        with pytest.raises(TimeoutError):
            await await_model(asyncio.sleep(.1), timeout_s=.01)
    run(example())


def test_production_default_keeps_wait_for_cancel_suppression_semantics():
    async def suppress():
        try:
            await asyncio.sleep(.2)
        except asyncio.CancelledError:
            return "late value"
    assert run(await_model(suppress(), timeout_s=.01)) == "late value"


@pytest.mark.parametrize("bad", [0, -.1, True, None, "8", math.inf, math.nan])
def test_invalid_model_budget_is_rejected_without_executing_coroutine(bad):
    calls = []
    async def body():
        calls.append(1)
    with pytest.raises(ValueError, match="timeout_invalid"):
        run(await_model(body(), timeout_s=bad))
    assert calls == []


@pytest.mark.parametrize("limits", [
    {"overhead_limit_s": 0}, {"overhead_limit_s": 31}, {"overhead_limit_s": True},
    {"overhead_limit_s": math.inf}, {"cleanup_grace_s": -1},
    {"cleanup_grace_s": 1.1}, {"cleanup_grace_s": False},
])
def test_invalid_admin_or_cleanup_budget_is_rejected(limits):
    with pytest.raises(ValueError, match="policy_invalid"):
        EvaluationDeadlinePolicy(**limits)


def test_all_three_management_scopes_are_excluded_but_active_work_remains_bounded():
    async def body():
        for name in ("queue", "admission", "seal"):
            async with overhead(name):
                await asyncio.sleep(.025)
            assert_model_active()
        await asyncio.sleep(.01)
        return "completed"
    async def example():
        with eval_deadlines(policy()):
            started = time.monotonic()
            assert await await_model(body(), timeout_s=.04) == "completed"
            assert time.monotonic() - started >= .075
    run(example())


def test_excluding_metadata_never_extends_active_network_budget():
    async def body():
        async with overhead("admission"):
            await asyncio.sleep(.025)
        assert_model_active()
        await asyncio.sleep(.1)
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(body(), timeout_s=.02)
            assert caught.value.reason == "active" and caught.value.code == "timeout"
    run(example())


def test_two_individually_short_admin_intervals_share_one_cumulative_limit():
    sends = []
    async def body():
        async with overhead("queue"):
            await asyncio.sleep(.03)
        async with overhead("admission"):
            await asyncio.sleep(.03)
        sends.append(1)
    async def example():
        with eval_deadlines(policy(.05)):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(body(), timeout_s=.1)
            assert caught.value.reason == "overhead"
    run(example())
    assert sends == []


def test_nested_scopes_count_outer_elapsed_once_not_twice():
    async def body():
        async with overhead("admission"):
            await asyncio.sleep(.015)
            async with overhead("seal"):
                await asyncio.sleep(.04)
            await asyncio.sleep(.015)
        assert_model_active()
        return True
    async def example():
        with eval_deadlines(policy(.095)):
            assert await await_model(body(), timeout_s=.025)
    run(example())


def test_management_overrun_without_yield_is_not_allowed_to_resume_stage():
    async def body():
        async with overhead("admission"):
            time.sleep(.03)
        assert_model_active()
    async def example():
        with eval_deadlines(policy(.015)):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(body(), timeout_s=.1)
            assert caught.value.reason == "overhead"
    run(example())


def test_a_deadline_past_due_cannot_be_resurrected_by_pausing_before_callback_runs():
    async def body():
        time.sleep(.025)
        async with overhead("admission"):
            raise AssertionError("expired stage entered excluded scope")
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(body(), timeout_s=.01)
            assert caught.value.reason == "active"
    run(example())


def test_network_send_inside_overhead_is_rejected_even_if_caller_swallows_error():
    async def body():
        async with overhead("admission"):
            with pytest.raises(ModelDeadlineContextError):
                assert_model_active()
        return "cannot turn invalid scope into success"
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineContextError):
                await await_model(body(), timeout_s=.1)
    run(example())


@pytest.mark.parametrize("name", ["network", "model", "", None, ["admission"]])
def test_only_fixed_management_scope_names_are_accepted(name):
    async def body():
        async with overhead(name):
            raise AssertionError("unknown scope accepted")
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineContextError):
                await await_model(body(), timeout_s=.1)
    run(example())


def test_missing_policy_or_missing_active_frame_cannot_pause():
    async def attempt():
        async with overhead("admission"):
            raise AssertionError("missing frame accepted")
    with pytest.raises(ModelDeadlineContextError):
        run(attempt())
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineContextError):
                await attempt()
    run(example())


def test_child_task_cannot_pause_inherited_parent_clock():
    async def child():
        with pytest.raises(ModelDeadlineContextError):
            async with overhead("admission"):
                raise AssertionError("child changed parent clock")
    async def body():
        await asyncio.create_task(child())
        assert_model_active()
        return "parent valid"
    async def example():
        with eval_deadlines(policy()):
            assert await await_model(body(), timeout_s=.1) == "parent valid"
    run(example())


def test_worker_thread_cannot_pause_clock_copied_by_to_thread():
    async def attempt():
        async with overhead("admission"):
            raise AssertionError("thread changed clock")
    def worker():
        with pytest.raises(ModelDeadlineContextError):
            asyncio.run(attempt())
    async def body():
        await asyncio.to_thread(worker)
        assert_model_active()
        return True
    async def example():
        with eval_deadlines(policy()):
            assert await await_model(body(), timeout_s=.1)
    run(example())


def test_parallel_model_calls_have_separate_active_and_admin_budgets():
    async def excluded():
        async with overhead("admission"):
            await asyncio.sleep(.06)
        return "first success"
    async def example():
        with eval_deadlines(policy(.1)):
            results = await asyncio.gather(await_model(excluded(), timeout_s=.02),
                await_model(asyncio.sleep(.09), timeout_s=.03), return_exceptions=True)
            assert results[0] == "first success"
            assert isinstance(results[1], ModelDeadlineExceeded) and results[1].reason == "active"
    run(example())


def test_nested_model_stage_in_same_task_is_rejected_without_running_second_call():
    called = []
    async def inner(): called.append(1)
    async def body(): return await await_model(inner(), timeout_s=.1)
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineContextError):
                await await_model(body(), timeout_s=.1)
    run(example())
    assert called == []


def test_stale_evaluation_context_in_delayed_child_cannot_start_later_model():
    release = None
    calls = []
    async def model(): calls.append(1)
    async def child():
        await release.wait()
        await await_model(model(), timeout_s=.1)
    async def example():
        nonlocal release
        release = asyncio.Event()
        with eval_deadlines(policy()):
            task = asyncio.create_task(child())
        release.set()
        with pytest.raises(ModelDeadlineContextError):
            await task
    run(example())
    assert calls == []


def test_already_started_model_cannot_publish_after_its_evaluation_context_closes():
    async def example():
        entered, release = asyncio.Event(), asyncio.Event()
        async def model():
            entered.set()
            await release.wait()
            return "late result"
        with eval_deadlines(policy()):
            task = asyncio.create_task(await_model(model(), timeout_s=.3))
            await entered.wait()
        release.set()
        with pytest.raises(ModelDeadlineContextError):
            await task
    run(example())


def test_external_cancellation_keeps_original_message_and_does_not_become_timeout():
    async def example():
        entered = asyncio.Event()
        async def model():
            entered.set()
            await asyncio.Event().wait()
        with eval_deadlines(policy()):
            task = asyncio.create_task(await_model(model(), timeout_s=.5))
            await entered.wait()
            task.cancel("user_cancel")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("user_cancel",)
    run(example())


def test_repeated_external_cancel_during_admin_cleanup_never_resumes_a_send():
    async def example():
        entered, cleanup_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sends = []
        async def finish():
            cleanup_entered.set()
            await release.wait()
        cleanup = None
        async def model():
            nonlocal cleanup
            async with overhead("admission"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup = asyncio.create_task(finish())
                    await asyncio.shield(cleanup)
            sends.append(1)
        with eval_deadlines(policy(.3)):
            task = asyncio.create_task(await_model(model(), timeout_s=.2))
            await entered.wait()
            task.cancel("first")
            await cleanup_entered.wait()
            task.cancel("second")
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await cleanup
            assert sends == []
    run(example())


def test_swallowed_active_timeout_is_still_failure_in_evaluation():
    async def suppress():
        try: await asyncio.sleep(.2)
        except asyncio.CancelledError: return "late"
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(suppress(), timeout_s=.01)
            assert caught.value.reason == "active"
    run(example())


def test_independent_wall_timer_still_cancels_when_active_cancel_is_swallowed():
    async def example():
        cancellations = []
        async def model():
            while True:
                try: await asyncio.sleep(.5)
                except asyncio.CancelledError:
                    cancellations.append(time.monotonic())
                    if len(cancellations) == 2:
                        raise
        with eval_deadlines(policy(.035, .005)):
            started = time.monotonic()
            with pytest.raises(ModelDeadlineExceeded):
                await await_model(model(), timeout_s=.02)
            assert len(cancellations) == 2
            assert .05 <= time.monotonic() - started < .3
    run(example())


def test_external_cancel_swallowed_inside_admin_cannot_return_success():
    async def example():
        entered = asyncio.Event()
        async def model():
            try:
                async with overhead("admission"):
                    entered.set()
                    await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "not a successful request"
        with eval_deadlines(policy()):
            task = asyncio.create_task(await_model(model(), timeout_s=.3))
            await entered.wait()
            task.cancel("external")
            with pytest.raises(asyncio.CancelledError): await task
    run(example())


def test_nested_evaluation_context_cannot_replace_policy():
    with eval_deadlines(policy()):
        with pytest.raises(ModelDeadlineContextError):
            with eval_deadlines(policy(.01)):
                raise AssertionError("nested policy accepted")


def test_transport_observes_active_expiry_before_timeout_context_exits():
    observed = []
    async def model():
        assert current_deadline_failure_reason() is None
        try:
            await asyncio.sleep(.2)
        except asyncio.CancelledError:
            observed.append(current_deadline_failure_reason())
            raise RuntimeError("controlled transport converts cancellation")
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(model(), timeout_s=.01)
            assert caught.value.reason == "active"
        assert current_deadline_failure_reason() is None
    run(example())
    assert observed == ["active"]


def test_transport_observes_overhead_expiry_and_never_a_network_timeout():
    observed = []
    async def model():
        async with overhead("admission"):
            try:
                await asyncio.sleep(.2)
            except asyncio.CancelledError:
                observed.append(current_deadline_failure_reason())
                raise
    async def example():
        with eval_deadlines(policy(.015)):
            with pytest.raises(ModelDeadlineExceeded) as caught:
                await await_model(model(), timeout_s=.1)
            assert caught.value.reason == "overhead"
    run(example())
    assert observed == ["overhead"]


def test_external_cancel_has_priority_if_an_active_timer_also_fired():
    observed = []
    async def model():
        try:
            await asyncio.sleep(.2)
        except asyncio.CancelledError:
            asyncio.current_task().cancel("additional external cancellation")
            observed.append(current_deadline_failure_reason())
            raise
    async def example():
        with eval_deadlines(policy()):
            task = asyncio.create_task(await_model(model(), timeout_s=.01))
            with pytest.raises(asyncio.CancelledError):
                await task
    run(example())
    assert observed == [None]


def test_child_never_observes_parent_expiry_reason():
    observed = []
    async def child():
        observed.append(current_deadline_failure_reason())
    async def model():
        try:
            await asyncio.sleep(.2)
        except asyncio.CancelledError:
            await asyncio.create_task(child())
            raise
    async def example():
        with eval_deadlines(policy()):
            with pytest.raises(ModelDeadlineExceeded):
                await await_model(model(), timeout_s=.01)
    run(example())
    assert observed == [None]
