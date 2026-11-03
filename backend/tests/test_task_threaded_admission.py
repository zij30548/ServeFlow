"""Read-only threaded admission with real local ledger and controlled HTTP.

No live models/services are contacted. Shortened test-only guard deadlines
exercise ordering; production HTTP_LIMITS and the default 8s guard stay intact.
"""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
import socket
import threading
import time

import httpx
import pytest

from evaluation import benchmark as bench, task_transport as module
from evaluation.intent_transport import TrialTransportError, canonical_payload_hash
from evaluation.task_budget import TaskBudget, TaskBudgetError
from evaluation.task_transport import (
    CURRENT, Frame, TaskHTTPTransport, ThreadedTaskHTTPTransport, TIMING_FIELDS, context,
)

BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PLAN = [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]} for i in range(1, 7)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("threaded admission tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def request(role="general"):
    tokens, temperature = module.AGENT_PARAMETERS[role]
    return httpx.Request("POST", BASE + "/v1/messages", json={
        "model": "qwen3.7-flash-2026-07-15", "max_tokens": tokens, "temperature": temperature,
        "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "controlled input"}]})


def response():
    return httpx.Response(200, json={"content": [{"type": "text", "text": "controlled known response"}],
        "model": "controlled-upstream", "stop_reason": "end_turn", "usage": {"input_tokens": 7, "output_tokens": 3}})


@contextmanager
def harness(tmp_path, factory, verify=lambda: None, *, transport_type=ThreadedTaskHTTPTransport):
    root = tmp_path / "batch"
    root.mkdir(mode=0o700)
    budget = TaskBudget(root / "ledger", "a" * 64, PLAN, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        yield root, budget, transport_type(BASE, root, budget, lease, verify, delegate_factory=factory)


async def send(transport, role="general", req=None, *, guard=None):
    with context(Frame("c1", "t1", "agent", role, evidence_guard=guard)):
        return await transport.handle_async_request(req or request(role))


async def wait_thread(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(.001)


def saved(root, number=1):
    return bench.read_json(root / f"physical-{number:03d}.json")


def test_default_transport_stays_inline_and_does_not_add_timing(tmp_path):
    main, observed = threading.get_ident(), []
    async def handle(req):
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle),
                 lambda: observed.append(threading.get_ident()), transport_type=TaskHTTPTransport) as (root, budget, transport):
        assert asyncio.run(send(transport)).status_code == 200
        assert observed == [main] and budget.status()["attempts"] == 1
        assert "timing" not in saved(root)


def test_slow_full_verification_does_not_block_inflight_network_and_flock_is_serialized(tmp_path):
    entered, release = threading.Event(), threading.Event()
    network_entered, network_release, network_read = asyncio.Event(), asyncio.Event(), asyncio.Event()
    main, verifier_threads, reserve_threads, finish_threads, owner = threading.get_ident(), [], [], [], {}
    async def handle(req):
        if CURRENT.get().agent == "general":
            network_entered.set()
            await network_release.wait()
            network_read.set()
        return response()
    def verify():
        verifier_threads.append(threading.get_ident())
        assert owner["manifest"].read_text() == "complete frozen evidence"
        # Deliberately hold the real nonblocking file lock to make any unsafe
        # concurrent reserve/finish deterministically fail instead of racing.
        with owner["budget"]._locked():
            if CURRENT.get().agent == "billing":
                entered.set()
                assert release.wait(2)
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (root, budget, transport):
        manifest = tmp_path / "manifest"; manifest.write_text("complete frozen evidence")
        owner.update(budget=budget, manifest=manifest)
        original_reserve, original_finish = budget.reserve, budget.finish
        def reserve(*args, **kwargs):
            reserve_threads.append(threading.get_ident()); return original_reserve(*args, **kwargs)
        def finish(*args, **kwargs):
            finish_threads.append(threading.get_ident()); return original_finish(*args, **kwargs)
        budget.reserve, budget.finish = reserve, finish
        async def run():
            first = asyncio.create_task(send(transport))
            await network_entered.wait()
            second = asyncio.create_task(send(transport, "billing"))
            try:
                await wait_thread(entered)
                network_release.set()
                # Verification is still blocked in a worker, but the existing
                # network response is read on-loop. Only sealing waits safely.
                await asyncio.wait_for(network_read.wait(), .2)
                assert not release.is_set() and not first.done()
            finally:
                release.set()
            assert [r.status_code for r in await asyncio.gather(first, second)] == [200, 200]
        asyncio.run(run())
        assert len(verifier_threads) == 2 and all(value != main for value in verifier_threads)
        assert reserve_threads == finish_threads == [main, main]
        assert budget.status()["attempts"] == 2 and not budget.status()["pending"]
        assert transport.fault is None
        assert saved(root)["outcome"] == saved(root, 2)["outcome"] == "received"


def test_cancelled_admission_reaps_readonly_worker_never_reserves_or_sends(tmp_path):
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls, owner = [], {}
    def verify():
        assert owner["manifest"].read_text() == "frozen"
        entered.set()
        try:
            assert release.wait(2)
            assert owner["manifest"].read_text() == "frozen"
        finally:
            completed.set()
    async def handle(req):
        calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (root, budget, transport):
        manifest = tmp_path / "manifest"; manifest.write_text("frozen"); owner["manifest"] = manifest
        async def run():
            task = asyncio.create_task(send(transport))
            try:
                await wait_thread(entered)
                task.cancel(); await asyncio.sleep(.005); task.cancel()
                assert not completed.is_set() and not task.done()
                assert budget.status()["attempts"] == 0 and calls == []
            finally:
                release.set()
            with pytest.raises(TrialTransportError, match="cancelled"):
                await task
        asyncio.run(run())
        assert completed.is_set() and calls == [] and budget.status()["attempts"] == 0
        assert manifest.read_text() == "frozen" and not list(root.glob("physical-*.json"))
        assert transport.fault == "cancelled"


def test_failed_version_validation_never_creates_delegate_reserves_or_sends(tmp_path):
    factories = []
    def verify():
        raise TrialTransportError("protocol")
    def factory(frame):
        factories.append(frame); return httpx.MockTransport(lambda req: response())
    with harness(tmp_path, factory, verify) as (root, budget, transport):
        with pytest.raises(TrialTransportError, match="protocol"):
            asyncio.run(send(transport))
        assert factories == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json")) and transport.fault == "protocol"


def test_equivalent_json_wire_mutation_during_thread_wait_is_rejected_before_reservation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def verify():
        entered.set(); assert release.wait(2)
    async def handle(req):
        calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (root, budget, transport):
        req = request(); original = req.content
        async def run():
            task = asyncio.create_task(send(transport, req=req))
            try:
                await wait_thread(entered)
                req._content = json.dumps(json.loads(original), indent=2).encode()
                assert canonical_payload_hash(json.loads(req.content)) == canonical_payload_hash(json.loads(original))
                assert req.content != original
            finally:
                release.set()
            with pytest.raises(TrialTransportError, match="protocol"):
                await task
        asyncio.run(run())
        assert calls == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


def test_evidence_guard_is_rechecked_on_loop_after_complete_thread_validation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    state, calls = {"stale": False}, []
    def verify():
        entered.set(); assert release.wait(2)
    def guard():
        if state["stale"]:
            raise TrialTransportError("protocol")
    async def handle(req):
        calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (_, budget, transport):
        async def run():
            task = asyncio.create_task(send(transport, guard=guard))
            try:
                await wait_thread(entered); state["stale"] = True
            finally:
                release.set()
            with pytest.raises(TrialTransportError):
                await task
        asyncio.run(run())
        assert calls == [] and budget.status()["attempts"] == 0


def test_concurrent_reservation_keeps_the_real_ledger_limit_atomic_on_loop(tmp_path):
    calls, threads = [], []
    async def handle(req):
        calls.append(req); await asyncio.sleep(.02); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_, budget, transport):
        original = budget.reserve
        def limited(*args, **kwargs):
            threads.append(threading.get_ident())
            # A smaller controlled threshold exercises contention without
            # filling 168 metadata entries or changing production quota units.
            if budget.status()["attempts"] >= 1:
                raise TaskBudgetError("quota")
            return original(*args, **kwargs)
        budget.reserve = limited
        async def run():
            return await asyncio.gather(send(transport), send(transport, "billing"), return_exceptions=True)
        outputs = asyncio.run(run())
        assert sum(isinstance(v, httpx.Response) for v in outputs) == 1
        assert sum(isinstance(v, TrialTransportError) and v.code == "quota" for v in outputs) == 1
        assert len(calls) == budget.status()["attempts"] == 1
        assert threads == [threading.get_ident()] * 2
        assert not budget.status()["pending"]


def test_network_deadline_starts_after_admission_not_before_thread_wait(tmp_path):
    guards, calls = [], []
    def verify():
        time.sleep(.08)
    async def handle(req):
        calls.append(req); await asyncio.sleep(.01); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (root, budget, transport):
        def scaled_guard(*args, **kwargs):
            guard = module._ObservedIntentGuard(*args, timeout_s=.04, **kwargs)
            guards.append(guard); return guard
        transport._guard_type = scaled_guard
        assert asyncio.run(send(transport)).status_code == 200
        record = saved(root)
        assert guards[0]._timeout_s == .04 and module.HTTP_LIMITS["request_deadline_s"] == 8
        assert record["timing"]["admission_ms"] >= 70 and record["timing"]["network_ms"] < 40
        assert record["timing"]["wall_ms"] >= 80 and len(calls) == budget.status()["attempts"] == 1


def test_network_timeout_keeps_sent_attempt_and_never_retries(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req); await asyncio.Future()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        transport._guard_type = lambda *args, **kwargs: module._ObservedIntentGuard(*args, timeout_s=.025, **kwargs)
        with pytest.raises(TrialTransportError, match="timeout"):
            asyncio.run(send(transport))
        record = saved(root)
        assert len(calls) == budget.status()["attempts"] == 1 and not budget.status()["pending"]
        assert record["delegate_entered"] and record["http_status"] is None
        assert record["error_code"] == "timeout" and record["outcome"] == "execution_error"
        assert record["timing"]["network_ms"] >= 20
        assert transport.fault == "timeout" and module.HTTP_LIMITS["request_deadline_s"] == 8


def test_queue_cancellation_is_sticky_with_no_quota_or_delegate(tmp_path):
    calls, verifications = [], []
    async def handle(req):
        calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), lambda: verifications.append(True)) as (root, budget, transport):
        async def run():
            for _ in range(3): await transport._slots.acquire()
            task = asyncio.create_task(send(transport))
            try:
                await asyncio.sleep(.005); task.cancel()
                with pytest.raises(asyncio.CancelledError): await task
            finally:
                for _ in range(3): transport._slots.release()
            with pytest.raises(TrialTransportError): await send(transport)
        asyncio.run(run())
        assert calls == verifications == [] and budget.status()["attempts"] == 0
        assert transport.fault == "cancelled" and not list(root.glob("physical-*.json"))


def test_timing_is_bounded_numeric_queue_is_not_network_and_default_guard_remains_eight(tmp_path):
    guards = []
    class ClosingTransport(httpx.MockTransport):
        async def aclose(self):
            await asyncio.sleep(.01); await super().aclose()
    async def handle(req):
        await asyncio.sleep(.01); return response()
    with harness(tmp_path, lambda frame: ClosingTransport(handle)) as (root, _, transport):
        def observed_guard(*args, **kwargs):
            guard = module._ObservedIntentGuard(*args, **kwargs); guards.append(guard); return guard
        transport._guard_type = observed_guard
        async def run():
            for _ in range(3): await transport._slots.acquire()
            task = asyncio.create_task(send(transport))
            await asyncio.sleep(.02); transport._slots.release()
            try:
                assert (await task).status_code == 200
            finally:
                for _ in range(2): transport._slots.release()
        asyncio.run(run())
        timing = saved(root)["timing"]
        assert set(timing) == TIMING_FIELDS
        assert all(type(v) is float and math.isfinite(v) and v >= 0 for v in timing.values())
        assert timing["queue_ms"] >= 15 and timing["cleanup_ms"] >= 8 and timing["network_ms"] >= 8
        assert sum(timing[key] for key in TIMING_FIELDS if key != "wall_ms") <= timing["wall_ms"] + 3
        assert guards[0]._timeout_s == 8 and saved(root)["response"]["usage"]["input_tokens"] == 7


def test_known_response_is_preserved_if_cancelled_while_waiting_to_seal(tmp_path):
    entered, release = threading.Event(), threading.Event()
    network_entered, network_release, cleanup_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner, calls, raw_responses = {}, [], []
    def verify():
        with owner["budget"]._locked():
            if CURRENT.get().agent == "billing":
                entered.set(); assert release.wait(2)
    async def handle(req):
        calls.append(CURRENT.get().agent)
        network_entered.set()
        await network_release.wait()
        value = response(); raw_responses.append(value.content)
        return value
    class NotifyingGuard(module._ObservedIntentGuard):
        async def _cleanup(self, value):
            try:
                return await super()._cleanup(value)
            finally:
                cleanup_done.set()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify) as (root, budget, transport):
        owner["budget"] = budget
        transport._guard_type = NotifyingGuard
        async def run():
            first = asyncio.create_task(send(transport))
            await network_entered.wait()
            second = asyncio.create_task(send(transport, "billing"))
            try:
                await wait_thread(entered)
                network_release.set()
                # On cleanup completion the coroutine proceeds synchronously
                # through parse until its asynchronous seal-gate wait.
                await cleanup_done.wait()
                assert not first.done()
                first.cancel(); await asyncio.sleep(.005); first.cancel()
                assert not first.done()  # canceled finalization is protected
            finally:
                release.set()
            values = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(v, TrialTransportError) for v in values)
            assert values[0].code == "cancelled"
        asyncio.run(run())
        assert calls == ["general"] and budget.status()["attempts"] == 1
        assert not budget.status()["pending"]
        assert len(list(root.glob("physical-*.json"))) == 1
        record = saved(root)
        assert record["outcome"] == record["error_code"] == "cancelled"
        assert record["http_status"] == 200 and record["delegate_entered"]
        assert record["response"]["content"] == [{"type": "text", "text": "controlled known response"}]
        assert record["response_sha256"] == hashlib.sha256(raw_responses[0]).hexdigest()
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert transport.fault == "cancelled"


@pytest.mark.parametrize("direct_async", [False, True])
def test_async_verifier_cannot_silently_skip_full_readonly_validation(tmp_path, direct_async):
    calls = []
    async def async_verifier():
        calls.append("must not execute asynchronously")
    verifier = async_verifier if direct_async else lambda: async_verifier()
    async def handle(req):
        calls.append("must not send"); return response()
    if direct_async:
        with pytest.raises(TrialTransportError, match="protocol"):
            with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verifier):
                pytest.fail("async verifier must be rejected before use")
    else:
        with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verifier) as (_, budget, transport):
            with pytest.raises(TrialTransportError, match="protocol"):
                asyncio.run(send(transport))
            assert budget.status()["attempts"] == 0
    assert calls == []
