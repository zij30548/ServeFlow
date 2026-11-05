"""Deadline separation with actual local accounting and controlled HTTP only.

The short budgets below are deliberate time-scaled checks, not production
deadlines or measured model/provider performance. No service is contacted.
"""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import socket
import threading
import time

import httpx
import pytest

from core.model_deadline import (
    EvaluationDeadlinePolicy, ModelDeadlineExceeded, await_model,
    eval_deadlines, overhead,
)
from evaluation import benchmark as bench, task_transport as original
from evaluation.intent_transport import TrialTransportError, canonical_payload_hash
from evaluation.task_budget import TaskBudget
from evaluation.task_deadline_transport import DeadlineTaskHTTPTransport
from evaluation.task_transport import Frame, context


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PLAN = [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
        for i in range(1, 7)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("deadline transport checks must not contact a service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def request():
    return httpx.Request("POST", BASE + "/v1/messages", json={
        "model": "qwen3.7-flash-2026-07-15", "max_tokens": 900,
        "temperature": .3, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "controlled input"}],
    })


def response():
    return httpx.Response(200, json={
        "content": [{"type": "text", "text": "controlled known result"}],
        "model": "controlled-upstream", "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    })


@contextmanager
def harness(tmp_path, handle, verify=lambda: None):
    root = tmp_path / "batch"
    root.mkdir(mode=0o700)
    budget = TaskBudget(root / "ledger", "a" * 64, PLAN, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        transport = DeadlineTaskHTTPTransport(BASE, root, budget, lease, verify,
            delegate_factory=lambda frame: httpx.MockTransport(handle))
        yield root, budget, transport


async def send(transport, *, timeout=.05, req=None, evidence_guard=None):
    with context(Frame("c1", "t1", "agent", "general", evidence_guard=evidence_guard)):
        return await await_model(transport.handle_async_request(req or request()), timeout_s=timeout)


def policy(limit=.5):
    return EvaluationDeadlinePolicy(overhead_limit_s=limit, cleanup_grace_s=.02)


def saved(root):
    return bench.read_json(root / "physical-001.json")


async def wait_thread(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(.001)


def test_without_explicit_eval_context_fails_before_quota_or_send(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        with pytest.raises(TrialTransportError, match="protocol"):
            asyncio.run(send(transport))
        assert calls == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json")) and transport.fault == "protocol"


def test_slow_complete_admission_does_not_consume_active_stage_budget(tmp_path):
    verifications, calls = [], []
    def verify():
        verifications.append(threading.get_ident())
        time.sleep(.07)
    async def handle(req):
        calls.append(req)
        await asyncio.sleep(.015)
        return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                return await send(transport, timeout=.04)
        started = time.monotonic()
        assert asyncio.run(run()).status_code == 200
        assert time.monotonic() - started >= .07
        assert verifications and verifications[0] != threading.get_ident()
        assert len(calls) == budget.status()["attempts"] == 1
        assert saved(root)["timing"]["admission_ms"] >= 60
        assert saved(root)["outcome"] == "received"
        assert original.HTTP_LIMITS["request_deadline_s"] == 8


def test_slot_wait_is_excluded_but_the_network_body_is_not(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req)
        await asyncio.sleep(.012)
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            for _ in range(3):
                await transport._slots.acquire()
            with eval_deadlines(policy()):
                task = asyncio.create_task(send(transport, timeout=.04))
                await asyncio.sleep(.07)
                transport._slots.release()
                try:
                    return await task
                finally:
                    for _ in range(2):
                        transport._slots.release()
        assert asyncio.run(run()).status_code == 200
        assert saved(root)["timing"]["queue_ms"] >= 60
        assert len(calls) == budget.status()["attempts"] == 1


def test_sealing_wait_is_excluded_after_a_completed_network_response(tmp_path):
    network_entered, network_release = asyncio.Event(), asyncio.Event()
    calls = []
    async def handle(req):
        calls.append(req)
        network_entered.set()
        await network_release.wait()
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                task = asyncio.create_task(send(transport, timeout=.04))
                await network_entered.wait()
                await transport._metadata_gate.acquire()
                try:
                    network_release.set()
                    await asyncio.sleep(.07)
                    assert not task.done()
                finally:
                    transport._metadata_gate.release()
                return await task
        assert asyncio.run(run()).status_code == 200
        record = saved(root)
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert record["timing"]["wall_ms"] >= 60
        assert len(calls) == budget.status()["attempts"] == 1


def test_parallel_model_frames_keep_independent_budgets_and_request_diagnostics(tmp_path):
    first_entered = asyncio.Event()
    calls, verifies = [], []
    def verify():
        verifies.append(True)
        if len(verifies) == 2:
            time.sleep(.11)
    async def handle(req):
        calls.append(req)
        if len(calls) == 1:
            first_entered.set()
            await asyncio.sleep(.055)
        else:
            await asyncio.sleep(.005)
        return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                first = asyncio.create_task(send(transport, timeout=.09))
                await first_entered.wait()
                second = asyncio.create_task(send(transport, timeout=.035))
                return await asyncio.gather(first, second)
        assert [v.status_code for v in asyncio.run(run())] == [200, 200]
        assert len(calls) == budget.status()["attempts"] == 2
        assert not budget.status()["pending"] and transport.fault is None
        records = [bench.read_json(p) for p in root.glob("physical-*.json")]
        assert len(records) == 2 and all(p["outcome"] == "received" for p in records)
        assert all("stage_deadline_reason" not in p for p in records)


def test_network_still_consumes_the_active_model_deadline_and_keeps_its_claim(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req)
        await asyncio.sleep(.12)
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                await send(transport, timeout=.03)
        with pytest.raises(ModelDeadlineExceeded):
            asyncio.run(run())
        assert len(calls) == budget.status()["attempts"] == 1
        assert saved(root)["delegate_entered"]
        assert saved(root)["outcome"] == "execution_error"
        assert saved(root)["stage_deadline_reason"] == "active"
        assert transport.fault == "timeout" and not budget.status()["pending"]


def test_independent_network_guard_is_not_extended_by_the_paused_stage_clock(tmp_path):
    calls, guards = [], []
    async def handle(req):
        calls.append(req)
        await asyncio.Future()
    with harness(tmp_path, handle, lambda: time.sleep(.06)) as (root, budget, transport):
        def guard(*args, **kwargs):
            value = original._ObservedIntentGuard(*args, timeout_s=.025, **kwargs)
            guards.append(value)
            return value
        transport._guard_type = guard
        async def run():
            with eval_deadlines(policy()):
                await send(transport, timeout=.08)
        with pytest.raises(TrialTransportError, match="timeout"):
            asyncio.run(run())
        assert guards[0]._timeout_s == .025 and original.HTTP_LIMITS["request_deadline_s"] == 8
        assert len(calls) == budget.status()["attempts"] == 1
        assert saved(root)["error_code"] == "timeout" and saved(root)["delegate_entered"]


def test_management_budget_is_bounded_and_expiration_never_sends_after_reader_finishes(tmp_path):
    calls, complete = [], threading.Event()
    def verify():
        try:
            time.sleep(.075)
        finally:
            complete.set()
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy(.02)):
                await send(transport, timeout=.04)
        with pytest.raises(ModelDeadlineExceeded):
            asyncio.run(run())
        assert complete.is_set() and calls == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


def test_seal_management_limit_preserves_received_response_as_failed_not_success(tmp_path):
    entered, network_release = asyncio.Event(), asyncio.Event()
    calls = []
    async def handle(req):
        calls.append(req)
        entered.set()
        await network_release.wait()
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy(.04)):
                task = asyncio.create_task(send(transport, timeout=.2))
                await entered.wait()
                await transport._metadata_gate.acquire()
                try:
                    network_release.set()
                    await asyncio.sleep(.07)
                finally:
                    transport._metadata_gate.release()
                with pytest.raises(ModelDeadlineExceeded) as caught:
                    await task
                assert caught.value.reason == "overhead"
        asyncio.run(run())
        record = saved(root)
        assert len(calls) == budget.status()["attempts"] == 1
        assert record["http_status"] == 200 and record["outcome"] == "execution_error"
        assert record["error_code"] == "timeout" and record["stage_deadline_reason"] == "overhead"
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert record["response"]["content"] == [{"type": "text", "text": "controlled known result"}]
        assert not budget.status()["pending"]


def test_external_repeated_cancel_reaps_admission_without_reserving_or_sending(tmp_path):
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def verify():
        entered.set()
        try:
            assert release.wait(2)
        finally:
            completed.set()
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                task = asyncio.create_task(send(transport))
                try:
                    await wait_thread(entered)
                    task.cancel()
                    await asyncio.sleep(.005)
                    task.cancel()
                    assert not task.done()
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
        asyncio.run(run())
        assert completed.is_set() and calls == [] and budget.status()["attempts"] == 0
        assert transport.fault == "cancelled" and not list(root.glob("physical-*.json"))


def test_cancel_after_response_preserves_known_body_and_usage_without_cross_task_clock(tmp_path):
    entered, network_release, cleanup_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, raw = [], []
    async def handle(req):
        calls.append(req)
        entered.set()
        await network_release.wait()
        value = response()
        raw.append(value.content)
        return value
    class NotifyingGuard(original._ObservedIntentGuard):
        async def _cleanup(self, value):
            try:
                return await super()._cleanup(value)
            finally:
                cleanup_done.set()
    with harness(tmp_path, handle) as (root, budget, transport):
        transport._guard_type = NotifyingGuard
        async def run():
            with eval_deadlines(policy()):
                task = asyncio.create_task(send(transport))
                await entered.wait()
                await transport._metadata_gate.acquire()
                try:
                    network_release.set()
                    await cleanup_done.wait()
                    task.cancel()
                    await asyncio.sleep(.005)
                    task.cancel()
                    assert not task.done()
                finally:
                    transport._metadata_gate.release()
                with pytest.raises(asyncio.CancelledError):
                    await task
        asyncio.run(run())
        record = saved(root)
        assert record["http_status"] == 200 and record["outcome"] == "cancelled"
        assert record["response"]["content"] == [{"type": "text", "text": "controlled known result"}]
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert record["response_sha256"] == hashlib.sha256(raw[0]).hexdigest()
        assert len(calls) == budget.status()["attempts"] == 1
        assert len(list(root.glob("physical-*.json"))) == 1 and not budget.status()["pending"]


def test_model_send_cannot_hide_inside_a_management_scope(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def hidden():
            with context(Frame("c1", "t1", "agent", "general")):
                async with overhead("admission"):
                    return await transport.handle_async_request(request())
        async def run():
            with eval_deadlines(policy()):
                return await await_model(hidden(), timeout_s=.1)
        with pytest.raises(TrialTransportError, match="protocol"):
            asyncio.run(run())
        assert calls == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


@pytest.mark.parametrize("damage", ["version", "wire", "evidence"])
def test_all_pre_send_validation_remains_mandatory_with_suspended_management_clock(tmp_path, damage):
    entered, release = threading.Event(), threading.Event()
    calls, stale = [], {"value": False}
    req = request()
    def verify():
        if damage == "version":
            raise TrialTransportError("protocol")
        entered.set()
        assert release.wait(2)
    def evidence_guard():
        if stale["value"]:
            raise TrialTransportError("protocol")
    async def handle(value):
        calls.append(value)
        return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()):
                task = asyncio.create_task(send(transport, req=req, evidence_guard=evidence_guard))
                if damage != "version":
                    try:
                        await wait_thread(entered)
                        if damage == "wire":
                            old = req.content
                            req._content = json.dumps(json.loads(old), indent=2).encode()
                            assert canonical_payload_hash(json.loads(old)) == canonical_payload_hash(json.loads(req.content))
                            assert old != req.content
                        else:
                            stale["value"] = True
                    finally:
                        release.set()
                with pytest.raises(TrialTransportError):
                    await task
        asyncio.run(run())
        assert calls == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


def test_actual_intent_deadline_and_sdk_serialization_keep_original_model_parameters(tmp_path, monkeypatch):
    """Real production method/SDK, fake provider; no semantic-quality claim."""
    from anthropic import AsyncAnthropic
    from core.intent_recognizer import IntentCategory, IntentRecognizer
    from evaluation.task_transport import BoundClient
    calls = []
    monkeypatch.setenv("ECHOMIND_INTENT_TIMEOUT_S", ".1")  # Explicit test-only scale; runtime default is 8.
    async def handle(req):
        calls.append(json.loads(req.content))
        await asyncio.sleep(.025)
        return httpx.Response(200, json={
            "id": "controlled_response_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": json.dumps({"intent": "refund", "confidence": .9, "reasoning": "controlled classification"})}],
            "model": "qwen3.7-flash-2026-07-15", "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })
    with harness(tmp_path, handle, lambda: time.sleep(.14)) as (root, budget, transport):
        async def run():
            sdk = AsyncAnthropic(api_key="controlled-only-never-persisted", base_url=BASE, max_retries=0,
                http_client=httpx.AsyncClient(transport=transport, timeout=8, trust_env=False, follow_redirects=False))
            recognizer = IntentRecognizer.__new__(IntentRecognizer)
            recognizer.client, recognizer.model = BoundClient(sdk, "intent"), "qwen3.7-flash-2026-07-15"
            try:
                with eval_deadlines(policy()), context(Frame("c1", "t1", "intent", None)):
                    return await recognizer._llm_recognize("查询退款状态", None)
            finally:
                await sdk.close()
        result = asyncio.run(run())
        assert result["intent"] is IntentCategory.REFUND and not result.get("failed")
        assert len(calls) == budget.status()["attempts"] == 1
        assert calls[0]["model"] == "qwen3.7-flash-2026-07-15"
        assert calls[0]["max_tokens"] == 256 and calls[0]["temperature"] == .1
        assert calls[0]["thinking"] == {"type": "disabled"} and calls[0].get("stream", False) is False
        assert saved(root)["timing"]["admission_ms"] >= 120 and saved(root)["outcome"] == "received"


def test_actual_rewrite_stage_keeps_ten_second_contract_with_explicit_test_time_scale(tmp_path, monkeypatch):
    from anthropic import AsyncAnthropic
    from evaluation.task_transport import BoundClient
    from tooling import tool_manager
    calls, configured_deadlines = [], []
    async def scaled(awaitable, timeout_s):
        configured_deadlines.append(timeout_s)
        return await await_model(awaitable, timeout_s=.1)
    monkeypatch.setattr(tool_manager, "await_model", scaled)
    async def handle(req):
        calls.append(json.loads(req.content))
        await asyncio.sleep(.025)
        return httpx.Response(200, json={
            "id": "controlled_response_2", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": '["请核对拆单。", "请核对时效。", "请核对范围。"]'}],
            "model": "qwen3.7-flash-2026-07-15", "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })
    with harness(tmp_path, handle, lambda: time.sleep(.14)) as (root, budget, transport):
        async def run():
            sdk = AsyncAnthropic(api_key="controlled-only-never-persisted", base_url=BASE, max_retries=0,
                http_client=httpx.AsyncClient(transport=transport, timeout=8, trust_env=False, follow_redirects=False))
            manager = tool_manager.MCPToolManager.__new__(tool_manager.MCPToolManager)
            manager._client, manager._model = BoundClient(sdk), "qwen3.7-flash-2026-07-15"
            try:
                with eval_deadlines(policy()), context(Frame("c1", "t1", "rag_rewrite", "general")):
                    return await manager.rewrite_query("请核对拆单。请核对时效。请核对范围。", n=3)
            finally:
                await sdk.close()
        assert asyncio.run(run()) == ["请核对拆单。请核对时效。请核对范围。", "请核对拆单。", "请核对时效。", "请核对范围。"]
        assert configured_deadlines == [10]
        assert len(calls) == budget.status()["attempts"] == 1
        assert calls[0]["max_tokens"] == 512 and calls[0]["temperature"] == .3
        assert calls[0]["thinking"] == {"type": "disabled"}
        assert saved(root)["phase"] == "rag_rewrite" and saved(root)["outcome"] == "received"
