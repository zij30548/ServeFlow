"""Real SDK serialization and local durable accounting; all upstreams mocked."""
import asyncio
from contextlib import contextmanager
import json
import socket
import time

from anthropic import AsyncAnthropic
import httpx
import pytest

from core.model_deadline import EvaluationDeadlinePolicy, ModelDeadlineExceeded, await_model, eval_deadlines
from evaluation import benchmark as bench, intent_trial as base
from evaluation.intent_transport import GuardedIntentTransport, TrialTransportError, canonical_payload_hash
from evaluation.phase_deadline_transport import (
    PHASE_DEADLINES, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S, PhaseDeadlineTaskHTTPTransport,
    _PhaseObservedGuard,
)
from evaluation.task_budget import TaskBudget
from evaluation.task_transport import AGENT_PARAMETERS, PHASE_PARAMETERS, Frame, context


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PLAN = [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]} for i in range(1, 7)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("phase transport tests must not contact a service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def payload(phase="agent"):
    tokens, temperature = AGENT_PARAMETERS["general"] if phase == "agent" else PHASE_PARAMETERS[phase]
    return {"model": base.MODEL, "max_tokens": tokens, "temperature": temperature,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "controlled input"}]}


def request(phase="agent", timeout=30):
    return httpx.Request("POST", BASE + "/v1/messages", json=payload(phase),
                        extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), timeout)})


def response():
    return httpx.Response(200, headers={"request-id": "controlled-request"}, json={
        "id": "msg_controlled", "type": "message", "role": "assistant", "model": base.MODEL,
        "content": [{"type": "text", "text": "controlled known result"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    })


@contextmanager
def harness(tmp_path, handle, verify=lambda: None, *, transport_cls=PhaseDeadlineTaskHTTPTransport):
    root = tmp_path / "batch"; root.mkdir(mode=0o700)
    base._write(root, "protocol.json", {"initial_intent_requests": []})
    budget = TaskBudget(root / "ledger", "a" * 64, PLAN, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        transport = transport_cls(BASE, root, budget, lease, verify,
            delegate_factory=lambda frame: httpx.MockTransport(handle))
        yield root, budget, transport


async def send(transport, *, phase="agent", req=None):
    with context(Frame("c1", "t1", phase, "general" if phase == "agent" else None)):
        return await await_model(transport.handle_async_request(req or request(phase)), timeout_s=PHASE_DEADLINES[phase])


def policy():
    return EvaluationDeadlinePolicy(overhead_limit_s=.5, cleanup_grace_s=.02)


def saved(root, number=1):
    return bench.read_json(root / f"physical-{number:03d}.json")


def test_closed_contract_matches_original_stage_values_and_old_guard_stays_eight():
    assert PHASE_DEADLINES == {"intent": 8, "agent": 15, "rag_rewrite": 10, "rag_rerank": 30, "composer": 8, "judge": 8}
    assert SDK_TIMEOUT_S == PHASE_HTTP_LIMITS["request_deadline_s"] == 30
    with pytest.raises(TrialTransportError, match="invalid_request"):
        GuardedIntentTransport(BASE, "a" * 64, lambda: None, timeout_s=15)
    with pytest.raises(TypeError):
        _PhaseObservedGuard(BASE, "a" * 64, lambda: None, timeout_s=15)
    with pytest.raises(TrialTransportError, match="invalid_request"):
        _PhaseObservedGuard(BASE, "a" * 64, lambda: None)


@pytest.mark.parametrize("invalid_phase", [False, True])
def test_missing_evaluation_clock_or_unknown_phase_never_claims_or_sends(tmp_path, invalid_phase):
    sent = []
    async def handle(req): sent.append(req); return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            frame = Frame("c1", "t1", "unknown" if invalid_phase else "agent", "general")
            with context(frame):
                if invalid_phase:
                    with eval_deadlines(policy()):
                        return await await_model(transport.handle_async_request(request()), timeout_s=.1)
                return await transport.handle_async_request(request())
        with pytest.raises(TrialTransportError, match="protocol"): asyncio.run(run())
        assert sent == [] and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


@pytest.mark.parametrize("phase", tuple(PHASE_DEADLINES))
def test_actual_sdk_timeout_extensions_are_tightened_and_generation_payload_unchanged(tmp_path, phase):
    seen = []
    async def handle(req):
        seen.append((dict(req.extensions["timeout"]), json.loads(req.content)))
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            http = httpx.AsyncClient(transport=transport, timeout=30, follow_redirects=False, trust_env=False)
            sdk = AsyncAnthropic(api_key="controlled-not-a-credential", base_url=BASE, max_retries=0,
                                 timeout=30, http_client=http)
            try:
                with eval_deadlines(policy()), context(Frame("c1", "t1", phase, "general" if phase == "agent" else None)):
                    kwargs = payload(phase)
                    thinking = kwargs.pop("thinking")
                    return await await_model(sdk.messages.create(**kwargs, extra_body={"thinking": thinking}),
                                             timeout_s=PHASE_DEADLINES[phase])
            finally:
                assert http.timeout.as_dict() == dict.fromkeys(("connect", "read", "write", "pool"), 30)
                await sdk.close()
        result = asyncio.run(run())
        assert result.content[0].text == "controlled known result"
        assert seen == [(dict.fromkeys(("connect", "read", "write", "pool"), float(PHASE_DEADLINES[phase])), payload(phase))]
        record = saved(root)
        assert record["network_limit_s"] == record["active_stage_limit_s"] == PHASE_DEADLINES[phase]
        assert record["sdk_limit_s"] == 30 and record["delegate_entered"] is True
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert budget.status()["attempts"] == 1 and not budget.status()["paused"]
        assert len(list(root.glob("request-*.json"))) == 1


def test_scaled_actor_can_exceed_old_network_cap_within_unchanged_phase_budget(tmp_path, monkeypatch):
    # 80/240ms are deliberate scales of the old 8s vs phase budget relation,
    # not new production limits or measured provider latency.
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .24)
    async def handle(req):
        await asyncio.sleep(.12)
        return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await send(transport)
        assert asyncio.run(run()).status_code == 200
        assert saved(root)["timing"]["network_ms"] >= 110
        assert budget.status()["attempts"] == 1
    req = request()
    old = GuardedIntentTransport(BASE, canonical_payload_hash(payload()), lambda: None,
        transport=httpx.MockTransport(handle), timeout_s=.08)
    with pytest.raises(TrialTransportError, match="timeout"):
        asyncio.run(old.handle_async_request(req))


@pytest.mark.parametrize("bad", [8, None, True, float("nan"), "30"])
def test_old_or_invalid_sdk_extension_fails_before_claim_or_send(tmp_path, bad):
    calls = []
    async def handle(req): calls.append(req); return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await send(transport, req=request(timeout=bad))
        with pytest.raises(TrialTransportError, match="protocol"):
            asyncio.run(run())
        assert not calls and budget.status()["attempts"] == 0
        assert not list(root.glob("physical-*.json"))


def test_full_body_shares_total_phase_limit_and_stream_chunks_cannot_extend_it(tmp_path, monkeypatch):
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .06)
    closed = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(.01)
                yield b"x"
        async def aclose(self): closed.append(True)
    async def handle(req): return httpx.Response(200, headers={"request-id": "controlled-body"}, stream=Stream())
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await send(transport)
        with pytest.raises((TrialTransportError, ModelDeadlineExceeded), match="timeout|model_deadline_active"):
            asyncio.run(run())
        record = saved(root)
        assert record["http_status"] == 200 and record["request_id"] == "controlled-body"
        assert record["delegate_entered"] is True and record["error_code"] == "timeout"
        assert record["response"]["usage"]["input_tokens"] is None
        assert budget.status()["attempts"] == 1 and budget.status()["paused"]
        assert closed


def test_management_exclusion_is_still_separate_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .06)
    def verify(): time.sleep(.09)
    async def handle(req): await asyncio.sleep(.01); return response()
    with harness(tmp_path, handle, verify) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await send(transport)
        assert asyncio.run(run()).status_code == 200
        assert saved(root)["timing"]["admission_ms"] >= 80
        assert saved(root)["active_stage_limit_s"] == .06


def test_parallel_requests_have_independent_phase_configuration(tmp_path, monkeypatch):
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .20)
    monkeypatch.setitem(PHASE_DEADLINES, "intent", .12)
    async def handle(req): await asyncio.sleep(.01); return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await asyncio.gather(send(transport), send(transport, phase="intent"))
        assert [r.status_code for r in asyncio.run(run())] == [200, 200]
        records = [saved(root, n) for n in (1, 2)]
        assert {r["phase"]: r["network_limit_s"] for r in records} == {"agent": .20, "intent": .12}
        assert all(r["sdk_limit_s"] == 30 for r in records)


def test_repeated_cancel_after_response_keeps_known_usage_and_phase_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .5)
    async def run(root, budget, transport):
        entered, release = asyncio.Event(), asyncio.Event()
        async def handle(req):
            entered.set(); await release.wait(); return response()
        transport.factory = lambda frame: httpx.MockTransport(handle)
        with eval_deadlines(policy()):
            task = asyncio.create_task(send(transport))
            await entered.wait()
            await transport._metadata_gate.acquire()
            release.set()
            await asyncio.sleep(.02)
            task.cancel(); await asyncio.sleep(.01); task.cancel()
            transport._metadata_gate.release()
            with pytest.raises(asyncio.CancelledError): await task
    with harness(tmp_path, lambda req: response()) as (root, budget, transport):
        asyncio.run(run(root, budget, transport))
        record = saved(root)
        assert record["outcome"] == "cancelled" and record["delegate_entered"] is True
        assert record["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert record["network_limit_s"] == record["active_stage_limit_s"] == .5
        assert record["sdk_limit_s"] == 30
        assert budget.status()["attempts"] == 1 and budget.status()["paused"]


@pytest.mark.parametrize("kind", ["expiry", "external_cancel"])
def test_post_reserve_preflight_never_sends_after_observed_expiry_or_cancel(tmp_path, monkeypatch, kind):
    monkeypatch.setitem(PHASE_DEADLINES, "agent", .05)
    sent = []
    async def handle(req): sent.append(req); return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        reserve = budget.reserve
        def late(*args, **kwargs):
            claim = reserve(*args, **kwargs)
            if kind == "expiry": time.sleep(.07)
            else: asyncio.current_task().cancel()
            return claim
        budget.reserve = late
        async def run():
            with eval_deadlines(policy()): return await send(transport)
        wanted = asyncio.CancelledError if kind == "external_cancel" else (TrialTransportError, ModelDeadlineExceeded)
        with pytest.raises(wanted): asyncio.run(run())
        record = saved(root)
        assert sent == [] and record["delegate_entered"] is False
        assert record["outcome"] == ("cancelled" if kind == "external_cancel" else "execution_error")
        assert record["error_code"] == ("cancelled" if kind == "external_cancel" else "timeout")
        assert budget.status()["attempts"] == 1 and budget.status()["paused"]
        assert budget.status()["pending"] == []


def test_legacy_quality_transport_keeps_old_limit_and_diagnostics(tmp_path):
    from evaluation.quality_trial import QualityHTTPTransport
    async def handle(req):
        # The old path does not rewrite the caller's timeout extensions.
        assert req.extensions["timeout"]["read"] == 8
        return response()
    with harness(tmp_path, handle, transport_cls=QualityHTTPTransport) as (root, budget, transport):
        async def run():
            with eval_deadlines(policy()): return await send(transport, req=request(timeout=8))
        assert asyncio.run(run()).status_code == 200
        assert not {"network_limit_s", "sdk_limit_s", "active_stage_limit_s"} & saved(root).keys()


@pytest.mark.parametrize("field,value", [("network_limit_s", True), ("sdk_limit_s", 31),
    ("active_stage_limit_s", float("nan")), ("network_limit_s", 0), ("untrusted_name", 15)])
def test_invalid_diagnostic_cannot_be_persisted_or_refund_occupied_claim(tmp_path, field, value):
    async def handle(req): return response()
    with harness(tmp_path, handle) as (root, budget, transport):
        transport._physical_diagnostics = lambda: {field: value}
        async def run():
            with eval_deadlines(policy()): return await send(transport)
        with pytest.raises(TrialTransportError): asyncio.run(run())
        assert budget.status()["attempts"] == 1
        assert len(budget.status()["pending"]) == 1
        assert not list(root.glob("physical-*.json"))
        assert (root / "stopped.json").exists()
