"""Dynamic E2E transport checks with real SDK serialization and no network."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import socket

import httpx
import pytest
from anthropic import AsyncAnthropic

from evaluation import benchmark as bench
from evaluation import task_transport as module
from evaluation.intent_transport import GuardedIntentTransport, TrialTransportError, canonical_payload_hash
from evaluation.task_budget import TaskBudget, TaskBudgetError
from evaluation.task_transport import BoundClient, CURRENT, Frame, TaskHTTPTransport, context


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
URL = BASE + "/v1/messages"
KEY = "controlled-key-never-persist"
PROMPT = "CONTROLLED ORIGINAL PROMPT MUST NOT BE WRITTEN"
PLAN = [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
        for i in range(1, 7)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("task transport tests must not contact a service")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def body(phase="intent", agent=None):
    tokens, temperature = (module.AGENT_PARAMETERS[agent] if phase == "agent"
                           else module.PHASE_PARAMETERS[phase])
    return {"model": "qwen3.7-flash-2026-07-15", "max_tokens": tokens,
            "temperature": temperature, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": PROMPT}]}


def request(payload=None, *, url=URL, method="POST"):
    return httpx.Request(method, url, json=payload or body(), headers={
        "Authorization": "Bearer " + KEY, "X-Api-Key": KEY})


def sdk_arguments(phase="intent", agent=None):
    payload = body(phase, agent)
    payload["extra_body"] = {"thinking": payload.pop("thinking")}
    return payload


def response(*, tool=False):
    content = ([{"type": "tool_use", "id": "call_1", "name": "read_demo_order",
                 "input": {"order_id": "DEMO-1001"}}] if tool else
               [{"type": "text", "text": "受控回复，仅为传输测试"}])
    return httpx.Response(200, headers={"request-id": "controlled-request-1", "x-private-header": KEY}, json={
        "id": "controlled-response", "type": "message", "role": "assistant",
        "model": "qwen3.7-flash-2026-07-15", "content": content,
        "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 10}})


@contextmanager
def harness(tmp_path, factory, *, verify=lambda: None):
    root = tmp_path / "batch"; root.mkdir(mode=0o700)
    budget = TaskBudget(root / "ledger", "a" * 64, PLAN, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        transport = TaskHTTPTransport(BASE, root, budget, lease, verify, delegate_factory=factory)
        yield root, budget, transport


def saved(root, attempt=1):
    return bench.read_json(root / f"physical-{attempt:03d}.json")


@pytest.mark.parametrize("choice,phase,registered,allowed", [
    ({"type": "tool", "name": "search_knowledge_base"}, "agent", True, True),
    ({"type": "auto"}, "agent", True, False),
    ({"type": "any"}, "agent", True, False),
    ({"type": "none"}, "agent", True, False),
    ({"type": "tool", "name": "read_demo_order"}, "agent", True, True),
    ({"type": "tool", "name": "build_diagnostic_plan"}, "agent", True, True),
    ({"type": "tool", "name": "lookup_error_code"}, "agent", True, True),
    ({"type": "tool", "name": "arbitrary_tool"}, "agent", True, False),
    ({"type": "tool", "name": "search_knowledge_base", "disable_parallel_tool_use": True}, "agent", True, False),
    ({"type": "tool", "name": "search_knowledge_base"}, "agent", False, False),
    ({"type": "tool", "name": "search_knowledge_base"}, "intent", True, False),
])
def test_only_registered_named_readonly_tool_choices_are_admitted(tmp_path, choice, phase, registered, allowed):
    calls = []
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        role = "technical" if choice.get("name") in {"build_diagnostic_plan", "lookup_error_code"} else "general"
        payload = body(phase, role if phase == "agent" else None)
        payload["tool_choice"] = choice
        if registered:
            payload["tools"] = [{"name": choice.get("name", "search_knowledge_base"), "description": "controlled tool",
                                 "input_schema": {"type": "object", "properties": {}}}]
        async def run():
            with context(Frame("c1", "t1", phase, role if phase == "agent" else None)):
                return await transport.handle_async_request(request(payload))
        if allowed:
            assert asyncio.run(run()).status_code == 200
        else:
            with pytest.raises(TrialTransportError):
                asyncio.run(run())
        assert len(calls) == budget.status()["attempts"] == int(allowed)



@pytest.mark.parametrize("name,role", [("search_knowledge_base", "general"),
                                      ("read_demo_order", "billing"),
                                      ("build_diagnostic_plan", "technical"),
                                      ("lookup_error_code", "technical")])
@pytest.mark.parametrize("copies", [0, 1, 2])
def test_named_tool_requires_exactly_one_matching_definition_before_counting(tmp_path, name, role, copies):
    calls = []
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_, budget, transport):
        payload = body("agent", role if role in module.AGENT_PARAMETERS else "general")
        payload["tool_choice"] = {"type": "tool", "name": name}
        payload["tools"] = [{"name": name, "description": "controlled tool",
                             "input_schema": {"type": "object", "properties": {}}} for _ in range(copies)]
        async def run():
            with context(Frame("c1", "t1", "agent", role)):
                return await transport.handle_async_request(request(payload))
        if copies == 1:
            assert asyncio.run(run()).status_code == 200
        else:
            with pytest.raises(TrialTransportError):
                asyncio.run(run())
        assert len(calls) == budget.status()["attempts"] == int(copies == 1)


@pytest.mark.parametrize("name,role", [("read_demo_order", "technical"),
                                      ("read_demo_order", "escalation"),
                                      ("build_diagnostic_plan", "general"),
                                      ("build_diagnostic_plan", "billing"),
                                      ("lookup_error_code", "general"),
                                      ("lookup_error_code", "billing")])
def test_named_readonly_tools_cannot_expand_role_permissions(tmp_path, name, role):
    calls = []
    async def handle(req):
        calls.append(req)
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_, budget, transport):
        payload = body("agent", role if role in module.AGENT_PARAMETERS else "general")
        payload["tool_choice"] = {"type": "tool", "name": name}
        payload["tools"] = [{"name": name, "description": "controlled tool",
                             "input_schema": {"type": "object", "properties": {}}}]
        async def run():
            with context(Frame("c1", "t1", "agent", role)):
                await transport.handle_async_request(request(payload))
        with pytest.raises(TrialTransportError):
            asyncio.run(run())
        assert calls == [] and budget.status()["attempts"] == 0


def test_dynamic_hashes_and_claim_precede_delegate_and_sealed_hash_matches(tmp_path):
    holder = {}; calls = []; verified = []
    async def handle(req):
        root, budget = holder["root"], holder["budget"]
        calls.append(req)
        state = budget.status()
        assert verified == [True] and state["attempts"] == 1
        assert len(state["pending"]) == 1
        operation_id = state["pending"][0]
        filename = hashlib.sha256(operation_id.encode()).hexdigest() + ".json"
        claim = bench.read_json(root / "ledger/claims" / filename)
        assert claim["phase"] == "intent" and claim["agent"] is None
        assert claim["payload_sha256"] == canonical_payload_hash(json.loads(req.content))
        assert claim["request_bytes_sha256"] == hashlib.sha256(req.content).hexdigest()
        assert req.headers["accept-encoding"] == "identity"
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify=lambda: verified.append(True)) as (root, budget, transport):
        holder.update(root=root, budget=budget)
        async def run():
            with context(Frame("c1", "t1", "intent")):
                result = await transport.handle_async_request(request())
            assert result.status_code == 200
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["attempts"] == 1
        assert not budget.status()["pending"] and transport.fault is None
        output = saved(root)
        assert output["evidence_kind"] == "controlled-transport" and output["delegate_entered"] is True
        assert output["outcome"] == "received" and output["response"]["usage"] == {"input_tokens": 20, "output_tokens": 10}
        assert budget.results()["operations"][output["operation_id"]]["output_sha256"] == bench.sha256(root / "physical-001.json")
        all_files = "\n".join(p.read_text() for p in root.rglob("*") if p.is_file())
        assert KEY not in all_files and PROMPT not in all_files and "x-private-header" not in all_files


@pytest.mark.parametrize("tool", [False, True])
def test_sdk_native_text_and_tool_use_once_with_no_sdk_retry(tmp_path, tool):
    calls = []
    async def handle(req):
        calls.append(req)
        return response(tool=tool)
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        async def run():
            http = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False)
            sdk = AsyncAnthropic(api_key=KEY, base_url=BASE, max_retries=0, http_client=http)
            try:
                assert sdk.max_retries == 0
                client = BoundClient(sdk, "intent")
                with context(Frame("c1", "t1", "agent", "technical")):
                    value = await client.messages.create(**sdk_arguments())
                    assert CURRENT.get() == Frame("c1", "t1", "agent", "technical")
                assert value.content[0].type == ("tool_use" if tool else "text")
            finally:
                await sdk.close()
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["attempts"] == 1
        output = saved(root)
        assert output["phase"] == "intent" and output["agent"] is None
        assert output["response"]["content"][0]["type"] == ("tool_use" if tool else "text")


def test_contextvars_isolate_parallel_agent_calls_and_nonagent_facade_clears_actor(tmp_path):
    seen = []
    def factory(frame):
        async def handle(req):
            seen.append((frame, json.loads(req.content)))
            await asyncio.sleep(.01 if frame.agent == "technical" else 0)
            return response()
        return httpx.MockTransport(handle)
    with harness(tmp_path, factory) as (root, budget, transport):
        async def run():
            http = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False)
            sdk = AsyncAnthropic(api_key=KEY, base_url=BASE, max_retries=0, http_client=http)
            async def call(role):
                with context(Frame("c1", "t1", "agent", "billing")):
                    client = BoundClient(sdk, "agent", role)
                    await client.messages.create(**sdk_arguments("agent", role))
                    assert CURRENT.get().agent == "billing"
            try:
                await asyncio.gather(call("technical"), call("general"), call("billing"))
            finally:
                await sdk.close()
            assert CURRENT.get() is None
        asyncio.run(run())
        assert {frame.agent for frame, _ in seen} == {"general", "technical", "billing"}
        assert all(frame.phase == "agent" for frame, _ in seen)
        assert budget.status()["attempts"] == 3 and not budget.status()["pending"]
        assert {saved(root, i)["agent"] for i in range(1, 4)} == {"general", "technical", "billing"}


@pytest.mark.parametrize("change", ["unknown_phase", "wrong_model", "stream", "thinking", "extra", "bool_temperature", "wrong_tokens", "url", "method", "system"])
def test_invalid_envelopes_are_rejected_before_any_delegate(tmp_path, change):
    calls = []
    async def handle(req): calls.append(req); return response()
    phase = "judge" if change == "bool_temperature" else "intent"
    payload = body(phase)
    frame = Frame("c1", "t1", phase)
    url, method = URL, "POST"
    if change == "unknown_phase": frame = Frame("c1", "t1", "memory")
    elif change == "wrong_model": payload["model"] = "unapproved-model"
    elif change == "stream": payload["stream"] = True
    elif change == "thinking": payload["thinking"] = {"type": "enabled"}
    elif change == "extra": payload["user_url"] = "https://unapproved.invalid/"
    elif change == "bool_temperature": payload["temperature"] = False
    elif change == "wrong_tokens": payload["max_tokens"] = 4097
    elif change == "url": url = "https://unapproved.invalid/v1/messages"
    elif change == "method": method = "GET"
    else: payload["system"] = "nonagent system is not permitted"
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_root, budget, transport):
        async def run():
            with context(frame), pytest.raises(TrialTransportError):
                await transport.handle_async_request(request(payload, url=url, method=method))
        asyncio.run(run())
        assert not calls and budget.status()["attempts"] == 0
        assert transport.fault == "protocol"


@pytest.mark.parametrize("base", ["https://unapproved.invalid/apps/anthropic", "http://dashscope.aliyuncs.com/apps/anthropic", BASE + "?x=1", BASE + "#secret", "https://dashscope.aliyuncs.com:443/apps/anthropic"])
def test_arbitrary_base_url_is_never_accepted(tmp_path, base):
    with pytest.raises(TrialTransportError):
        TaskHTTPTransport(base, tmp_path, None, None, lambda: None, delegate_factory=lambda frame: None)


def test_missing_context_and_none_factory_cannot_fall_through_to_network(tmp_path):
    with harness(tmp_path, lambda frame: None) as (_root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError):
                await transport.handle_async_request(request())
        asyncio.run(run())
        assert budget.status()["attempts"] == 0
    with pytest.raises(TrialTransportError):
        with context(): pass


@pytest.mark.parametrize("failure,expected", [
    (httpx.Response(401, text=KEY), "auth"),
    (httpx.Response(429, text=KEY), "rate_limit"),
    (httpx.Response(302, headers={"location": "https://unapproved.invalid/"}), "network"),
    (httpx.ReadTimeout(KEY), "timeout"), (httpx.ConnectError(KEY), "network"),
    (asyncio.CancelledError(), "cancelled"),
])
def test_sent_failures_pause_occupy_and_do_not_retry_or_leak(tmp_path, failure, expected):
    calls = []
    async def handle(req):
        calls.append(req)
        if isinstance(failure, BaseException): raise failure
        return failure
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")):
                with pytest.raises(TrialTransportError, match=expected):
                    await transport.handle_async_request(request())
                with pytest.raises(TrialTransportError):
                    await transport.handle_async_request(request())
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["attempts"] == 1 and budget.status()["paused"]
        assert not budget.status()["pending"] and transport.fault == expected
        assert saved(root)["outcome"] in {"execution_error", "cancelled"}
        assert saved(root)["response"]["content"] == []
        assert KEY not in "\n".join(p.read_text() for p in root.rglob("*") if p.is_file())


class Stream(httpx.AsyncByteStream):
    def __init__(self, data): self.data, self.closed = data, False
    async def __aiter__(self): yield self.data
    async def aclose(self): self.closed = True


@pytest.mark.parametrize("kind,expected", [("size", "response_limit"), ("encoding", "network")])
def test_response_size_and_encoding_fail_closed(tmp_path, kind, expected):
    calls = []; stream = Stream(b"x" * (262145 if kind == "size" else 10))
    async def handle(req):
        calls.append(req)
        return httpx.Response(200, stream=stream, headers={"content-encoding": "gzip"} if kind == "encoding" else {})
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError, match=expected):
                await transport.handle_async_request(request())
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["attempts"] == 1 and budget.status()["paused"]
        assert stream.closed is True


def test_guard_timeout_is_controlled_and_never_retried(tmp_path, monkeypatch):
    calls = []
    original = GuardedIntentTransport
    monkeypatch.setattr(module, "GuardedIntentTransport", lambda *args, **kwargs: original(*args, timeout_s=.01, **kwargs))
    async def handle(req):
        calls.append(req)
        await asyncio.sleep(1)
        return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError, match="timeout"):
                await transport.handle_async_request(request())
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["paused"]
        assert budget.status()["attempts"] == 1


def test_request_size_cap_has_no_reservation_or_delegate(tmp_path):
    calls = []
    async def handle(req): calls.append(req); return response()
    payload = body(); payload["messages"][0]["content"] = "x" * 65537
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError):
                await transport.handle_async_request(request(payload))
        asyncio.run(run())
        assert not calls and budget.status()["attempts"] == 0


def test_second_judge_attempt_is_blocked_before_delegate_and_keeps_quota_reason(tmp_path):
    calls = []
    async def handle(req): calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "judge")):
                await transport.handle_async_request(request(body("judge")))
                with pytest.raises(TrialTransportError, match="quota"):
                    await transport.handle_async_request(request(body("judge")))
        asyncio.run(run())
        assert len(calls) == 1 and budget.status()["attempts"] == 1
        assert transport.fault == "quota" and bench.read_json(root / "stopped.json")["error_code"] == "quota"


def test_ordinary_response_structure_failure_is_sealed_without_automatic_retry(tmp_path):
    calls = []
    async def handle(req):
        calls.append(req)
        return httpx.Response(200, content=b"not valid JSON") if len(calls) == 1 else response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "agent", "general")):
                first = await transport.handle_async_request(request(body("agent", "general")))
                assert first.content == b"not valid JSON" and len(calls) == 1
                # A distinct next planned model step, never a hidden retry.
                await transport.handle_async_request(request(body("agent", "general")))
        asyncio.run(run())
        assert budget.status()["attempts"] == 2 and not budget.status()["paused"]
        assert saved(root, 1)["outcome"] == "structure_failed"
        assert saved(root, 2)["outcome"] == "received"
        assert transport.fault is None
        for index in (1, 2):
            value = saved(root, index)
            assert budget.results()["operations"][value["operation_id"]]["output_sha256"] == bench.sha256(root / f"physical-{index:03d}.json")


@pytest.mark.parametrize("raw", [b"{", b'{"model":"x","model":"x"}', b'{"x":NaN}', b'\xff'])
def test_malformed_request_json_never_reserves_or_sends(tmp_path, raw):
    calls = []
    async def handle(req): calls.append(req); return response()
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (_root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError, match="protocol"):
                await transport.handle_async_request(httpx.Request("POST", URL, content=raw))
        asyncio.run(run())
        assert not calls and budget.status()["attempts"] == 0


def test_admission_integrity_failure_happens_before_reservation(tmp_path):
    calls = []
    async def handle(req): calls.append(req); return response()
    def changed_version():
        raise RuntimeError("sensitive detail that must not enter evidence")
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify=changed_version) as (root, budget, transport):
        async def run():
            with context(Frame("c1", "t1", "intent")), pytest.raises(TrialTransportError):
                await transport.handle_async_request(request())
        asyncio.run(run())
        assert not calls and budget.status()["attempts"] == 0
        assert "sensitive detail" not in (root / "stopped.json").read_text()


@pytest.mark.parametrize("case_id", ["e01", "e02", "e03", "e04", "e05", "e06"])
def test_real_sdk_transport_and_orchestrator_are_connected(tmp_path, monkeypatch, case_id):
    """Real SDK serialization, bounded claims and production loops, fake replies.

    This controlled wiring test is not a model quality measurement.
    """
    from test_task_runtime import ControlledSDK, ControlledData, case
    from evaluation.task_runtime import execute_case
    for key in tuple(__import__("os").environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)
    async def run():
        root = tmp_path / "integrated"; root.mkdir(mode=0o700)
        ids = (case_id, *[f"e{i:02d}" for i in range(1, 7) if f"e{i:02d}" != case_id])
        values = [case(c) for c in ids]
        plan = [{"case_id": c["id"], "turn_ids": [f"turn-{i}" for i in range(1, len(c["turns"]) + 1)]} for c in values]
        ledger = TaskBudget(root / "ledger", "a" * 64, plan, initialize=True)
        script, data, sent = ControlledSDK(), ControlledData(), []
        async def upstream(request):
            sent.append(json.loads(request.content))
            output = await script.create(**sent[-1])
            return httpx.Response(200, headers={"request-id": f"controlled-{len(sent)}"}, json={
                "id": f"controlled-{len(sent)}", "type": "message", "role": "assistant",
                "model": "qwen3.7-flash-2026-07-15", "content": output.content,
                "stop_reason": output.stop_reason, "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 10}})
        with ledger.exclusive_run() as lease:
            ledger.start_case(case_id, lease=lease)
            transport = TaskHTTPTransport(BASE, root, ledger, lease, data.assert_healthy,
                                          delegate_factory=lambda frame: httpx.MockTransport(upstream))
            http = httpx.AsyncClient(transport=transport, timeout=8, follow_redirects=False, trust_env=False)
            sdk = AsyncAnthropic(api_key=KEY, base_url=BASE, max_retries=0, timeout=8, http_client=http)
            try:
                def healthy():
                    assert transport.fault is None
                    data.assert_healthy()
                output = await execute_case(values[0], sdk, data, ensure_healthy=healthy)
            finally:
                await sdk.close()
            state = ledger.status()
            assert state["attempts"] == len(sent) > 2 and state["pending"] == []
            assert not state["paused"] and transport.fault is None
            assert output["judge_ready"] and output["semantic_review"] is None
            assert output["deterministic"]["blockers"] == []
            assert all(output["deterministic"]["checks"].values())
            physical = [saved(root, i) for i in range(1, len(sent) + 1)]
            assert all(row["evidence_kind"] == "controlled-transport" for row in physical)
            assert sum(row["phase"] == "judge" for row in physical) == len(values[0]["turns"])
            if case_id == "e03":
                phases = {row["phase"] for row in physical}
                assert phases >= {"rag_rewrite", "rag_rerank"}
                assert "composer" not in phases  # Policy merge is deterministic.
                assert "controlled composed reply" not in output["answer"]
                assert "配送说明" in output["answer"]
                assert all(payload.get("tool_choice") == {"type": "tool", "name": "search_knowledge_base"}
                           for payload, (frame, _) in zip(sent, script.calls) if frame.phase == "agent")
            assert not (root / "stopped.json").exists()
    asyncio.run(run())


def test_overdeep_json_is_retained_as_structure_failure_without_execution_pause(tmp_path):
    async def handle(request):
        return httpx.Response(200, content=b'[' * 10000 + b'0' + b']' * 10000)
    async def run():
        with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, ledger, transport):
            with context(Frame("c1", "t1", "intent")):
                response = await transport.handle_async_request(request())
            assert response.status_code == 200
            assert ledger.status()["attempts"] == 1 and not ledger.status()["paused"]
            assert saved(root)["outcome"] == "structure_failed"
            assert transport.fault is None and not (root / "stopped.json").exists()
    asyncio.run(run())


def test_facade_checks_shared_turn_latch_before_any_sdk_call_and_context_resets():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from evaluation.task_evidence import CaptureTools, EvidenceError
    sdk = SimpleNamespace(base_url=BASE, messages=SimpleNamespace(create=AsyncMock(return_value='controlled')))
    client = BoundClient(sdk, 'agent', 'general')
    failed, fresh = CaptureTools('case-a', 1), CaptureTools('case-b', 1)
    failed._block('evidence_invalid_json')
    async def run():
        with context(Frame('case-a', 't1', evidence_guard=failed.assert_healthy)):
            with pytest.raises(EvidenceError, match='evidence_invalid_json'):
                await client.messages.create()
        assert CURRENT.get() is None
        with context(Frame('case-b', 't1', evidence_guard=fresh.assert_healthy)):
            assert await client.messages.create() == 'controlled'
        assert CURRENT.get() is None
    asyncio.run(run())
    assert sdk.messages.create.await_count == 1


def test_direct_transport_rechecks_evidence_before_reserve_not_storage_failure(tmp_path):
    from evaluation.task_evidence import CaptureTools
    capture, sent = CaptureTools('c1', 1), []
    async def handle(req): sent.append(req); return response()
    def verify(): capture._block('evidence_invalid_json')
    with harness(tmp_path, lambda frame: httpx.MockTransport(handle), verify=verify) as (root, ledger, transport):
        async def run():
            with context(Frame('c1','t1','intent', evidence_guard=capture.assert_healthy)):
                with pytest.raises(TrialTransportError, match='protocol'):
                    await transport.handle_async_request(request())
        asyncio.run(run())
        assert sent == [] and ledger.status()['attempts'] == 0
        stopped = bench.read_json(root/'stopped.json')
        assert stopped['error_code'] == 'protocol' and stopped['evidence_error'] == 'evidence_invalid_json'


def test_queued_other_role_cannot_send_after_shared_capture_failure(tmp_path):
    from evaluation.task_evidence import CaptureTools
    capture, sent = CaptureTools('c1', 1), []
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def handle(req):
            sent.append(req)
            if len(sent) == 3: entered.set()
            await release.wait()
            return response()
        with harness(tmp_path, lambda frame: httpx.MockTransport(handle)) as (root, ledger, transport):
            async def invoke(role):
                with context(Frame('c1', 't1', 'agent', role, capture.assert_healthy)):
                    return await transport.handle_async_request(request(body('agent',role)))
            active = [asyncio.create_task(invoke(role)) for role in ('general','technical','billing')]
            await entered.wait()
            queued = asyncio.create_task(invoke('general'))
            await asyncio.sleep(0)
            capture._block('evidence_invalid_json')
            release.set()
            values = await asyncio.gather(*active, queued, return_exceptions=True)
            assert all(isinstance(v,httpx.Response) for v in values[:3])
            assert isinstance(values[3],TrialTransportError)
            assert len(sent) == ledger.status()['attempts'] == 3
            assert ledger.status()['pending'] == []
            assert all(saved(root,i)['outcome']=='received' for i in (1,2,3))
            assert bench.read_json(root/'stopped.json')['evidence_error']=='evidence_invalid_json'
    asyncio.run(run())


@pytest.mark.parametrize('case_id', ['e01','e02'])
def test_real_sdk_native_capture_fault_spends_only_initial_two_requests(tmp_path, monkeypatch, case_id):
    """Serialization damage cannot spend a loop, fallback, composer or Judge send."""
    from test_task_runtime import ControlledSDK, ControlledData, case
    from evaluation.task_evidence import EvidenceError
    from evaluation.task_runtime import execute_case
    for key in tuple(__import__('os').environ):
        if key.startswith(('ECHOMIND_','SILICONFLOW_','EVAL_JUDGE_')): monkeypatch.delenv(key)
    class NonJSONData(ControlledData):
        async def order_reader(self, customer, order):
            value=await super().order_reader(customer,order)
            value['order']['entitlements']=()
            return value
    async def run():
        root=tmp_path/'contract-failure';root.mkdir(mode=0o700)
        ids=(case_id,*[f'e{i:02d}' for i in range(1,7) if f'e{i:02d}'!=case_id])
        values=[case(c) for c in ids]
        plan=[{'case_id':c['id'],'turn_ids':[f'turn-{i}' for i in range(1,len(c['turns'])+1)]} for c in values]
        ledger=TaskBudget(root/'ledger','a'*64,plan,initialize=True)
        script,data,sent=ControlledSDK(),NonJSONData(),[]
        async def upstream(request):
            sent.append(CURRENT.get())
            output=await script.create(**json.loads(request.content))
            return httpx.Response(200,json={'id':f'controlled-{len(sent)}','type':'message','role':'assistant',
                'model':'qwen3.7-flash-2026-07-15','content':output.content,'stop_reason':output.stop_reason,
                'stop_sequence':None,'usage':{'input_tokens':20,'output_tokens':10}})
        with ledger.exclusive_run() as lease:
            ledger.start_case(case_id,lease=lease)
            transport=TaskHTTPTransport(BASE,root,ledger,lease,data.assert_healthy,
                                       delegate_factory=lambda frame:httpx.MockTransport(upstream))
            http=httpx.AsyncClient(transport=transport,timeout=8,follow_redirects=False,trust_env=False)
            sdk=AsyncAnthropic(api_key=KEY,base_url=BASE,max_retries=0,timeout=8,http_client=http)
            try:
                with pytest.raises(EvidenceError,match='evidence_invalid_json'):
                    await execute_case(values[0],sdk,data,ensure_healthy=data.assert_healthy)
            finally:await sdk.close()
            assert [frame.phase for frame in sent]==['intent','agent']
            assert ledger.status()['attempts']==2 and ledger.status()['pending']==[]
            assert len(data.orders)==1
            assert all(saved(root,i)['outcome']=='received' for i in (1,2))
            assert not (root/'physical-003.json').exists()
        # No future runner lease may resume this unsealed interrupted case.
        with pytest.raises(TaskBudgetError,match='pending'):
            with ledger.exclusive_run():pass
    asyncio.run(run())


def test_restored_turn_frame_cannot_revive_even_when_same_case_is_reopened():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from test_task_evidence import Agent, orc
    from evaluation.task_evidence import CaptureTools, EvidenceError
    sdk=SimpleNamespace(base_url=BASE,messages=SimpleNamespace(create=AsyncMock(return_value='controlled')))
    agent=Agent()
    old=CaptureTools('same-case',1).install(orc(agent))
    frame=Frame('same-case','turn-1',evidence_guard=old.assert_send_allowed)
    old.restore()
    fresh=CaptureTools('same-case',1).install(orc(agent))
    client=BoundClient(sdk,'agent','general')
    async def run():
        with context(frame),pytest.raises(EvidenceError,match='capture_not_active'):
            await client.messages.create()
        with context(Frame('same-case','turn-1',evidence_guard=fresh.assert_send_allowed)):
            assert await client.messages.create()=='controlled'
        assert CURRENT.get() is None
    asyncio.run(run())
    assert sdk.messages.create.await_count==1
    fresh.restore()
