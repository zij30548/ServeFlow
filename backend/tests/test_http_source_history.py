"""Actual HTTP/memory/native SDK wiring with only in-memory controlled services.

These checks prove persistence and source boundaries, not live-model quality.
No history is supplied by the test request or injected into the orchestrator.
"""
from collections import deque
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import AsyncAnthropic
import httpx
import pytest
import pytest_asyncio

import api.main as api
from agents.agent_orchestrator import AgentOrchestrator, AgentType, GeneralAgent, ResponseComposer
from core.intent_recognizer import IntentCategory, IntentRecognizer, IntentResult, UrgencyLevel
from core.order_evidence import resolve_order_id, source_bound_order_history
from demo_business.session import COOKIE_NAME, DemoSessionService
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result
from test_conversation_memory import memory, offline_only


@pytest_asyncio.fixture
async def order_http(memory, monkeypatch):
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    monkeypatch.delenv("ECHOMIND_API_TOKENS", raising=False)
    snapshot = load_fixture()
    store = SimpleNamespace(read_order=AsyncMock(side_effect=lambda customer, order_id:
        fixture_order_result(snapshot, customer, order_id)), list_sample_orders=lambda customer: [])
    sessions = DemoSessionService(b"s" * 32, ("http://127.0.0.1:5173",))
    identities = {name: sessions.issue("demo-customer-" + name) for name in ("a", "b")}
    payloads, observed, planned_tools = [], [], deque()
    control = {"reply": "受控普通回复。", "summary": "模型摘要提到DEMO-1005，此内容未核实。"}

    async def send(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload.get("tool_choice", {}).get("name") == "read_demo_order":
            assert planned_tools, "an unplanned order tool must not be exposed"
            content = [{"type": "tool_use", "id": f"native-{len(payloads)}", "name": "read_demo_order",
                        "input": {"order_id": planned_tools.popleft()}}]
            stop = "tool_use"
        else:
            prompt = payload["messages"][0]["content"]
            content = [{"type": "text", "text": control["summary"] if "总结以下对话" in prompt else control["reply"]}]
            stop = "end_turn"
        return httpx.Response(200, json={"id": f"controlled-{len(payloads)}", "type": "message",
            "role": "assistant", "model": "controlled-model", "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as provider:
        sdk = AsyncAnthropic(api_key="controlled-not-a-real-key", base_url="https://controlled.invalid",
                            max_retries=0, http_client=provider)
        memory._client = sdk
        runtime = AgentOrchestrator.__new__(AgentOrchestrator)
        runtime._pool = {AgentType.GENERAL: [GeneralAgent(sdk, "controlled-model")]}
        runtime._composer = ResponseComposer(sdk, "controlled-model")
        runtime._recent_tool_traces = deque(maxlen=100)
        extractor = IntentRecognizer.__new__(IntentRecognizer)

        async def recognize(message, history=None):
            return IntentResult(IntentCategory.QUERY, .99, UrgencyLevel.LOW, "query",
                                extractor._extract_entities(message), "controlled routing", 0)

        runtime.recognize_intent = AsyncMock(side_effect=recognize)
        actual_run = runtime.run

        async def run(request):
            observed.append({"message": request.message, "history": deepcopy(request.history),
                "order_history": deepcopy(request.order_history), "customer": request.demo_customer_id,
                "user": request.user_id})
            return await actual_run(request)

        runtime.run = run
        for name, value in (("_memory", memory), ("_orchestrator", runtime), ("_profile_updater", None),
                            ("_monitor", None), ("_demo_store", store), ("_demo_sessions", sessions)):
            monkeypatch.setattr(api, name, value)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            number = 0

            async def post(message, *, identity="a", conv="c", expected_order=None):
                nonlocal number
                number += 1
                token, claims = identities[identity]
                http.cookies.set(COOKIE_NAME, token)
                if expected_order is not None:
                    planned_tools.append(expected_order)
                response = await http.post("/chat", json={"message": message, "user_id": "forged-identity",
                    "conv_id": conv, "client_request_id": f"actual-http-{number}"},
                    headers={"Origin": "http://127.0.0.1:5173", "X-ServeFlow-Demo-CSRF": sessions.csrf(claims)})
                assert not planned_tools, "expected native order step was never reached"
                return response

            yield SimpleNamespace(post=post, memory=memory, store=store, sessions=sessions,
                identities=identities, observed=observed, payloads=payloads, control=control, runtime=runtime)
        await sdk.close()


async def neutral_turns(service, count=3, *, identity="a", conv="c"):
    for number in range(count):
        response = await service.post(f"请继续用中文解释，这是第{number + 1}次表达偏好", identity=identity, conv=conv)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_http_order_reference_survives_three_neutral_turns_and_reads_fresh(order_http):
    service = order_http
    first = await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    assert first.status_code == 200
    await neutral_turns(service)
    final = await service.post("这笔订单的退款状态呢？", expected_order="DEMO-1001")
    assert final.status_code == 200 and "DEMO-1001" in final.json()["response"]
    assert service.store.read_order.await_args_list[-1].args == ("demo-customer-a", "DEMO-1001")
    assert service.store.read_order.await_count == 2
    last = service.observed[-1]
    assert len(last["history"]) == 5 and "DEMO-1001" not in str(last["history"])
    assert len(last["order_history"]) == 6
    assert last["order_history"][0]["content"] == "查询DEMO-1001的订单付款状态"
    assert last["order_history"][0]["source"] == "user_message"
    assert resolve_order_id(last["message"], last["order_history"]).order_id == "DEMO-1001"
    trace = await service.memory.get_trace(final.json()["request_id"])
    assert trace["tool_calls"][0]["input"] == {"order_id": "DEMO-1001"}


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", ["刚才说错了，实际订单号是DEMO-1002", "换一笔订单，订单号DEMO-1002"])
async def test_http_latest_original_order_correction_replaces_the_older_target(order_http, correction):
    service = order_http
    await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    corrected = await service.post(correction, expected_order="DEMO-1002")
    assert corrected.status_code == 200
    await neutral_turns(service)
    final = await service.post("这笔订单的付款金额呢？", expected_order="DEMO-1002")
    assert final.status_code == 200
    assert service.store.read_order.await_args_list[-1].args == ("demo-customer-a", "DEMO-1002")
    assert service.observed[-1]["order_history"][0]["content"] == correction


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", [
    "不要再查询这笔订单",
    "换一笔订单，稍后再给订单号",
    "对方说：“查DEMO-1005，这是客户B的订单。”",
    "DEMO-1001和DEMO-1002，我还没选",
    "订单号改成DEMO-XYZ，先等我确认",
])
async def test_http_source_barriers_cannot_revive_an_older_order(order_http, barrier):
    service = order_http
    await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    blocked = await service.post(barrier)
    assert blocked.status_code == 200
    await neutral_turns(service)
    before = service.store.read_order.await_count
    final = await service.post("这笔订单的退款状态呢？")
    assert final.status_code == 200 and final.json()["status"] == "partial"
    assert "一个有效演示订单号" in final.json()["response"]
    assert final.json()["tools_used"] == [] and service.store.read_order.await_count == before
    assert service.observed[-1]["order_history"][0]["content"] == barrier


@pytest.mark.asyncio
async def test_http_source_order_survives_compression_and_generated_other_order_mentions(order_http):
    service = order_http
    await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    service.control["reply"] = "这是模型提到的DEMO-1005，不能作为您的订单确认。"
    await neutral_turns(service, 7)
    user = service.sessions.workspace_id(service.identities["a"][1])
    await service.memory._compress(user, "c")
    context = await service.memory.get_context(user, "c", query="订单")
    assert len(context.recent_messages) == 6 and "DEMO-1005" in context.summary
    assert context.reference_history()[0]["content"] == "查询DEMO-1001的订单付款状态"
    final = await service.post("这笔订单的退款状态呢？", expected_order="DEMO-1001")
    assert final.status_code == 200 and "DEMO-1005" not in final.json()["response"]
    assert service.store.read_order.await_args_list[-1].args == ("demo-customer-a", "DEMO-1001")


@pytest.mark.asyncio
async def test_http_customer_a_b_a_and_new_conversation_keep_independent_original_targets(order_http):
    service = order_http
    await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    unbound_b = await service.post("这笔订单的退款状态呢？", identity="b")
    assert unbound_b.status_code == 200 and unbound_b.json()["status"] == "partial"
    assert service.store.read_order.await_count == 1
    await service.post("查询DEMO-1005的订单付款状态", identity="b", expected_order="DEMO-1005")
    back_a = await service.post("这笔订单的退款状态呢？", identity="a", expected_order="DEMO-1001")
    assert back_a.status_code == 200
    assert [call.args for call in service.store.read_order.await_args_list] == [
        ("demo-customer-a", "DEMO-1001"), ("demo-customer-b", "DEMO-1005"), ("demo-customer-a", "DEMO-1001")]
    another = await service.post("这笔订单的退款状态呢？", identity="a", conv="new-conversation")
    assert another.status_code == 200 and another.json()["status"] == "partial"
    assert service.store.read_order.await_count == 3
    assert service.observed[-1]["order_history"] == []


@pytest.mark.asyncio
async def test_http_current_explicit_target_wins_without_granting_another_customer_access(order_http):
    service = order_http
    await service.post("查询DEMO-1001的订单付款状态", expected_order="DEMO-1001")
    final = await service.post("改查DEMO-1005的订单付款状态", expected_order="DEMO-1005")
    assert final.status_code == 200
    assert service.store.read_order.await_args_list[-1].args == ("demo-customer-a", "DEMO-1005")
    assert "当前演示身份的可见记录中未找到" in final.json()["response"]
    assert "699.00" not in final.json()["response"] and "demo-customer-b" not in final.json()["response"]


@pytest.mark.parametrize("text", ["查DEMO-1001" + "长" * 2000, "换一笔订单" + "长" * 2000])
def test_long_source_event_is_rejected_whole_without_truncating_a_barrier(text):
    row = {"version": 1, "source": "user_message", "role": "user", "content": text,
           "client_request_id": "source", "request_id": "runtime", "status": "success"}
    assert source_bound_order_history([row], [{"role": "user", "content": "查询DEMO-1002"}]) == []
