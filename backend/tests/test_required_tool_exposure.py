"""Current-step tool menus and unchanged native rejection, with no live services.

The rejected double-call fixture is structurally equivalent to the observed
lookup-plus-RAG failure; replay is a rejection regression, not a live fix score.
"""
from copy import deepcopy
from dataclasses import replace
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import AsyncAnthropic
import httpx
import pytest

from agents.agent_orchestrator import BillingAgent, GeneralAgent, Request, TechnicalAgent
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory
from core.provider_options import FLASH_SNAPSHOT
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("controlled tests must not contact external services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def block(name, args, identity="tool_1"):
    return {"type": "tool_use", "id": identity, "name": name, "input": args}


def native(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use")


def text(value="受控答复"):
    return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")


class Client:
    def __init__(self, *responses):
        self.messages = self
        self.responses, self.calls = list(responses), []

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "unexpected retry or continuation"
        return self.responses.pop(0)


class Retrieval:
    def __init__(self):
        self.search_with_rewrite = AsyncMock(return_value=SimpleNamespace(
            success=True, reranked=False, error=None, data=[{
                "document_id": "a" * 64, "title": "受控规则", "content": "仅作政策参考，不是订单状态。",
                "version": "b" * 64, "chunk": 0, "score": .9,
            }]))


def request(message="核对DEMO-1009付款并排查401故障"):
    fixture = load_fixture()
    async def read(customer, order):
        return fixture_order_result(fixture, customer, order)
    return Request(message=message, user_id="isolated", conv_id="controlled",
                   intent=IntentCategory.TECHNICAL_LOGIN, demo_customer_id="demo-customer-a",
                   demo_order_reader=AsyncMock(side_effect=read))


def menus(client):
    return [[tool["name"] for tool in call["tools"]] for call in client.calls]


@pytest.mark.asyncio
@pytest.mark.parametrize("role,message,name,args", [
    (TechnicalAgent, "核对DEMO-1009付款并排查401故障", "lookup_error_code", {"error_code": "401"}),
    (TechnicalAgent, "DEMO-1003付款后权益开通失败，请技术排障", "build_diagnostic_plan",
     {"environment": "浏览器", "reproduced": True}),
    (BillingAgent, "核对DEMO-1004退款状态", "read_demo_order", {"order_id": "DEMO-1004"}),
    (GeneralAgent, "查询DEMO-1001物流状态", "read_demo_order", {"order_id": "DEMO-1001"}),
    (BillingAgent, "说明退款规则", "search_knowledge_base", {"query": "退款规则"}),
])
async def test_bound_step_advertises_only_required_tool(role, message, name, args):
    client, manager = Client(native(block(name, args))), Retrieval()
    agent = role(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    registry = agent.get_tools()
    result = await agent.handle(request(message))
    assert result.status == "success" and result.tools_used == [name]
    assert menus(client) == [[name]]
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": name}
    # The role's complete capability registry remains available for later stages.
    assert agent.get_tools() == registry and len(registry) > 1
    assert client.calls[0]["tools"][0] == {
        "name": registry[name].name, "description": registry[name].description,
        "input_schema": registry[name].input_schema,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("role,message,first,args,status", [
    (BillingAgent, "核对DEMO-1004退款状态，并说明退款规则", "read_demo_order", {"order_id": "DEMO-1004"}, "success"),
    (TechnicalAgent, "核对DEMO-1009付款并排查401故障，也说明退款规则", "lookup_error_code", {"error_code": "401"}, "partial"),
    (TechnicalAgent, "DEMO-1003付款后权益开通失败，请技术排障，也说明退款规则", "build_diagnostic_plan",
     {"environment": "浏览器", "reproduced": False}, "partial"),
])
async def test_each_bound_stage_gets_its_own_menu_and_prior_native_result(role, message, first, args, status):
    client = Client(native(block(first, args, "first_1")),
                    native(block("search_knowledge_base", {"query": "退款规则"}, "policy_2")))
    manager, req = Retrieval(), request(message)
    agent = role(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    registry = agent.get_tools()
    result = await agent.handle(req)
    assert result.status == status and result.tools_used == [first, "search_knowledge_base"]
    assert menus(client) == [[first], ["search_knowledge_base"]]
    assert [call["tool_choice"]["name"] for call in client.calls] == [first, "search_knowledge_base"]
    prior_call = client.calls[1]["messages"][-2]["content"][0]
    prior_result = client.calls[1]["messages"][-1]["content"][0]
    assert prior_call == block(first, args, "first_1")
    assert prior_result["type"] == "tool_result" and prior_result["tool_use_id"] == "first_1"
    assert prior_result["is_error"] is False
    manager.search_with_rewrite.assert_awaited_once_with(
        "knowledge_search", message, top_k=5, context={"original_query": message})
    assert agent.get_tools() == registry
    if role is TechnicalAgent:
        # Tool selection succeeds without granting the diagnostic role record
        # access or hiding the mixed request's still-unverified order part.
        req.demo_order_reader.assert_not_awaited()
        assert "订单或交易部分尚未核实" in result.content


class FreeSummaryBilling(BillingAgent):
    """Existing protected free-prose continuation; no new production option."""
    def _project_order_content(self):
        return False


@pytest.mark.asyncio
async def test_genuinely_needed_optional_continuation_restores_full_unchanged_registry():
    client = Client(native(block("read_demo_order", {"order_id": "DEMO-1004"})), text())
    agent = FreeSummaryBilling(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(Retrieval()))
    registry = agent.get_tools()
    result = await agent.handle(request("核对DEMO-1004退款状态"))
    assert result.status == "success" and len(client.calls) == 2
    assert menus(client)[0] == ["read_demo_order"]
    assert menus(client)[1] == list(registry)
    assert "tool_choice" not in client.calls[1]
    assert agent.get_tools() == registry


@pytest.mark.asyncio
async def test_unbound_legal_multi_tool_round_keeps_existing_optional_scope():
    client = Client(native(block("inspect_request_context", {"focus": "咨询"}, "context_1"),
                           block("suggest_required_fields", {}, "fields_2")), text())
    agent, req = GeneralAgent(client, "controlled-model"), request("请帮我整理现有信息")
    agent.set_shared_tools(build_shared_rag_tools(Retrieval()))
    registry = agent.get_tools()
    result = await agent.handle(req)
    assert result.status == "success" and result.tools_used == ["inspect_request_context", "suggest_required_fields"]
    assert menus(client) == [list(registry), list(registry)]
    assert all("tool_choice" not in call for call in client.calls)
    assert [entry["tool_use_id"] for entry in client.calls[1]["messages"][-1]["content"]] == ["context_1", "fields_2"]
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks,error", [
    # The observed failure's shape remains an invalid required single-step batch.
    ([block("lookup_error_code", {"error_code": "401"}),
      block("search_knowledge_base", {"query": "订单DEMO-1009 支付状态 权益核对 重复扣款 退款", "top_k": 5}, "rag_2")], None),
    ([block("lookup_error_code", {"error_code": "401"}),
      block("lookup_error_code", {"error_code": "401"}, "lookup_2")], None),
    ([block("search_knowledge_base", {"query": "退款规则"})], None),
    ([block("unknown_tool", {})], None),
    ([block("lookup_error_code", {"error_code": "401"}),
      block("search_knowledge_base", {"query": "退款规则"})], "agent_protocol_error"),
    ([block("lookup_error_code", {"error_code": "401"}, "invalid id")], "agent_protocol_error"),
    ([block("lookup_error_code", [])], "agent_protocol_error"),
    ([block("lookup_error_code", {"error_code": "401", "extra": "forbidden"})], None),
    ([block("lookup_error_code", {"error_code": "500"})], None),
])
async def test_bad_required_response_is_never_trimmed_repaired_retried_or_dispatched(blocks, error, monkeypatch):
    manager, req, client = Retrieval(), request(), Client(native(*blocks))
    agent = TechnicalAgent(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    registry = agent.get_tools()
    spies = {name: AsyncMock(wraps=spec.handler) for name, spec in registry.items()}
    registry = {name: replace(spec, handler=spies[name]) for name, spec in registry.items()}
    monkeypatch.setattr(agent, "get_tools", lambda: dict(registry))
    result = await agent.handle(req)
    assert result.status in {"partial", "failed"} and result.error_code == error
    assert menus(client) == [["lookup_error_code"]] and len(client.calls) == 1
    assert result.tools_used == req.tool_attempts == []
    assert result.diagnostic_evidence is None
    for spy in spies.values():
        spy.assert_not_awaited()
    manager.search_with_rewrite.assert_not_awaited()
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"order_id": "DEMO-1001"},
    {"order_id": "DEMO-1004", "customer_id": "demo-customer-b"},
])
async def test_read_menu_does_not_weaken_original_target_or_strict_argument_checks(args):
    req, client = request("核对DEMO-1004退款状态"), Client(native(block("read_demo_order", args)))
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and result.tools_used == req.tool_attempts == []
    assert menus(client) == [["read_demo_order"]]
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_double_call", [False, True])
async def test_actual_sdk_serialization_exposes_only_current_lookup_and_preserves_native_rejection(invalid_double_call):
    captured = []
    blocks = [block("lookup_error_code", {"error_code": "401"})]
    if invalid_double_call:
        blocks.append(block("search_knowledge_base", {"query": "订单DEMO-1009 支付状态"}, "rag_2"))

    def send(http_request):
        captured.append(json.loads(http_request.content))
        return httpx.Response(200, json={
            "id": "msg_controlled", "type": "message", "role": "assistant", "model": FLASH_SNAPSHOT,
            "content": blocks, "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    req, manager = request(), Retrieval()
    async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as http:
        client = AsyncAnthropic(api_key="controlled-not-real", base_url="https://dashscope.aliyuncs.com/apps/anthropic/",
                                http_client=http, max_retries=0)
        agent = TechnicalAgent(client, FLASH_SNAPSHOT)
        agent.set_shared_tools(build_shared_rag_tools(manager))
        registry = agent.get_tools()
        result = await agent.handle(req)
    assert len(captured) == 1
    assert captured[0]["tools"] == [{"name": registry["lookup_error_code"].name,
        "description": registry["lookup_error_code"].description,
        "input_schema": registry["lookup_error_code"].input_schema}]
    assert captured[0]["tool_choice"] == {"type": "tool", "name": "lookup_error_code"}
    assert captured[0]["model"] == FLASH_SNAPSHOT
    assert captured[0]["max_tokens"] == agent.profile.max_tokens
    assert captured[0]["temperature"] == agent.profile.temperature
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert result.status == ("partial" if invalid_double_call else "success")
    assert result.tools_used == ([] if invalid_double_call else ["lookup_error_code"])
    manager.search_with_rewrite.assert_not_awaited()
    req.demo_order_reader.assert_not_awaited()
