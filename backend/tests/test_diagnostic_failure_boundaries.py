"""Offline failure boundaries for already-bound native diagnostic requests.

The short deadline below is a test-only substitute for the production clock:
we assert that the actor still requests 15 seconds, then simulate its expiry.
These checks do not establish why any live upstream request timed out or prove
that a real provider/network timeout has been repaired. All order reads are
pure projections of the static demo fixture, never database queries.
"""
import asyncio
from collections import deque
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import AsyncAnthropic
import httpx
import pytest

import agents.agent_orchestrator as runtime
from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, BillingAgent, Request, ResponseComposer,
    TechnicalAgent,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.provider_options import FLASH_SNAPSHOT
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline diagnostic checks forbid socket access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.delenv("ECHOMIND_LLM_TIMEOUT_S", raising=False)


@pytest.fixture
def simulated_deadline(monkeypatch):
    """Cancel an incomplete await quickly; leave the requested production cap intact."""
    requested_limits = []

    async def scaled_await_model(awaitable, timeout_s):
        requested_limits.append(timeout_s)
        return await asyncio.wait_for(awaitable, timeout=0.03)

    monkeypatch.setattr(runtime, "await_model", scaled_await_model)
    return requested_limits


def native(name="lookup_error_code", args=None):
    return SimpleNamespace(content=[{
        "type": "tool_use", "id": "controlled_tool_1", "name": name,
        "input": {"error_code": "403"} if args is None else args,
    }], stop_reason="tool_use")


class ControlledClient:
    def __init__(self, *responses, wait=False):
        self.messages = self
        self.responses, self.calls = list(responses), []
        self.wait = wait
        self.entered, self.cancelled = asyncio.Event(), asyncio.Event()

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        self.entered.set()
        if self.wait:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        assert self.responses, "no automatic retry, repair or summary request allowed"
        return self.responses.pop(0)


class ObservedTechnical(TechnicalAgent):
    """Wrap actual registered handlers solely to observe controlled dispatch."""
    def __init__(self, client, handler=None):
        super().__init__(client, FLASH_SNAPSHOT)
        self.handler_calls = []
        self.handler = handler

    def get_tools(self):
        tools = super().get_tools()
        original = tools["lookup_error_code"]

        async def observed(req, args):
            self.handler_calls.append(deepcopy(args))
            result = (self.handler or original.handler)(req, args)
            return await result if inspect.isawaitable(result) else result

        tools["lookup_error_code"] = replace(original, handler=observed)
        return tools


def request():
    fixture = load_fixture()

    async def fixture_order_reader(customer, identity):
        return fixture_order_result(fixture, customer, identity)

    return Request(
        message=("核对DEMO-1002的付款和预售物流，同时客户端HTTP 403报错，"
                 "请分别说明记录与排查方向。"),
        user_id="controlled-user", conv_id="controlled-conversation",
        request_id="controlled-diagnostic-request",
        intent=IntentCategory.PAYMENT_ISSUE, urgency=UrgencyLevel.LOW,
        intent_confidence=0.99, demo_customer_id="demo-customer-a",
        demo_order_reader=AsyncMock(side_effect=fixture_order_reader),
    )


def assert_required_lookup(payload):
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup_error_code"}
    assert [tool["name"] for tool in payload["tools"]] == ["lookup_error_code"]


def orchestrator(*agents):
    # Only the two concrete specialists are installed. A General fallback is
    # deliberately absent, so no unrelated fallback response masks either role.
    instance = AgentOrchestrator.__new__(AgentOrchestrator)
    instance._pool = {agent.agent_type: [agent] for agent in agents}
    instance._composer = ResponseComposer(ControlledClient(), FLASH_SNAPSHOT)
    instance._recent_tool_traces = deque(maxlen=10)
    return instance


@pytest.mark.asyncio
async def test_bound_diagnostic_expiry_dispatches_no_handler_or_retry(simulated_deadline):
    req, client = request(), ControlledClient(wait=True)
    agent = ObservedTechnical(client)

    result = await agent.handle(req)

    assert simulated_deadline == [15]
    assert not result.success and result.status == "failed"
    assert result.error_code == "agent_timeout"
    assert len(client.calls) == 1 and client.cancelled.is_set()
    assert_required_lookup(client.calls[0])
    assert agent.handler_calls == req.tool_attempts == []
    assert result.tools_used == result.tool_traces == []
    assert result.diagnostic_evidence is None
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_sdk_request_expiry_keeps_native_binding_without_network(simulated_deadline):
    captured, cancelled = [], asyncio.Event()

    async def send(http_request):
        captured.append(json.loads(http_request.content))
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    req = request()
    async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as http:
        client = AsyncAnthropic(
            api_key="controlled-not-real",
            base_url="https://dashscope.aliyuncs.com/apps/anthropic/",
            http_client=http, max_retries=0,
        )
        try:
            agent = ObservedTechnical(client)
            result = await agent.handle(req)
        finally:
            await client.close()

    assert simulated_deadline == [15]
    assert result.status == "failed" and result.error_code == "agent_timeout"
    assert len(captured) == 1 and cancelled.is_set()
    assert_required_lookup(captured[0])
    assert captured[0]["model"] == FLASH_SNAPSHOT
    assert req.message == captured[0]["messages"][-1]["content"]
    assert agent.handler_calls == req.tool_attempts == []
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancel_during_model_wait_does_not_dispatch_or_continue():
    req, client = request(), ControlledClient(wait=True)
    agent = ObservedTechnical(client)
    task = asyncio.create_task(agent.handle(req))
    await asyncio.wait_for(client.entered.wait(), timeout=1)
    task.cancel()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled() and client.cancelled.is_set()
    assert len(client.calls) == 1
    assert_required_lookup(client.calls[0])
    assert agent.handler_calls == req.tool_attempts == []
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_before_actor_starts_sends_nothing():
    req, client = request(), ControlledClient(wait=True)
    agent = ObservedTechnical(client)
    task = asyncio.create_task(agent.handle(req))
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.calls == agent.handler_calls == req.tool_attempts == []
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_billing_cannot_hide_sibling_diagnostic_timeout(simulated_deadline):
    req, technical_client = request(), ControlledClient(wait=True)
    billing_client = ControlledClient(native("read_demo_order", {"order_id": "DEMO-1002"}))
    technical = ObservedTechnical(technical_client)
    billing = BillingAgent(billing_client, FLASH_SNAPSHOT)
    service = orchestrator(billing, technical)

    result = await service.run(req)

    assert sorted(simulated_deadline) == [15, 15]
    assert result.status == "partial" and not result.retryable
    assert {result.primary_agent, *result.supporting_agents} == {AgentType.BILLING, AgentType.TECHNICAL}
    assert result.agent_types == [AgentType.BILLING]
    assert result.tools_used == ["read_demo_order"]
    assert "899.00" in result.response and "paid" in result.response
    assert "preorder" in result.response
    assert "物流单号：未知" in result.response and "预计送达：未知" in result.response
    assert "演示数据" in result.response and "数据截至：" in result.response
    assert "技术排查部分本次未完成；未确认故障原因或修复结果。" in result.response
    assert "权限不足" not in result.response  # No handler result was obtained.
    assert len(technical_client.calls) == len(billing_client.calls) == 1
    assert technical_client.cancelled.is_set() and technical.handler_calls == []
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1002")
    assert service._composer._client.calls == []
    assert service.get_tool_trace(req.request_id)["status"] == "partial"


@pytest.mark.asyncio
async def test_parallel_cancel_does_not_turn_finished_sibling_into_complete_answer():
    req, technical_client = request(), ControlledClient(wait=True)
    billing_client = ControlledClient(native("read_demo_order", {"order_id": "DEMO-1002"}))
    technical = ObservedTechnical(technical_client)
    service = orchestrator(BillingAgent(billing_client, FLASH_SNAPSHOT), technical)
    task = asyncio.create_task(service.run(req))
    await asyncio.wait_for(technical_client.entered.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled() and technical_client.cancelled.is_set()
    assert len(technical_client.calls) == len(billing_client.calls) == 1
    assert technical.handler_calls == []
    assert req.demo_order_reader.await_count <= 1
    assert service._composer._client.calls == []
    assert service.get_tool_trace(req.request_id) is None


@pytest.mark.asyncio
async def test_unavailable_diagnostic_handler_remains_partial_not_verified_fact():
    def unavailable(req, args):
        raise RuntimeError("controlled local diagnostic unavailable")

    req, client = request(), ControlledClient(native())
    agent = ObservedTechnical(client, unavailable)
    result = await agent.handle(req)

    assert result.success and result.status == "partial"
    assert len(client.calls) == 1 and agent.handler_calls == [{"error_code": "403"}]
    assert result.tools_used == []
    assert req.tool_attempts == [{"tool_name": "lookup_error_code", "retry_safe": True}]
    assert result.diagnostic_evidence.status == "invalid"
    assert len(result.tool_traces) == 1 and result.tool_traces[0]["success"] is False
    assert "技术工具结果未通过校验" in result.content
    assert "未执行后台修复" in result.content and "权限不足" not in result.content
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_during_handler_does_not_create_diagnostic_evidence_or_next_send():
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def wait_for_tool(req, args):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    req, client = request(), ControlledClient(native())
    agent = ObservedTechnical(client, wait_for_tool)
    task = asyncio.create_task(agent.handle(req))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set() and len(client.calls) == 1
    assert agent.handler_calls == [{"error_code": "403"}]
    # An entered attempt is not silently removed or represented as completion.
    assert req.tool_attempts == [{"tool_name": "lookup_error_code", "retry_safe": True}]
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_received_typed_diagnostic_needs_no_unbounded_summary_request():
    req, client = request(), ControlledClient(native())
    agent = ObservedTechnical(client)
    result = await agent.handle(req)

    assert result.success and result.status == "success"
    assert len(client.calls) == 1 and agent.handler_calls == [{"error_code": "403"}]
    assert result.tools_used == ["lookup_error_code"]
    assert result.diagnostic_evidence.status == "suggestions"
    assert "本地工具提供的排查建议，不代表已查询后台、确认根因或完成修复。" in result.content
    assert "权限不足" in result.content and "未经系统核验" in result.content
    assert result.order_evidence is None
    req.demo_order_reader.assert_not_awaited()
