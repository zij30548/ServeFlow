"""Current-question diagnostic binding; controlled native replies, no services."""
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import socket

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, GeneralAgent, Request,
    ResponseComposer, TechnicalAgent,
)
from core.diagnostic_evidence import select_diagnostic_request
from core.intent_recognizer import IntentCategory


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("diagnostic binding regression is offline")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.mark.parametrize("message,code", [
    ("团队接口返回403。没有后台权限，请解释通常含义和先核对哪些权限。", "403"),
    ("不用查订单，只解释当前HTTP 403。", "403"),
    ("接口提示“403”，怎么排查？", "403"),
    ("API returns '403 Forbidden', what should I check?", "403"),
    ("HTTP401是什么含义？", "401"),
    ("接口错误码为403，请解释含义。", "403"),
    ("看到错误码是403，该怎么排查？", "403"),
    ("请识别错误码403。", "403"),
    ("请求返回404，先检查哪些项目？", "404"),
    ("App调用接口出现500，网络已确认正常并已重启，下一步怎么办？", "500"),
    ("DEMO-1002已付款，客户端报403，请分别核对。", "403"),
    ("设备型号403，接口返回500，请排查。", "500"),
    ("应用一直崩溃，请给出排查建议。", None),
    ("客户端登录失败，怎么诊断？", None),
    ("HTTP401或403应该检查什么？", None),
    ("接口分别返回401和403，请分别说明两个错误含义。", None),
    ("分别解释HTTP401和403。", None),
    ("HTTP 401/403怎么排查？", None),
    ("HTTP403和502有何不同，如何检查？", None),
    ("HTTP502，请提供排查方向。", None),
    ("不是HTTP403，而是HTTP500，请排查。", "500"),
])
def test_explicit_diagnostic_queries_bind_only_one_contextual_supported_code(message, code):
    selected = select_diagnostic_request(message)
    assert selected.required is True and selected.error_code == code


@pytest.mark.parametrize("message", [
    None, {}, [], 403, "", "x" * 2001,
    "订单价格403元，查一下物流。",
    "型号403的设备多少钱？",
    "第403页有个标题。",
    "DEMO-0403的订单价格是多少？",
    "把HTTP403翻译成英文。",
    "文档写了“接口返回403”，我只是引用。",
    "客服说‘403 Forbidden’，没有提出检查请求。",
    "不是来问HTTP403，只查退款。",
    "不用解释错误码403，我问订单。",
    "以前接口403已恢复，现在没有问题。",
    "Do not explain HTTP403, I only need the invoice.",
    "请解释‘技术排障’这个词。",
    "我这次不是返回403，只查订单状态。",
    "接口没有返回403，只想知道正常回包格式。",
    "文档里包含HTTP403这个状态码。",
    "HTTP403",
    "HTTP4030，请解释。",
    "HTTP403abc，请解释。",
    "错误码403.2，请解释。",
    "别解释HTTP403，我只查账单。",
    "你别查错误码403，我只问发票。",
])
def test_numbers_declined_quotes_translations_and_resolved_topics_do_not_grant_lookup(message):
    selected = select_diagnostic_request(message)
    assert selected.required is False and selected.error_code is None


class Client:
    def __init__(self, *replies):
        self.messages, self.calls, self.replies = self, [], list(replies)

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.replies, "no hidden retry or summary request"
        return self.replies.pop(0)


def native(name="lookup_error_code", args=None, identity="diagnostic_native", prose=None):
    blocks = [{"type": "tool_use", "id": identity, "name": name,
               "input": {"error_code": "403"} if args is None else args}]
    if prose:
        blocks.insert(0, {"type": "text", "text": prose})
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


def request(message="接口返回403，请解释含义并说明先核对哪些权限。"):
    return Request(message=message, user_id="controlled-user", conv_id="controlled-diagnostic",
        intent=IntentCategory.TECHNICAL, intent_confidence=.99,
        entities={"error_codes": ["401"]}, demo_order_reader=AsyncMock())


@pytest.mark.asyncio
async def test_standalone_technical_query_requires_real_native_lookup_then_program_display():
    req = request()
    client = Client(native(prose="已查服务器，权限已全部恢复。"))
    agent = TechnicalAgent(client, "controlled-model")
    registry = agent.get_tools()
    result = await agent.handle(req)
    assert len(client.calls) == 1
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "lookup_error_code"}
    assert [t["name"] for t in client.calls[0]["tools"]] == ["lookup_error_code"]
    assert result.status == "success" and result.tools_used == ["lookup_error_code"]
    assert result.diagnostic_evidence.units[0].error_code == "403"
    assert "参考含义：权限不足" in result.content and "账号或套餐权限" in result.content
    assert "不代表已查询后台、确认根因或完成修复" in result.content
    assert "已查服务器" not in result.content and "权限已全部恢复" not in result.content
    req.demo_order_reader.assert_not_awaited()
    assert agent.get_tools() == registry and "read_demo_order" not in registry


@pytest.mark.asyncio
async def test_old_free_text_shape_now_ends_partial_without_guessing_tool_or_repair():
    raw = "已检查后台，时间窗口就是5分钟，已修复。"
    client = Client(SimpleNamespace(content=[{"type": "text", "text": raw}], stop_reason="end_turn"))
    req = request()
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert len(client.calls) == 1 and result.tools_used == []
    assert result.status == "partial" and result.diagnostic_evidence is None
    assert "技术排查部分本次未完成" in result.content and raw not in result.content
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    native(args={"error_code": "401"}),
    native(args={"error_code": "403", "extra": "discard me"}),
    native("build_diagnostic_plan", {"environment": "App", "reproduced": True}),
    SimpleNamespace(content=[
        {"type": "tool_use", "id": "first", "name": "lookup_error_code", "input": {"error_code": "403"}},
        {"type": "tool_use", "id": "second", "name": "lookup_error_code", "input": {"error_code": "403"}},
    ], stop_reason="tool_use"),
    SimpleNamespace(content=[{"type": "tool_use", "id": "cut", "name": "lookup_error_code", "input": {"error_code": "403"}}], stop_reason="max_tokens"),
])
async def test_bad_bound_selection_never_dispatches_or_retries(reply):
    client, req = Client(reply), request()
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert len(client.calls) == 1 and result.status in {"partial", "failed"}
    assert result.tools_used == [] and not result.tool_traces
    assert result.diagnostic_evidence is None
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("message,status", [
    ("接口HTTP401或403，如何排查？", "partial"),
    ("接口返回502，请说明具体含义。", "partial"),
    ("应用崩溃，请给出排查建议。", "success"),
])
async def test_existing_plan_separates_generic_advice_from_unanswered_code_meanings(message, status):
    client = Client(native("build_diagnostic_plan", {"environment": "App", "reproduced": False}))
    result = await TechnicalAgent(client, "controlled-model").handle(request(message))
    assert client.calls[0]["tool_choice"]["name"] == "build_diagnostic_plan"
    assert result.status == status and result.tools_used == ["build_diagnostic_plan"]
    assert result.diagnostic_evidence.units[0].error_code is None
    assert ("尚未逐项核对" in result.content) == (status == "partial")
    assert ("仅为通用排查建议" in result.content) == (status == "partial")


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeded", [True, False])
async def test_unverified_sibling_or_fallback_cannot_complete_technical_request(succeeded):
    client = Client()
    response = AgentResponse(AgentType.GENERAL, "权限已恢复，已查后台。", succeeded,
                            status="success" if succeeded else "failed")
    result = await ResponseComposer(client, "controlled-model").compose_result(request(), [response])
    assert result.status == ("partial" if succeeded else "failed")
    assert "技术排查部分本次未完成" in result.content
    assert "权限已恢复" not in result.content and "已查后台。" not in result.content
    assert client.calls == []


@pytest.mark.asyncio
async def test_single_general_route_without_evidence_stays_unknown_not_false_success():
    client = Client(SimpleNamespace(content=[{"type": "text", "text": "已修复后台。"}], stop_reason="end_turn"))
    general = GeneralAgent(client, "controlled-model")
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {AgentType.GENERAL: [general]}
    orch._composer = ResponseComposer(Client(), "controlled-model")
    orch._recent_tool_traces = deque(maxlen=5)
    req = request()
    result = await orch.run(req)
    assert result.status == "partial" and "技术排查部分本次未完成" in result.response
    assert "已修复后台" not in result.response and result.tools_used == []
    assert len(client.calls) == 1
