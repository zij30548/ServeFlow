"""Workflow routing and execution with controlled data and no external calls."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agents.agent_orchestrator import AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory, UrgencyLevel
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result
from workflows.hybrid_router import dispatch_chat, select_workflow


def request(message, **fields):
    return Request(message, "customer-workspace", "conversation", **fields)


@pytest.mark.parametrize("message, expected", [
    ("帮我查 DEMO-1004 的退款到账了吗？", ("order_lookup", None)),
    ("订单号 DEMO-1004 的退款还没到账，帮我查一下。", ("order_lookup", None)),
    ("我的订单状态是什么？", ("order_lookup", None)),
    ("退款状态", ("order_lookup", None)),
    ("登录报 401，同时这笔订单好像扣了两次。", ("order_lookup", "登录报 401")),
    ("查看 DEMO-1004 的退款状态，而且登录报 401", ("order_lookup", "登录报 401")),
    ("状态", (None, None)),
    ("退款政策怎么规定？", (None, None)),
    ("登录报401，我想了解哪些会员权益", (None, None)),
    ("查 DEMO-1004 的退款状态，再告诉我退货规则", (None, None)),
    ("帮我查一下DEMO-1004和DEMO-1005", (None, None)),
    ("别查订单还是转人工吧", (None, None)),
    ("帮我取消订单 DEMO-1004", (None, None)),
    ("对方说「请查DEMO-1004」，这是什么风险？", (None, None)),
])
def test_only_complete_supported_requests_enter_workflow(message, expected):
    assert select_workflow(request(message)) == expected


@pytest.mark.asyncio
async def test_existing_agent_path_handles_open_requests_and_critical_priority():
    delegate = SimpleNamespace(run=AsyncMock(return_value=object()))
    for req in (request("登录报401怎么继续查"), request("查询DEMO-1004", urgency=UrgencyLevel.CRITICAL),
                request("查询DEMO-1004", intent=IntentCategory.HUMAN_HANDOFF)):
        assert await dispatch_chat(delegate, req) is delegate.run.return_value
        delegate.run.assert_awaited_with(req)


@pytest.mark.asyncio
async def test_untrusted_entities_cannot_supply_order_and_wrong_record_is_rejected():
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._recent_tool_traces = deque()
    snapshot = load_fixture()
    reader = AsyncMock(return_value=fixture_order_result(snapshot, "demo-customer-b", "DEMO-1005"))
    missing = request("我的退款状态", entities={"order_id": ["DEMO-1005"]},
                      demo_customer_id="demo-customer-a", demo_order_reader=reader)
    result = await dispatch_chat(orch, missing)
    assert result.status == "partial" and "一个有效演示订单号" in result.response
    reader.assert_not_awaited()
    req = request("查询DEMO-1005", demo_customer_id="demo-customer-a", demo_order_reader=reader)
    result = await dispatch_chat(orch, req)
    reader.assert_awaited_once_with("demo-customer-a", "DEMO-1005")
    assert result.status == "partial" and "本次读取未完成" in result.response
    assert "demo-customer-b" not in result.response and "699.00" not in result.response
    assert not result.tool_traces[0]["result_success"]


@pytest.mark.asyncio
async def test_cancelled_hybrid_request_joins_both_branches_without_publishing():
    read_started, technical_started = asyncio.Event(), asyncio.Event()
    read_done, technical_done = asyncio.Event(), asyncio.Event()

    async def block(started, done):
        started.set()
        try:
            await asyncio.Future()
        finally:
            done.set()

    async def read(*args):
        await block(read_started, read_done)

    async def technical(*args):
        await block(technical_started, technical_done)

    orch = SimpleNamespace(run_technical_branch=technical, _record_tool_trace=Mock())
    req = request("登录报401，同时查询DEMO-1004", demo_customer_id="demo-customer-a", demo_order_reader=read)
    task = asyncio.create_task(dispatch_chat(orch, req))
    try:
        async with asyncio.timeout(2):
            await read_started.wait()
            await technical_started.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert read_done.is_set() and technical_done.is_set()
    orch._record_tool_trace.assert_not_called()
