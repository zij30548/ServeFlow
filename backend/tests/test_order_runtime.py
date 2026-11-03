"""Controlled native execution and typed facts; not a live quality score."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import socket

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BillingAgent, GeneralAgent, Request,
    ResponseComposer, TechnicalAgent,
)
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory
from core.order_evidence import extract_order_evidence, render_order_evidence
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("controlled test must not contact external services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def native(name="read_demo_order", args=None, identity="tool_read_1", extra=None):
    blocks = [{"type": "tool_use", "id": identity, "name": name,
               "input": args or {"order_id": "DEMO-1004"}}]
    if extra:
        blocks.insert(0, {"type": "text", "text": extra})
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


class Client:
    def __init__(self, *responses):
        self.calls, self.responses, self.messages = [], list(responses), self

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "no model continuation or auto-retry allowed"
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def request(message="查DEMO-1004退款金额和到账日期", **changes):
    fixture = load_fixture()
    async def reader(customer, order):
        return fixture_order_result(fixture, customer, order)
    values = dict(message=message, user_id="untrusted", conv_id="controlled",
                  demo_customer_id="demo-customer-a", demo_order_reader=AsyncMock(side_effect=reader),
                  intent=IntentCategory.REFUND)
    values.update(changes)
    return Request(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [BillingAgent, GeneralAgent])
async def test_refund_projects_actual_record_not_free_actor_financial_claims(role):
    client = Client(native(extra="需扣手续费，原路退款，到账后自动通知"))
    req = request()
    result = await role(client, "controlled-model").handle(req)
    assert result.success and result.status == "success"
    assert len(client.calls) == 1 and result.tools_used == ["read_demo_order"]
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "read_demo_order"}
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1004")
    assert result.order_evidence.status == "found"
    assert result.content == render_order_evidence(result.order_evidence)
    assert "pending_review" in result.content and "记录金额：未知" in result.content
    assert "到账时间未知" in result.content and "数据截至" in result.content
    assert all(text not in result.content for text in ("手续费", "原路退款", "自动通知"))
    assert not result.escalate


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    native(args={"order_id": "DEMO-1001"}),
    native(args={"order_id": "DEMO-1004", "customer_id": "demo-customer-b"}),
    native(name="inspect_request_context", args={"focus": "refund"}),
    SimpleNamespace(content=[{"type": "text", "text": "已退款299元"}], stop_reason="end_turn"),
])
async def test_wrong_tool_or_target_never_reads_or_accepts_free_answer(response):
    req, client = request(), Client(response)
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and "已退款299元" not in result.content
    req.demo_order_reader.assert_not_awaited()
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_two_orders_need_clarification_and_never_choose_the_first():
    req = request("查DEMO-1001和DEMO-1004的状态")
    client = Client()
    result = await GeneralAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and "一个有效" in result.content
    assert client.calls == []
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_followup_reads_current_record_again_and_failed_read_cannot_use_old_answer():
    history = [{"role": "user", "content": "查询DEMO-1003的订阅"},
               {"role": "assistant", "content": "旧回复：有效期为2099年，不能作为本次事实"}]
    req = request("那个会员权益到什么时候有效？", history=history)
    client = Client(native(args={"order_id": "DEMO-1003"}))
    result = await GeneralAgent(client, "controlled-model").handle(req)
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1003")
    assert "有效期截至：未知" in result.content and "2099" not in result.content
    req.demo_order_reader = AsyncMock(side_effect=RuntimeError("private database failure"))
    failed = await GeneralAgent(Client(native(args={"order_id": "DEMO-1003"})), "controlled-model").handle(req)
    assert failed.status == "partial" and "本次读取未完成" in failed.content
    assert "2099" not in failed.content and "private database" not in failed.content


@pytest.mark.asyncio
async def test_cross_owner_not_found_is_a_usable_non_disclosing_result():
    req = request("查DEMO-1005", entities={"customer_id": ["demo-customer-b"]})
    result = await GeneralAgent(Client(native(args={"order_id": "DEMO-1005"})), "controlled-model").handle(req)
    assert result.status == "success" and result.order_evidence.status == "not_found"
    assert "当前演示身份" in result.content and "专属设备" not in result.content
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1005")


class Retrieval:
    def __init__(self): self.calls = []
    async def search_with_rewrite(self, name, query, top_k=3):
        self.calls.append((name, query))
        return SimpleNamespace(success=True, reranked=False, error=None, data=[{
            "document_id": "a" * 64, "title": "受控规则", "content": "仅作受控政策依据，不是订单状态。",
            "version": "b" * 64, "chunk": 0, "score": .9,
        }])


@pytest.mark.asyncio
async def test_mixed_policy_and_order_keep_both_actual_evidence_channels():
    req = request("核对DEMO-1004退款状态，并说明退款规则")
    client = Client(native(), native("search_knowledge_base", {"query": "退款规则"}, "policy_2"))
    manager = Retrieval()
    agent = BillingAgent(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    result = await agent.handle(req)
    assert result.success and result.status == "success"
    assert result.tools_used == ["read_demo_order", "search_knowledge_base"]
    assert len(client.calls) == 2 and len(manager.calls) == 1
    # The second forced tool receives the first native result, not a gold answer.
    assert client.calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert "pending_review" in result.content and "仅作受控政策依据" in result.content
    assert "订单或交易部分尚未核实" not in result.content


@pytest.mark.asyncio
async def test_composer_does_not_rewrite_records_or_hide_failed_technical_role():
    req = request("DEMO-1004退款状态，并排查登录401", intent=IntentCategory.TECHNICAL_LOGIN)
    record = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1004")
    evidence = extract_order_evidence(record, expected_order_id="DEMO-1004", trusted_customer_id="demo-customer-a")
    responses = [AgentResponse(AgentType.BILLING, "原路退款成功", True, order_evidence=evidence),
                 AgentResponse(AgentType.TECHNICAL, "已修复", False, status="failed")]
    client = Client()
    result = await ResponseComposer(client, "controlled-model").compose_result(req, responses)
    assert result.status == "partial" and client.calls == []
    assert render_order_evidence(evidence) in result.content
    assert "技术排查部分本次未完成" in result.content
    assert "原路退款成功" not in result.content and "已修复" not in result.content


@pytest.mark.asyncio
async def test_technical_role_returns_only_checked_local_diagnostic_and_no_financial_prose():
    req = request("DEMO-1003付款了但权益开通失败，请技术排障，网络已检查过")
    client = Client(native("build_diagnostic_plan", {"environment": "invented backend", "reproduced": True}, "diag_1",
                           extra="后台已修复，退款明天到账"))
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert result.status == "success" and result.diagnostic_evidence.status == "suggestions"
    assert len(client.calls) == 1 and "不能" not in str(client.calls[0]["tool_choice"])
    assert "不代表已查询后台" in result.content
    assert all(text not in result.content for text in ("invented backend", "退款明天到账", "后台已修复"))
    assert "确认网络、" not in result.content
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_read_never_becomes_a_success_or_second_send():
    req = request(demo_order_reader=AsyncMock(side_effect=asyncio.CancelledError))
    client = Client(native())
    with pytest.raises(asyncio.CancelledError):
        await BillingAgent(client, "controlled-model").handle(req)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_order_execution_request_remains_partial_even_when_read_succeeds():
    req = request("查询DEMO-1010，并现在直接替我退款")
    result = await BillingAgent(Client(native(args={"order_id": "DEMO-1010"})), "controlled-model").handle(req)
    assert result.status == "partial" and "不能执行退款" in result.content
    assert "active" in result.content and "39.00" in result.content


@pytest.mark.asyncio
async def test_unrequested_model_read_never_dispatches_or_becomes_a_record_answer():
    req = request("你好", intent=IntentCategory.GREETING)
    client = Client(native())
    result = await GeneralAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and "未读取订单" in result.content
    req.demo_order_reader.assert_not_awaited()
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["帮我查DEMO-1004退款状态", "现在DEMO-1004退款是什么状态？"])
async def test_query_language_is_not_confused_with_execution(question):
    result = await BillingAgent(Client(native()), "controlled-model").handle(request(question))
    assert result.status == "success" and "pending_review" in result.content
    assert "这些操作尚未完成" not in result.content


@pytest.mark.asyncio
async def test_missing_technical_role_keeps_verified_record_but_cannot_claim_complete():
    from collections import deque
    req = request("核对DEMO-1003付款和权益，并做技术排障", intent=IntentCategory.PAYMENT_ISSUE)
    client = Client(native(args={"order_id": "DEMO-1003"}))
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {AgentType.GENERAL: [GeneralAgent(client, "controlled-model")],
                  AgentType.BILLING: [BillingAgent(client, "controlled-model")],
                  AgentType.TECHNICAL: []}
    orch._composer = ResponseComposer(client, "controlled-model")
    orch._recent_tool_traces = deque(maxlen=10)
    result = await orch.run(req)
    assert result.status == "partial" and "技术排查部分本次未完成" in result.response
    assert "activation_failed" in result.response and "39.00" in result.response
    assert len(client.calls) == 1 and result.tools_used == ["read_demo_order"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["401", "403", "404", "500"])
async def test_original_unique_code_binds_technical_lookup_without_read_permission(code):
    req = request(f"核对DEMO-1009，并排查接口{code}故障")
    result = await TechnicalAgent(Client(native("lookup_error_code", {"error_code": code})), "controlled-model").handle(req)
    assert result.status == "success" and result.tools_used == ["lookup_error_code"]
    assert result.diagnostic_evidence.units[0].error_code == code
    assert "未经系统核验" in result.content
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrong_code_is_not_corrected_or_executed_as_another_diagnostic():
    req = request("DEMO-1009返回401，请核对付款并排查认证")
    client = Client(native("lookup_error_code", {"error_code": "500"}))
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and result.tools_used == []
    assert len(client.calls) == 1 and result.diagnostic_evidence is None
    assert "技术排查部分本次未完成" in result.content
