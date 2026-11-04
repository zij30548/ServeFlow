"""Question-directed order explanations through real controlled runtime paths.

The SDK is real but its transport is in-memory. Order reads use copied synthetic
fixtures, never a database. This is wiring/safety evidence, not model quality or
new scores for the immutable historical trials.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import re
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anthropic import AsyncAnthropic
import httpx
import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BillingAgent, GeneralAgent,
    Request, ResponseComposer, TechnicalAgent,
)
from core.intent_recognizer import IntentCategory
from core.order_evidence import extract_order_evidence, render_order_evidence
from core.provider_options import FLASH_SNAPSHOT
from demo_business.store import load_fixture
from evaluation import benchmark as bench
from evaluation.quality_arm_runtime import _ArmBilling, _ArmComposer, _ArmState, policy_for
from evaluation.task_evidence import fixture_order_result


ANSWER_HEADING = "针对本次问题的只读说明"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("order answer integration forbids external services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    # No old case/review/protocol is read or modified by these tests.
    paths = [bench.BACKEND_ROOT/"demo_business/fixtures.json",
             bench.BACKEND_ROOT/"evaluation/benchmark/diagnostic-repair-v1.json"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    yield
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


def frozen_question(case_id):
    path = bench.BACKEND_ROOT/"evaluation/benchmark/diagnostic-repair-v1.json"
    return next(row["input"]["message"] for row in bench.read_json(path)["cases"] if row["id"] == case_id)


def native(name="read_demo_order", args=None, identity="controlled_order", prose=None):
    blocks = [{"type": "tool_use", "id": identity, "name": name,
               "input": {"order_id": "DEMO-1007"} if args is None else args}]
    if prose is not None:
        blocks.insert(0, {"type": "text", "text": prose})
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


def text(value):
    return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")


class Client:
    def __init__(self, *responses):
        self.messages, self.calls, self.responses = self, [], list(responses)

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "unexpected extra model continuation, summary or retry"
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def request(message, *, customer="demo-customer-a", intent=IntentCategory.PAYMENT_ISSUE,
            reader=None, history=None):
    snapshot = load_fixture()
    async def read(identity, order_id):
        return fixture_order_result(snapshot, identity, order_id)
    return Request(message=message, user_id="controlled-order-answer-user",
        conv_id="controlled-order-answer-conversation", request_id="controlled-order-answer-request",
        intent=intent, intent_confidence=.99, history=history,
        entities={}, demo_customer_id=customer,
        demo_order_reader=reader if reader is not None else AsyncMock(side_effect=read))


def orchestrator(agents, composer=None):
    value = AgentOrchestrator.__new__(AgentOrchestrator)
    value._pool = {agent.agent_type: [agent] for agent in agents}
    value._composer = composer or ResponseComposer(Client(), "controlled-model")
    value._recent_tool_traces = deque(maxlen=10)
    return value


def assert_dispatch_unknown(answer):
    assert ANSWER_HEADING in answer
    assert re.search(r"(?:无法|不能).{0,24}(?:确定|判断).{0,30}(?:出库|发货)", answer)
    assert "当前快照未提供独立的实际出库或发货时间" in answer


def assert_payment_does_not_prove_delivery_or_permission(answer):
    assert ANSWER_HEADING in answer
    assert re.search(r"(?:付款|支付).{0,20}(?:不能|不等于|不代表).{0,30}(?:发货|出库)", answer)
    assert re.search(r"(?:付款|支付).{0,20}(?:不能|不等于|不代表).{0,30}接口权限", answer)
    assert re.search(r"(?:未|没有)查询或核验实际接口权限", answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    frozen_question("p01"), frozen_question("n02"),
    "查DEMO-1007，支付和物流记录都是空的吗？现有快照能确定哪天发货吗？别把它说成订单已取消。",
])
async def test_actual_role_main_and_composer_answer_unknown_dispatch_without_more_models(message):
    req = request(message)
    client = Client(native(prose="已取消，明天必定出库。"))
    composer_client = Client()
    runtime = orchestrator([BillingAgent(client, "controlled-model")],
                           ResponseComposer(composer_client, "controlled-model"))
    result = await runtime.run(req)
    assert result.status == "success" and result.tools_used == ["read_demo_order"]
    assert len(client.calls) == 1 and composer_client.calls == []
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1007")
    assert_dispatch_unknown(result.response)
    assert result.response.count(ANSWER_HEADING) == 1
    assert "159.00" in result.response and "未付款（unpaid）" in result.response
    assert "当前快照未提供付款记录" in result.response
    assert "当前快照未提供包裹运输记录" in result.response
    assert "订单取消状态：未知" in result.response
    assert "2026-10-25T00:00:00Z" in result.response
    assert "2026-10-24T10:00:00Z" in result.response
    assert "合成演示" in result.response and "serveflow-mysql-demo-v1" in result.response
    assert "明天必定出库" not in result.response
    assert runtime.get_tool_trace(req.request_id)["status"] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ["p09", "n03", "new-expression", "separate-explanations"])
async def test_actual_sdk_native_compound_flow_keeps_facts_and_answers_inferences(case_id):
    captured = []
    async def send(http_request):
        payload = json.loads(http_request.content)
        captured.append(payload)
        name = payload["tool_choice"]["name"]
        args = {"order_id": "DEMO-1002"} if name == "read_demo_order" else {"error_code": "403"}
        assert name in {"read_demo_order", "lookup_error_code"}
        return httpx.Response(200, json={"id": "controlled-sdk-response", "type": "message",
            "role": "assistant", "model": FLASH_SNAPSHOT,
            "content": [{"type": "text", "text": "付款89900元，已出库且权限恢复。"},
                        {"type": "tool_use", "id": "actual_native_"+name, "name": name, "input": args}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})
    if case_id == "new-expression":
        message = ("核对DEMO-1002：付款凭证是否足以证明包裹已经发货、API权限也已经开通？"
                   "客户端接口返回403，请把记录事实和排查建议分开。")
    elif case_id == "separate-explanations":
        message = "查询DEMO-1002，请分别说明付款记录能否证明已经发货和接口权限已开通，接口返回403。"
    else:
        message = frozen_question(case_id)
    req = request(message, intent=IntentCategory.ORDER_STATUS)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as http:
        sdk = AsyncAnthropic(api_key="controlled-not-a-real-key",
            base_url="https://dashscope.aliyuncs.com/apps/anthropic", max_retries=0, http_client=http)
        try:
            # Deliberately keep a coarse order intent and no extracted error
            # entity: explicit current diagnostic demand must still collaborate.
            runtime = orchestrator([GeneralAgent(sdk, FLASH_SNAPSHOT), TechnicalAgent(sdk, FLASH_SNAPSHOT)])
            result = await runtime.run(req)
        finally:
            await sdk.close()
    assert result.status == "success"
    assert set(result.agent_types) == {AgentType.GENERAL, AgentType.TECHNICAL}
    assert set(result.tools_used) == {"read_demo_order", "lookup_error_code"}
    assert len(captured) == 2  # No answer-writing model or Composer continuation.
    assert {payload["tool_choice"]["name"] for payload in captured} == set(result.tools_used)
    assert all(len(payload["tools"]) == 1 and payload["model"] == FLASH_SNAPSHOT
               and payload["thinking"] == {"type": "disabled"} for payload in captured)
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1002")
    assert_payment_does_not_prove_delivery_or_permission(result.response)
    assert result.response.count(ANSWER_HEADING) == 1
    for source_fact in ("899.00", "预售（preorder）", "物流单号：未知", "预计送达：未知",
                        "2026-10-25T00:00:00Z", "2026-10-24T09:10:00Z", "参考含义：权限不足"):
        assert source_fact in result.response
    assert "付款89900元" not in result.response and "已出库且权限恢复" not in result.response
    assert len(result.tool_traces) == 2 and all(trace["success"] for trace in result.tool_traces)


@pytest.mark.asyncio
async def test_missing_technical_keeps_order_answer_but_not_a_complete_permission_check():
    req = request(frozen_question("n03"))
    client = Client(native(args={"order_id": "DEMO-1002"}))
    runtime = orchestrator([BillingAgent(client, "controlled-model")])
    result = await runtime.run(req)
    assert result.status == "partial" and result.tools_used == ["read_demo_order"]
    assert len(client.calls) == 1
    assert_payment_does_not_prove_delivery_or_permission(result.response)
    assert "技术排查部分本次未完成" in result.response and "899.00" in result.response
    assert "参考含义：权限不足" not in result.response


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["provider_text", "wrong_target", "reader_failure", "foreign_identity", "invalid_owner"])
async def test_no_valid_current_evidence_cannot_invent_question_answers(kind):
    message = "查询DEMO-1007，现有记录可以确定哪天出库吗？"
    req = request(message, customer="demo-customer-b" if kind == "foreign_identity" else "demo-customer-a")
    reply = native()
    if kind == "provider_text": reply = text("159元，明天出库，我已查到。")
    if kind == "wrong_target": reply = native(args={"order_id": "DEMO-1002"})
    if kind == "reader_failure": req.demo_order_reader = AsyncMock(side_effect=RuntimeError("private-db-path-must-not-appear"))
    if kind == "invalid_owner":
        bad = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1007")
        bad["order"]["customer_id"] = "demo-customer-b"
        req.demo_order_reader = AsyncMock(return_value=bad)
    client = Client(reply)
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert len(client.calls) == 1 and ANSWER_HEADING not in result.content
    assert "明天出库" not in result.content and "private-db-path" not in result.content
    if kind in {"provider_text", "wrong_target"}:
        req.demo_order_reader.assert_not_awaited()
    if kind == "foreign_identity":
        req.demo_order_reader.assert_awaited_once_with("demo-customer-b", "DEMO-1007")
        assert "当前演示身份" in result.content
        assert "159.00" not in result.content and "智能插座" not in result.content
    else:
        assert result.status == "partial"


@pytest.mark.asyncio
async def test_unknown_dispatch_explanation_does_not_complete_requested_cancellation():
    req = request("核对DEMO-1007现有记录能否确定哪天出库，然后请替我取消订单。")
    client = Client(native())
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and len(client.calls) == 1
    assert_dispatch_unknown(result.content)
    assert "订单取消状态：未知" in result.content and "这些操作尚未完成" in result.content
    assert "已经取消订单" not in result.content


@pytest.mark.asyncio
async def test_unrelated_refund_query_retains_ordinary_record_without_new_shipping_summary():
    req = request("只查询DEMO-1004退款审核状态和退款金额。", intent=IntentCategory.REFUND)
    client = Client(native(args={"order_id": "DEMO-1004"}))
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert result.status == "success" and len(client.calls) == 1
    assert ANSWER_HEADING not in result.content
    assert result.content == render_order_evidence(result.order_evidence)
    assert "记录金额：未知" in result.content and "pending_review" in result.content


@pytest.mark.asyncio
async def test_failed_composition_does_not_gain_answer_or_get_upgraded_to_partial_success():
    req = request("查询DEMO-1007现有记录可以确定哪天出库吗？")
    client = Client()
    result = await ResponseComposer(client, "controlled-model").compose_result(req, [
        AgentResponse(AgentType.BILLING, "159元，明天已经安排出库。", False, status="failed")])
    assert result.status == "failed" and client.calls == []
    assert ANSWER_HEADING not in result.content
    assert "明天已经安排出库" not in result.content and "159元" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("mutation", [False, True])
async def test_free_order_experiment_never_receives_new_answer_even_after_mutation_guard(arm, mutation):
    message = "核对DEMO-1007，当前记录能否确定哪天出库？"
    if mutation: message += "然后请替我取消订单。"
    req = request(message)
    client = Client(native(), text("受控自由摘要，仅用于旧实验通道。"))
    agent = _ArmBilling(client, "controlled-model")
    agent._arm_state = _ArmState(policy_for(arm))
    result = await agent.handle(req)
    assert len(client.calls) == 2 and ANSWER_HEADING not in result.content
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1007")
    assert result.status == ("partial" if mutation else "success")
    if mutation: assert "这些操作尚未完成" in result.content


@pytest.mark.asyncio
async def test_c_historical_record_does_not_gain_current_question_projection():
    req = request("核对DEMO-1007，当前记录能否确定哪天出库？")
    raw = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1007")
    evidence = extract_order_evidence(raw, expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    client = Client(text("受控继续处理。"))
    agent = _ArmBilling(client, "controlled-model")
    agent._arm_state = _ArmState(policy_for("C"))
    agent._remember_order_evidence(req, "DEMO-1007", evidence)
    result = await agent.handle(req)
    req.demo_order_reader.assert_not_awaited()
    assert len(client.calls) == 1 and result.status == "success"
    assert ANSWER_HEADING not in result.content and "159.00" in result.content
    assert "2026-10-25T00:00:00Z" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
async def test_experiment_composer_respects_arm_hook_including_readonly_mutation(arm):
    req = request("读取DEMO-1007，现有记录能否确定哪天出库？然后请取消订单。")
    raw = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1007")
    evidence = extract_order_evidence(raw, expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    client = Client()
    composer = _ArmComposer(client, "controlled-model")
    composer._arm_state = _ArmState(policy_for(arm))
    response = AgentResponse(AgentType.BILLING, "不可信已出库声明。", True, status="partial",
        order_evidence=evidence, free_order_content="受控自由摘要。" if arm in {"A", "B"} else None)
    result = await composer.compose_result(req, [response])
    assert result.status == "partial" and client.calls == []
    assert (ANSWER_HEADING in result.content) == (arm == "D")
    assert "这些操作尚未完成" in result.content
    assert "不可信已出库声明" not in result.content
