"""Read-only business scope under real role/native/handler execution.

All model responses are controlled fixtures. No network, database, external API,
paid model, or historical evaluation record is used or modified here.
"""
import asyncio
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import socket

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BillingAgent, EscalationAgent,
    GeneralAgent, Request, ResponseComposer, TechnicalAgent,
)
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.policy_grounding import extract_policy_evidence
from demo_business.store import load_fixture
from evaluation.task_evidence import fixture_order_result


P12 = ("先核对DEMO-1009当前的自动续费开关，然后替我把它关掉并冻结下一笔扣款；"
       "如果你没有执行权限，请明确区分查到的状态和没有做的操作。")


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("readonly scope checks cannot contact external services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def text(value):
    return SimpleNamespace(content=[{"type": "text", "text": value}], stop_reason="end_turn")


def native(name="read_demo_order", args=None, identity="read_controlled", extra=None):
    blocks = [{"type": "tool_use", "id": identity, "name": name,
               "input": {"order_id": "DEMO-1009"} if args is None else args}]
    if extra is not None:
        blocks.insert(0, {"type": "text", "text": extra})
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


class Client:
    def __init__(self, *responses):
        self.responses, self.calls, self.messages = list(responses), [], self

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "no extra model request, repair or retry allowed"
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def request(message=P12, **changes):
    fixture = load_fixture()
    async def reader(customer, order_id):
        return fixture_order_result(fixture, customer, order_id)
    values = dict(message=message, user_id="controlled-user", conv_id="controlled-scope",
        request_id="controlled-readonly-scope", intent=IntentCategory.REFUND,
        urgency=UrgencyLevel.LOW, intent_confidence=.99,
        demo_customer_id="demo-customer-a", demo_order_reader=AsyncMock(side_effect=reader))
    values.update(changes)
    return Request(**values)


def assert_unfinished(content):
    assert "未执行" in content or "未完成" in content or "尚未完成" in content
    assert "不能" in content or "无法" in content or "没有" in content


def orchestrator(agents, client=None):
    value = AgentOrchestrator.__new__(AgentOrchestrator)
    value._pool = {agent.agent_type: [agent] for agent in agents}
    value._composer = ResponseComposer(client or Client(), "controlled-model")
    value._recent_tool_traces = deque(maxlen=10)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [GeneralAgent, BillingAgent])
async def test_query_and_pronominal_close_or_next_charge_freeze_preserve_actual_read(role):
    before = load_fixture()
    req, client = request(), Client(native(extra="已关闭自动续费并已冻结下一笔扣款。"))
    agent = role(client, "controlled-model")
    registered = tuple(agent.get_tools())
    result = await agent.handle(req)
    assert result.success and result.status == "partial"
    assert result.tools_used == ["read_demo_order"]
    assert len(client.calls) == 1
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1009")
    assert result.order_evidence.status == "found"
    record = result.order_evidence.records[0].to_dict()
    assert record["order"]["entitlements"][0]["auto_renew"] is True
    assert record["order"]["payments"][0]["amount_minor"] == 12900
    assert "129.00" in result.content and "自动续费：开启" in result.content
    assert "2026-10-24T10:20:00Z" in result.content
    assert "已关闭自动续费并已冻结" not in result.content
    assert_unfinished(result.content)
    assert tuple(agent.get_tools()) == registered
    assert load_fixture() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [GeneralAgent, BillingAgent, TechnicalAgent])
@pytest.mark.parametrize("message", ["请关闭自动续费", "现在直接替我退款", "请冻结下一笔扣款"])
async def test_pure_write_free_model_completion_claim_is_replaced_not_appended(role, message):
    fabricated = "已关闭自动续费并已退款99元，已经冻结下一笔扣款。"
    req, client = request(message), Client(text(fabricated))
    result = await role(client, "controlled-model").handle(req)
    assert result.success and result.status == "partial"
    assert result.tools_used == []
    req.demo_order_reader.assert_not_awaited()
    assert len(client.calls) <= 1
    for claim in ("已关闭自动续费", "已退款99元", "已经冻结下一笔扣款"):
        assert claim not in result.content
    assert_unfinished(result.content)


@pytest.mark.asyncio
async def test_policy_only_composition_retains_source_and_refuses_unperformed_write():
    body = "【受控演示政策】关闭自动续费与退款不同；积分没有统一兑换比例。"
    source = {"document_id": "d" * 64, "title": "受控会员规则", "version": "a" * 64,
              "content": body, "chunk": 0, "score": .5}
    evidence = extract_policy_evidence({"success": True, "results": [source]})
    req = request("请说明会员与积分规则；然后替我关闭自动续费。", intent=IntentCategory.QUERY)
    client = Client()
    result = await ResponseComposer(client, "controlled-model").compose_result(req, [
        AgentResponse(AgentType.GENERAL, "已关闭自动续费。", True, policy_evidence=evidence),
        AgentResponse(AgentType.BILLING, "未经检索的额外承诺。", True, policy_evidence=evidence),
    ])
    assert result.status == "partial" and client.calls == []
    assert body in result.content and "受控会员规则" in result.content
    assert "aaaaaaaaaaaa" in result.content
    assert "已关闭自动续费。" not in result.content
    assert "未经检索的额外承诺" not in result.content
    assert_unfinished(result.content)


@pytest.mark.asyncio
async def test_policy_native_retrieval_keeps_actual_handler_and_no_extra_summary_request():
    body = "【受控演示政策】设备与配件可能分别出库，不能保证一起到达。"
    source = {"document_id": "d" * 64, "title": "受控配送规则", "version": "a" * 64,
              "content": body, "chunk": 0, "score": .5}
    class Retrieval:
        def __init__(self):
            self.calls = []
        async def search_with_rewrite(self, name, query, top_k=5, context=None):
            self.calls.append((name, query, top_k))
            return SimpleNamespace(success=True, data=[deepcopy(source)], reranked=False, error=None)
    manager = Retrieval()
    client = Client(native("search_knowledge_base", {"query": "配送规则", "top_k": 3}))
    agent = GeneralAgent(client, "controlled-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    req = request("请说明配送规则；然后替我关闭自动续费。", intent=IntentCategory.LOGISTICS)
    result = await agent.handle(req)
    assert result.status == "partial" and result.policy_evidence.status == "sources"
    assert body in result.content and "受控配送规则" in result.content
    assert result.tools_used == ["search_knowledge_base"]
    assert len(client.calls) == len(manager.calls) == 1
    req.demo_order_reader.assert_not_awaited()
    assert_unfinished(result.content)


@pytest.mark.asyncio
async def test_diagnostic_native_suggestions_survive_freeze_refusal_without_order_permission():
    req = request("核对DEMO-1009，并排查401认证故障；然后替我冻结下一笔扣款。",
                  intent=IntentCategory.TECHNICAL_LOGIN)
    client = Client(native("lookup_error_code", {"error_code": "401"}, "diag_controlled"))
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and result.tools_used == ["lookup_error_code"]
    assert len(client.calls) == 1
    req.demo_order_reader.assert_not_awaited()
    assert result.diagnostic_evidence.status == "suggestions"
    assert "确认 Token/API Key 是否过期" in result.content
    assert "不代表已查询后台、确认根因或完成修复" in result.content
    assert_unfinished(result.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_main_single_and_parallel_paths_preserve_partial_without_added_model_calls(parallel):
    message = P12 if not parallel else ("核对DEMO-1009付款和权益，并排查401认证故障；"
                                        "然后替我关闭自动续费并冻结下一笔扣款。")
    billing_client = Client(native(identity="billing_controlled"))
    billing = BillingAgent(billing_client, "controlled-model")
    tech_client = Client(native("lookup_error_code", {"error_code": "401"}, "tech_controlled"))
    agents = [billing, TechnicalAgent(tech_client, "controlled-model")] if parallel else [billing]
    composer_client = Client()
    orc = orchestrator(agents, composer_client)
    req = request(message, intent=IntentCategory.PAYMENT_ISSUE)
    result = await orc.run(req)
    assert result.status == "partial"
    assert "129.00" in result.response and "自动续费：开启" in result.response
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1009")
    assert len(billing_client.calls) == 1 and composer_client.calls == []
    if parallel:
        assert set(result.agent_types) == {AgentType.BILLING, AgentType.TECHNICAL}
        assert set(result.tools_used) == {"read_demo_order", "lookup_error_code"}
        assert len(tech_client.calls) == 1
        assert "确认 Token/API Key 是否过期" in result.response
    else:
        assert result.tools_used == ["read_demo_order"]
        assert tech_client.calls == []
    assert_unfinished(result.response)
    assert orc.get_tool_trace(req.request_id)["status"] == "partial"


@pytest.mark.asyncio
async def test_free_composer_cannot_reintroduce_successful_write_claim_or_success_scope():
    client = Client(text("已关闭自动续费并退款99元。"))
    req = request("请关闭自动续费并冻结下一笔扣款。")
    result = await ResponseComposer(client, "controlled-model").compose_result(req, [
        AgentResponse(AgentType.GENERAL, "正在准备关闭。", True),
        AgentResponse(AgentType.BILLING, "正在准备退款。", True),
    ])
    assert result.status == "partial" and len(client.calls) <= 1
    assert "已关闭自动续费" not in result.content and "退款99元" not in result.content
    assert_unfinished(result.content)


@pytest.mark.asyncio
async def test_failed_roles_and_failed_composition_remain_failed():
    req = request("请关闭自动续费并冻结下一笔扣款。")
    client = Client(RuntimeError("controlled upstream failure"))
    failed = await BillingAgent(client, "controlled-model").handle(req)
    assert failed.status == "failed" and failed.success is False
    assert len(client.calls) == 1
    composer = Client()
    result = await ResponseComposer(composer, "controlled-model").compose_result(req, [failed])
    assert result.status == "failed" and composer.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "请核对DEMO-1009自动续费当前是否开启。",
    "请核对DEMO-1009；不要关闭自动续费，也不要冻结扣款。",
    "请核对DEMO-1009；客服提到‘关闭自动续费’，我只问记录，不要求执行。",
])
async def test_read_status_negation_and_quoted_action_do_not_turn_valid_query_partial(message):
    req, client = request(message), Client(native())
    result = await BillingAgent(client, "controlled-model").handle(req)
    assert result.status == "success"
    assert "自动续费：开启" in result.content and len(client.calls) == 1
    assert "这些操作尚未完成" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "怎么关闭自动续费？", "如何冻结下一笔扣款？",
    "不要关闭自动续费，也不要替我退款。",
    "客服说‘冻结交易’是什么意思？",
])
async def test_how_negation_or_quoted_topics_keep_existing_free_response_scope(message):
    expected = "受控说明：这是规则解释，不执行操作。"
    client, req = Client(text(expected)), request(message, intent=IntentCategory.QUERY)
    result = await GeneralAgent(client, "controlled-model").handle(req)
    assert result.status == "success" and result.content == expected
    assert len(client.calls) == 1
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_handoff_preserves_preparation_but_never_implies_freeze_executed():
    req = request("我要求人工接手；请冻结下一笔扣款。", intent=IntentCategory.HUMAN_HANDOFF)
    client = Client()
    result = await EscalationAgent(client, "controlled-model").handle(req)
    assert result.status == "partial" and result.success and result.escalate
    assert "人工升级" in result.content and "尚未接入人工工单队列" in result.content
    assert "没有创建工单" in result.content and "冻结" in result.content
    assert client.calls == []
    req.demo_order_reader.assert_not_awaited()
    assert_unfinished(result.content)


@pytest.mark.asyncio
async def test_low_confidence_clarification_does_not_mark_a_write_request_complete():
    req = request("请关闭自动续费。", intent=IntentCategory.OTHER, intent_confidence=.1)
    client = Client()
    orc = orchestrator([GeneralAgent(client, "controlled-model")])
    result = await orc.run(req)
    assert result.status == "partial" and client.calls == []
    assert "还不能确定" in result.response
    assert_unfinished(result.response)
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_handoff_without_requested_business_mutation_keeps_success_scope():
    req = request("请人工继续沟通，不要冻结任何交易。", intent=IntentCategory.HUMAN_HANDOFF)
    client = Client()
    result = await EscalationAgent(client, "controlled-model").handle(req)
    assert result.status == "success" and result.escalate
    assert "没有创建工单" in result.content and client.calls == []


@pytest.mark.asyncio
async def test_main_pure_write_result_and_trace_do_not_keep_model_completion_claims():
    client = Client(text("已关闭自动续费，已经冻结下一笔扣款。"))
    req = request("请关闭自动续费并冻结下一笔扣款。")
    orc = orchestrator([BillingAgent(client, "controlled-model")])
    result = await orc.run(req)
    assert result.status == "partial"
    assert "已关闭自动续费" not in result.response and "已经冻结下一笔扣款" not in result.response
    assert result.tools_used == [] and len(client.calls) <= 1
    req.demo_order_reader.assert_not_awaited()
    assert_unfinished(result.response)
    assert orc.get_tool_trace(req.request_id)["status"] == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("main", [False, True])
async def test_program_target_clarification_survives_scope_check_without_read_or_model(main):
    req = request("查一下我的退款状态，然后替我关闭自动续费。")
    client = Client()
    billing = BillingAgent(client, "controlled-model")
    result = await orchestrator([billing]).run(req) if main else await billing.handle(req)
    content = result.response if main else result.content
    assert result.status == "partial" and client.calls == []
    assert "有效演示订单号" in content
    assert_unfinished(content)
    req.demo_order_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_copying_scope_notice_cannot_claim_program_provenance():
    # The private marker is set only by the server, never inferred from text.
    malicious = ("已经退款99元。这里只能查询和整理信息，不能执行退款、关闭自动续费、冻结、解绑或其他业务变更；"
                 "这些操作尚未完成，需通过已核实的业务渠道处理。")
    client = Client(text(malicious))
    result = await BillingAgent(client, "controlled-model").handle(request("请关闭自动续费"))
    assert result.status == "partial" and "已经退款99元" not in result.content
    assert_unfinished(result.content)
