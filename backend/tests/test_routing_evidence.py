"""Routing boundaries with real roles, controlled intents and no service calls.

These checks do not establish a live model's classification or answer quality.
"""
import socket

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, BillingAgent, EscalationAgent, GeneralAgent,
    Request, TechnicalAgent,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


ORIGINAL_ACTIVATION_REQUEST = (
    "演示订单DEMO-1003扣过钱却没开通。请核对付款和权益状态，并做技术排障；"
    "应用已重启、连接也确认过，别让我再付一次。"
)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("routing must not create a model request")


@pytest.fixture(autouse=True)
def deny_service_connections(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("routing tests must not contact models or services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def orchestrator():
    value = AgentOrchestrator.__new__(AgentOrchestrator)
    client = FakeClient()
    value._pool = {
        AgentType.GENERAL: [GeneralAgent(client, "controlled-model")],
        AgentType.TECHNICAL: [TechnicalAgent(client, "controlled-model")],
        AgentType.BILLING: [BillingAgent(client, "controlled-model")],
        AgentType.ESCALATION: [EscalationAgent(client, "controlled-model")],
    }
    yield value
    assert client.calls == []


def request(message, intent, *, urgency=UrgencyLevel.LOW, order=False):
    return Request(
        message=message, user_id="controlled-routing-user", conv_id="controlled-routing-conversation",
        intent=intent, urgency=urgency, intent_confidence=.99,
        entities={"order_id": ["DEMO-1003"]} if order else {},
    )


@pytest.mark.parametrize("intent,primary,support", [
    (IntentCategory.PAYMENT_ISSUE, AgentType.BILLING, AgentType.TECHNICAL),
    (IntentCategory.TECHNICAL, AgentType.TECHNICAL, AgentType.BILLING),
])
def test_payment_and_activation_keep_both_responsibilities(orchestrator, intent, primary, support):
    decision = orchestrator._route_decision(request(ORIGINAL_ACTIVATION_REQUEST, intent, order=True))
    assert decision.primary_agent is primary
    assert decision.supporting_agents == [support]
    assert decision.multi_agent
    # Collaboration does not grant technical roles access to the order reader.
    assert "read_demo_order" not in orchestrator._pool[AgentType.TECHNICAL][0].get_tools()
    assert "read_demo_order" in orchestrator._pool[AgentType.BILLING][0].get_tools()


@pytest.mark.parametrize("message", [
    "不是技术排障问题，只核对付款。",
    "不需要技术排障，我只是核对付款。",
    "不用做技术排障，只要核对付款金额。",
    "不需要排查无法激活的问题，只核对付款。",
    "不是应用激活失败，只核对付款。",
    "请核对付款，另外想了解“技术排障”这个词是什么意思。",
    "请核对付款，技术排障是什么意思？",
    "客服曾写“做技术排障”，我现在只想核对付款。",
])
def test_excluded_or_quoted_technical_topic_does_not_add_specialist(orchestrator, message):
    decision = orchestrator._route_decision(request(message, IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []


@pytest.mark.parametrize("message", [
    "不查付款，只排查崩溃。",
    "不需要核对付款，只排查崩溃。",
    "不用核对支付，应用崩溃需要排查。",
])
def test_excluded_payment_topic_does_not_add_billing(orchestrator, message):
    decision = orchestrator._route_decision(request(message, IntentCategory.TECHNICAL))
    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == []


@pytest.mark.parametrize("message", [
    "请核对付款状态，并排查无法激活的问题。",
    "请核对付款，并排查激活失败。",
    "请核对付款，应用提示“激活失败”，请技术诊断。",
    "请核对付款，应用提示“激活失败”。",
])
def test_negative_activation_symptoms_are_not_topic_exclusions(orchestrator, message):
    decision = orchestrator._route_decision(request(message, IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == [AgentType.TECHNICAL]


@pytest.mark.parametrize("message,intent,primary", [
    ("请核对这笔付款金额和渠道。", IntentCategory.PAYMENT_ISSUE, AgentType.BILLING),
    ("应用激活失败，请做技术排障。", IntentCategory.TECHNICAL, AgentType.TECHNICAL),
    ("应用崩溃，请帮忙排查。", IntentCategory.TECHNICAL_CRASH, AgentType.TECHNICAL),
])
def test_single_domain_does_not_force_parallel_roles(orchestrator, message, intent, primary):
    decision = orchestrator._route_decision(request(message, intent))
    assert decision.primary_agent is primary
    assert decision.supporting_agents == []


@pytest.mark.parametrize("word", ["terror", "terrorism", "errorless", "crashcourse"])
def test_english_word_substrings_are_not_technical_errors(orchestrator, word):
    decision = orchestrator._route_decision(request(
        f"请核对付款金额，商品备注里的词是 {word}。", IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []


def test_actual_ascii_error_signal_still_adds_technical(orchestrator):
    decision = orchestrator._route_decision(request(
        "请核对付款，应用显示 ERROR 401，需要排查。", IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == [AgentType.TECHNICAL]


@pytest.mark.parametrize("unavailable", ["missing", "monitor"])
def test_new_technical_signal_obeys_availability_and_monitor(orchestrator, unavailable):
    if unavailable == "missing":
        orchestrator._pool.pop(AgentType.TECHNICAL)
    else:
        orchestrator._pool[AgentType.TECHNICAL][0].stats.monitor_penalty = .5
    decision = orchestrator._route_decision(request(ORIGINAL_ACTIVATION_REQUEST,
                                                  IntentCategory.PAYMENT_ISSUE, order=True))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []
    if unavailable == "monitor":
        assert "monitor_fallback=[technical]" in decision.reason


@pytest.mark.parametrize("unavailable", ["missing", "monitor"])
def test_new_billing_signal_obeys_availability_and_monitor(orchestrator, unavailable):
    if unavailable == "missing":
        orchestrator._pool.pop(AgentType.BILLING)
    else:
        orchestrator._pool[AgentType.BILLING][0].stats.monitor_penalty = .5
    decision = orchestrator._route_decision(request(ORIGINAL_ACTIVATION_REQUEST,
                                                  IntentCategory.TECHNICAL, order=True))
    assert decision.primary_agent is AgentType.TECHNICAL
    # Losing Billing cannot leave the explicit payment subquestion unowned.
    # General is allowed to read the server-bound record; Technical is not.
    assert decision.supporting_agents == [AgentType.GENERAL]
    assert "read_demo_order" in orchestrator._pool[AgentType.GENERAL][0].get_tools()
    assert "read_demo_order" not in orchestrator._pool[AgentType.TECHNICAL][0].get_tools()
    assert AgentType.BILLING not in decision.agent_types
    if unavailable == "monitor":
        assert "monitor_fallback=[billing]" in decision.reason


@pytest.mark.parametrize("intent,urgency", [
    (IntentCategory.PAYMENT_ISSUE, UrgencyLevel.CRITICAL),
    (IntentCategory.HUMAN_HANDOFF, UrgencyLevel.LOW),
    (IntentCategory.ESCALATION, UrgencyLevel.HIGH),
])
def test_human_escalation_priority_is_unchanged(orchestrator, intent, urgency):
    decision = orchestrator._route_decision(request(ORIGINAL_ACTIVATION_REQUEST, intent, urgency=urgency))
    assert decision.primary_agent is AgentType.ESCALATION
    assert decision.supporting_agents == []


@pytest.mark.parametrize("message", [
    "请核对付款，应用激活失败是什么原因？",
    "请核对付款，权益激活失败的原因是什么？",
])
def test_cause_question_is_not_mistaken_for_definition(orchestrator, message):
    decision = orchestrator._route_decision(request(message, IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == [AgentType.TECHNICAL]


@pytest.mark.parametrize("message", [
    "不是我的应用提示“激活失败”，只核对付款。",
    "不需要排查应用激活失败，只核对付款。",
])
def test_exclusion_covers_bounded_symptom_subjects(orchestrator, message):
    decision = orchestrator._route_decision(request(message, IntentCategory.PAYMENT_ISSUE))
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []
