"""Cancellation is unknown when the validated read-only source lacks that field.

Pure projections and real native-role/fixture-handler checks, never a live
model, business API, merchant database or a rescore of historical evidence.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import socket

import pytest

from agents.agent_orchestrator import (
    BillingAgent, GeneralAgent, Request, ResponseComposer,
)
from core.intent_recognizer import IntentCategory
from core.order_evidence import (
    extract_order_evidence, merge_order_evidence, render_order_evidence,
)
from demo_business.store import SOURCE, load_fixture
from evaluation.task_evidence import fixture_order_result


NOTICE = "订单取消状态：未知"


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("cancellation scope checks cannot contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def packet(order_id, customer="demo-customer-a"):
    return fixture_order_result(load_fixture(), customer, order_id)


def evidence(order_id, customer="demo-customer-a"):
    return extract_order_evidence(packet(order_id, customer), expected_order_id=order_id,
                                  trusted_customer_id=customer)


def assert_scope(content):
    assert NOTICE in content and "当前快照未提供取消状态" in content
    assert "不能证明订单已取消或未取消" in content
    assert "付款状态" in content and "付款/发货记录是否存在" in content


@pytest.mark.parametrize("order_id,payment_status,has_shipments", [
    ("DEMO-1007", "unpaid", False),
    ("DEMO-1001", "paid", True),
    ("DEMO-1002", "paid", True),
    ("DEMO-1004", "paid", True),
    ("DEMO-1006", "refunded", False),
    ("DEMO-1008", "paid", True),
])
def test_missing_cancellation_field_is_not_inferred_from_payment_or_transport(order_id, payment_status, has_shipments):
    value = packet(order_id)
    before = deepcopy(value)
    checked = extract_order_evidence(value, expected_order_id=order_id,
                                     trusted_customer_id="demo-customer-a")
    content = render_order_evidence(checked)
    assert checked.status == "found"
    assert_scope(content)
    assert content.count(NOTICE) == 1
    assert value["order"]["payment_status"] == payment_status
    assert bool(value["order"]["shipments"]) is has_shipments
    assert value == before == checked.records[0].to_dict()
    assert not any("cancel" in field for field in value["order"])
    assert value["as_of"] in content and value["order"]["record_updated_at"] in content
    total = value["order"]["total_amount_minor"]
    assert f"{total // 100}.{total % 100:02d} 元" in content


def test_unpaid_without_payment_or_transport_keeps_empty_records_and_unknown_ship_date():
    value = packet("DEMO-1007")
    checked = evidence("DEMO-1007")
    content = render_order_evidence(checked)
    assert value["order"]["payments"] == value["order"]["shipments"] == []
    assert "未付款（unpaid）" in content
    assert "当前快照未提供付款记录。" in content
    assert "当前快照未提供包裹运输记录。" in content
    assert "159.00 元" in content
    assert_scope(content)
    assert "该订单已取消" not in content and "该订单未取消" not in content


def test_pending_refund_unknown_amount_stays_unknown_instead_of_zero_or_order_price():
    value = evidence("DEMO-1004")
    before = value.records[0].canonical_json
    content = render_order_evidence(value)
    assert "退款 REF-1004：待审核（pending_review）；记录金额：未知" in content
    assert "记录金额：0.00 元" not in content
    assert "记录金额：299.00 元" not in content
    assert_scope(content)
    assert value.records[0].canonical_json == before


def test_authorization_not_found_does_not_acquire_found_cancellation_facts_or_leak_owner():
    checked = evidence("DEMO-1005")
    content = render_order_evidence(checked)
    assert checked.status == "not_found"
    assert "当前演示身份的可见记录中未找到" in content
    assert NOTICE not in content
    for private in ("演示客户B专属设备", "699.00", "DEMO-TRACK-1005", "demo-customer-b"):
        assert private not in content


def test_unavailable_or_invalid_reads_never_display_a_found_cancellation_row():
    unavailable = extract_order_evidence({"source": SOURCE, "is_demo": True,
        "success": False, "status": "unavailable", "error_code": "demo_store_unavailable"},
        expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    assert unavailable.status == "unavailable"
    assert NOTICE not in render_order_evidence(unavailable)
    invalid = extract_order_evidence(None, expected_order_id="DEMO-1007",
                                     trusted_customer_id="demo-customer-a")
    assert invalid.status == "invalid"
    assert "校验未通过" in render_order_evidence(invalid)
    assert NOTICE not in render_order_evidence(invalid)


@pytest.mark.parametrize("field,value", [("cancellation_status", "cancelled"), ("cancelled", True)])
def test_new_claimed_fields_are_still_rejected_instead_of_loosening_source_contract(field, value):
    result = packet("DEMO-1007")
    result["order"][field] = value
    checked = extract_order_evidence(result, expected_order_id="DEMO-1007",
                                     trusted_customer_id="demo-customer-a")
    assert checked.status == "invalid"
    assert NOTICE not in render_order_evidence(checked)
    assert field not in render_order_evidence(checked)


def test_merged_records_explain_each_found_order_once_without_promoting_not_found():
    first, second = evidence("DEMO-1007"), evidence("DEMO-1006")
    merged = merge_order_evidence([first, second, first, evidence("DEMO-1005")])
    assert merged.status == "partial"
    before = deepcopy(merged)
    content = render_order_evidence(merged)
    assert content.count(NOTICE) == 2
    assert "订单 DEMO-1005：在当前演示身份的可见记录中未找到" in content
    assert "本次仅取得部分查询结果" in content
    assert merged == before


class Client:
    def __init__(self, *responses):
        self.messages, self.calls, self.responses = self, [], list(responses)
    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "no summary model, retries or extra tools permitted"
        return self.responses.pop(0)


def native(identity):
    return SimpleNamespace(stop_reason="tool_use", content=[
        {"type": "text", "text": "整单已经取消，付款后会自动发货。"},
        {"type": "tool_use", "id": identity, "name": "read_demo_order",
         "input": {"order_id": "DEMO-1007"}},
    ])


def request():
    fixture = load_fixture()
    async def reader(customer, order):
        return fixture_order_result(fixture, customer, order)
    return Request(message="核对DEMO-1007：没付款也没发货记录，是不是已经取消了？",
        user_id="controlled-cancellation-user", conv_id="controlled-cancellation",
        request_id="controlled-cancellation-read", intent=IntentCategory.ORDER_STATUS,
        demo_customer_id="demo-customer-a", demo_order_reader=AsyncMock(side_effect=reader))


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [GeneralAgent, BillingAgent])
async def test_real_native_read_projects_unknown_scope_without_accepting_model_cancellation_claim(role):
    req, client = request(), Client(native("controlled_cancellation"))
    agent = role(client, "controlled-model")
    tools = tuple(agent.get_tools())
    result = await agent.handle(req)
    assert result.success and result.status == "success"
    assert result.tools_used == ["read_demo_order"]
    assert len(client.calls) == 1
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "read_demo_order"}
    req.demo_order_reader.assert_awaited_once_with("demo-customer-a", "DEMO-1007")
    assert_scope(result.content)
    assert "整单已经取消" not in result.content and "付款后会自动发货" not in result.content
    assert result.order_evidence.records[0].to_dict() == packet("DEMO-1007")
    assert "159.00 元" in result.content and "2026-10-24T10:00:00Z" in result.content
    assert tuple(agent.get_tools()) == tools


@pytest.mark.asyncio
async def test_composer_merges_actual_read_evidence_and_keeps_one_boundary_without_extra_model():
    req = request()
    general_client, billing_client = Client(native("general_cancellation")), Client(native("billing_cancellation"))
    responses = [await GeneralAgent(general_client, "controlled-model").handle(req),
                 await BillingAgent(billing_client, "controlled-model").handle(req)]
    composer_client = Client()
    result = await ResponseComposer(composer_client, "controlled-model").compose_result(req, responses)
    assert result.status == "success"
    assert result.content.count(NOTICE) == 1
    assert_scope(result.content)
    assert "整单已经取消" not in result.content and "付款后会自动发货" not in result.content
    assert req.demo_order_reader.await_count == 2
    assert len(general_client.calls) == len(billing_client.calls) == 1
    assert composer_client.calls == []
    assert len(result.order_evidence.records) == 1
    assert result.order_evidence.records[0].to_dict() == packet("DEMO-1007")
