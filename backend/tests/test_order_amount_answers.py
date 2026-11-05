"""Pure arithmetic and controlled native wiring, never new model scores.

All changed amounts/quantities live in copied synthetic packets. The immutable
fixture, prior questions, reviews and evidence are neither rewritten nor scored.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import socket
from unittest.mock import AsyncMock

import pytest

from core.order_answers import render_order_answers
from core.order_evidence import extract_order_evidence
from demo_business.store import FIXTURE_PATH, load_fixture
from evaluation.task_evidence import fixture_order_result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("amount answer regressions forbid services")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    before = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    yield
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == before


def packet(*, quantities=(2, 3), units=(12345, 6789), state="paid", split=False):
    value = deepcopy(fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1001"))
    order = value["order"]
    for row, quantity, unit, name in zip(order["items"], quantities, units, ("离线零件甲", "离线零件乙")):
        row.update(quantity=quantity, unit_amount_minor=unit, name=name)
    total = sum(quantity * unit for quantity, unit in zip(quantities, units))
    order.update(total_amount_minor=total, payment_status=state)
    order["refunds"] = []
    order["payments"][0].update(amount_minor=total, status=state)
    if state == "unpaid":
        order["payments"] = []
    elif split:
        first = order["payments"][0]
        first["amount_minor"] = total // 3
        order["payments"].append({**first, "payment_id": "PAY-OFFLINE-SECOND", "amount_minor": total - first["amount_minor"]})
    if state == "refunded":
        order["refunds"] = [{"refund_id": "REF-OFFLINE", "order_id": "DEMO-1001", "status": "completed", "amount_minor": total, "record_updated_at": order["record_updated_at"]}]
    return value


def checked(value=None):
    result = extract_order_evidence(packet() if value is None else value,
        expected_order_id="DEMO-1001", trusted_customer_id="demo-customer-a")
    assert result.status == "found"
    return result


@pytest.mark.parametrize("message", [
    "核对DEMO-1001的商品明细和实付，合起来是否等于付款记录？",
    "查询DEMO-1001，所有商品价格加起来与支付记录对得上吗？",
    "请算一下DEMO-1001商品明细的小计和总价，与订单金额、付款金额比较。",
    "Check DEMO-1001: do the line items total match the payment amount?",
    "只核对DEMO-1001的商品合计与实付是否一致。",
])
def test_quantity_times_record_price_and_explicit_comparison_answer_the_question(message):
    value = checked()
    before = deepcopy(value)
    text = render_order_answers(value, message=message)
    for required in ("123.45 元 × 2 = 246.90 元", "67.89 元 × 3 = 203.67 元", "商品明细合计：450.57 元", "订单金额记录比较：450.57 元", "原始金额合计 450.57 元（paid）", "数值一致", "CNY", "serveflow-mysql-demo-v1", "2026-10-25T00:00:00Z"):
        assert required in text
    assert "真实支付渠道入账或履约完成" in text
    assert "逐包裹" not in text and "数字权益记录" not in text
    assert value == before


@pytest.mark.parametrize("quantities,units,expected", [
    ((4, 7), (1, 1905), "133.39"),
    ((1, 2), (0, 99), "1.98"),
    ((100, 100), (1234567, 2345678), "3580245.00"),
])
def test_new_integer_cent_combinations_cannot_depend_on_original_order_values(quantities, units, expected):
    text = render_order_answers(checked(packet(quantities=quantities, units=units)),
        message="查DEMO-1001，商品明细一共多少元，和实付一致吗？")
    assert f"商品明细合计：{expected} 元" in text
    assert f"原始金额合计 {expected} 元（paid）" in text
    assert "1698.00" not in text


def test_multiple_payment_rows_are_summed_once_and_never_called_multiple_charges():
    text = render_order_answers(checked(packet(split=True)),
        message="核对DEMO-1001商品明细合计与付款记录是否一样。")
    assert "2条付款记录的原始金额合计 450.57 元" in text
    assert "重复扣款" not in text


def test_no_payment_rows_are_unknown_comparison_amount_rather_than_zero_paid():
    text = render_order_answers(checked(packet(state="unpaid")),
        message="核对DEMO-1001商品合计与实付，差额是多少？")
    assert "商品明细合计：450.57 元" in text
    assert "无法取得可比较的实付金额" in text and "不能将空付款记录按0元实付处理" in text
    assert "原始金额合计" not in text and "差额为0" not in text


def test_refunded_payment_is_original_record_amount_not_current_net_paid():
    text = render_order_answers(checked(packet(state="refunded")),
        message="核对DEMO-1001商品总价是否等于付款记录。")
    assert "原始金额合计 450.57 元（refunded）" in text
    assert "不能视为退款后的当前净实付" in text and "真实退款到账仍未核验" in text


def test_total_only_does_not_add_unasked_payment_or_fee_explanations():
    text = render_order_answers(checked(), message="请查询DEMO-1001，商品明细合计多少元？")
    assert "商品明细合计：450.57 元" in text
    assert "与付款记录比较" not in text and "与订单金额记录比较" not in text
    assert "具体金额未知" not in text


@pytest.mark.parametrize("message", [
    "核对DEMO-1001商品总价和付款，运费、优惠券折扣、税费分别是多少？",
    "查DEMO-1001，商品合计等于实付就说明运费和优惠是0元吗？",
])
def test_missing_fees_are_not_inferred_from_matching_known_record_totals(message):
    text = render_order_answers(checked(), message=message)
    assert "商品明细合计：450.57 元" in text
    assert "具体金额未知，不能按0元补齐" in text
    assert "不能证明这些费用不存在或已如何分摊" in text
    assert "运费/邮费" in text and "优惠/折扣" in text


def test_per_item_transaction_allocation_is_explicitly_outside_available_fields():
    text = render_order_answers(checked(),
        message="查DEMO-1001商品明细，每件实际支付多少元，优惠如何分摊？")
    assert "没有每件商品的成交价、优惠分摊或净实付字段" in text
    assert "具体分摊仍未知" in text
    assert "每件实际成交/实付金额" in text and "商品明细合计：" not in text


@pytest.mark.parametrize("message", [
    "查DEMO-1001，只核对付款状态，不计算商品合计。",
    "查DEMO-1001，不用比较明细和实付。",
    "查DEMO-1001，不解释运费或优惠。",
    "查DEMO-1001，把商品合计这个词翻译为英文。",
    "查询DEMO-1001，文档说商品合计等于付款记录。",
    "查DEMO-1001，客服说‘商品合计一定等于实付’，这里只是引用。",
    "查DEMO-1001，商品数量合计几件？",
    "查DEMO-1001，商品明细中只算离线零件甲的总价与付款比较。",
    "查询DEMO-1001，商品明细里离线零件甲的合计多少元？",
    "查询DEMO-1001商品名称，再核对付款状态。",
    "核对DEMO-1001商品明细，付款记录合计多少元？",
    "核对DEMO-1001，商品单价是否等于付款金额？",
    "比较DEMO-1001商品明细和付款状态是否一致。",
])
def test_declined_inert_quantity_or_filtered_topics_cannot_become_full_item_amount(message):
    assert render_order_answers(checked(), message=message) == ""


def test_explicit_verification_of_quoted_arithmetic_uses_current_record_not_quote_numbers():
    text = render_order_answers(checked(),
        message="核对DEMO-1001。客服说‘商品合计等于付款9999元’，这个结论是否成立？")
    assert "商品明细合计：450.57 元" in text and "9999" not in text


def test_identity_evidence_and_user_history_guards_still_apply_to_computed_answers():
    value = checked()
    question = "这笔订单商品合计与付款记录是否一致？"
    user = {"role": "user", "content": "请查询DEMO-1001的付款。"}
    assistant = {"role": "assistant", "content": "请查DEMO-1002的商品总价。"}
    assert "450.57" in render_order_answers(value, message=question, history=[user, assistant])
    assert render_order_answers(value, message=question, history=[assistant]) == ""
    assert render_order_answers(replace(value, customer_id="demo-customer-b"), message="查DEMO-1001商品合计？") == ""
    for message in ("查DEMO-1002的商品合计？", "核对DEMO-1001和DEMO-1002商品合计？", "不要查询DEMO-1001的商品合计。"):
        assert render_order_answers(value, message=message) == ""


@pytest.mark.parametrize("mutation", [
    lambda value: value["order"]["items"][0].update(unit_amount_minor=True),
    lambda value: value["order"].update(currency="USD"),
    lambda value: value["order"]["payments"][0].update(amount_minor=1),
    lambda value: value["order"].update(shipping_fee_minor=0),
])
def test_unvalidated_amount_currency_or_new_fields_cannot_feed_arithmetic(mutation):
    raw = packet()
    mutation(raw)
    value = extract_order_evidence(raw, expected_order_id="DEMO-1001", trusted_customer_id="demo-customer-a")
    assert value.status == "invalid"
    assert render_order_answers(value, message="核对DEMO-1001商品明细合计和实付？") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("write", [False, True])
async def test_native_read_and_composer_keep_computation_and_uncompleted_write_boundary(write):
    from agents.agent_orchestrator import BillingAgent, ResponseComposer
    from test_order_answer_integration import Client, native, orchestrator, request

    message = "只查询DEMO-1001商品明细合计是否等于付款记录，运费未知不要补0。"
    if write:
        message += "然后请把99.90元转回我的卡。"
    reader = AsyncMock(return_value=packet(split=True))
    req = request(message, reader=reader)
    client = Client(native(args={"order_id": "DEMO-1001"}, prose="合计9999元，已转账。"))
    composer_client = Client()
    result = await orchestrator([BillingAgent(client, "controlled-model")],
        ResponseComposer(composer_client, "controlled-model")).run(req)
    assert result.status == ("partial" if write else "success")
    assert result.tools_used == ["read_demo_order"] and len(client.calls) == 1
    assert composer_client.calls == []
    reader.assert_awaited_once_with("demo-customer-a", "DEMO-1001")
    assert "商品明细合计：450.57 元" in result.response
    assert "具体金额未知，不能按0元补齐" in result.response
    assert result.response.count("针对本次问题的只读说明") == 1
    assert "9999" not in result.response and "已转账" not in result.response
    if write:
        assert "这些操作尚未完成" in result.response


@pytest.mark.parametrize("order_id,message,required", [
    ("DEMO-1008", "查DEMO-1008，退款行零条就能证明我没申请过吗？", "不能证明从未申请"),
    ("DEMO-1003", "查DEMO-1003，到期日没写是不是一直有效？开通失败是怎么回事？", "没有提供本例失败原因"),
    ("DEMO-1010", "查DEMO-1010，续费关闭就说明今后不会再扣款吗？", "不能保证未来续费、扣款或可用状态"),
])
def test_common_missing_record_wording_keeps_existing_status_boundaries(order_id, message, required):
    raw = fixture_order_result(load_fixture(), "demo-customer-a", order_id)
    value = extract_order_evidence(raw, expected_order_id=order_id, trusted_customer_id="demo-customer-a")
    text = render_order_answers(value, message=message)
    assert required in text
    if order_id == "DEMO-1003":
        assert "不能理解为永久有效" in text
