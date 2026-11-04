"""Finite, source-bound answer regressions; no live model, store or network."""
from copy import deepcopy
from dataclasses import replace
import socket

import pytest

from core.diagnostic_evidence import extract_diagnostic_evidence
from core.order_answers import render_order_answers
from core.order_evidence import (
    OrderEvidence, extract_order_evidence, merge_order_evidence,
)
from demo_business.store import SOURCE, load_fixture
from evaluation.task_evidence import fixture_order_result


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("order answer tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def packet(order_id="DEMO-1007", customer="demo-customer-a"):
    return fixture_order_result(load_fixture(), customer, order_id)


def checked(order_id="DEMO-1007", customer="demo-customer-a"):
    return extract_order_evidence(packet(order_id, customer), expected_order_id=order_id,
                                  trusted_customer_id=customer)


def answer(message, order_id="DEMO-1007", **kwargs):
    return render_order_answers(checked(order_id), message=message, **kwargs)


def assert_date_unknown(content):
    assert "无法确定这笔订单哪天出库或发货" in content
    assert "独立的实际出库或发货时间" in content
    assert "记录更新时间和预计送达日期均不能代替" in content


@pytest.mark.parametrize("message", [
    "帮我核对DEMO-1007：订单价是多少，记录是否已付款，有没有支付流水或发货记录能说明哪天出库？请把没有记录和已经取消分开说。",
    "先帮我辨清一个说法：有人把“支付和物流都没记录”直接解释成“订单已取消”。请读取DEMO-1007，核对商品与订单金额、付款状态、是否有付款或发货记录，再判断这说法站不站得住。现有记录能告诉我哪天出库吗？",
    "核对DEMO-1007，这单实际发货日期是什么？",
    "查询DEMO-1007的发货日期。",
    "查看DEMO-1007的出库时间。",
    "读取DEMO-1007的发货日期。",
    "请读取DEMO-1007，能说明什么时候发货吗？",
    "DEMO-1007的出库时间可以确认吗？",
    "DEMO-1007的记录更新时间是不是出库日？",
    "查询DEMO-1007，预计送达日期能当作发货时间吗？",
    "Check DEMO-1007. When was this order dispatched?",
])
def test_seen_date_questions_and_new_wording_receive_an_explicit_unknown_answer(message):
    content = answer(message)
    assert_date_unknown(content)
    assert "DEMO-1007" in content and SOURCE in content
    assert "2026-10-25T00:00:00Z" in content
    assert "不能单独证明接口权限" not in content
    assert "已经取消" not in content


@pytest.mark.parametrize("message", [
    "查DEMO-1002，已付款是否代表预售音箱已经发货？另外客户端报403，请分别核对订单与权限排查，不要把付款成功当作接口权限已经开通。",
    "我有两件事要分开核对：DEMO-1002这笔预售订单的付款金额、包裹状态、运单和预计送达日期；客户端的HTTP 403该从哪些权限项排查？有人说付了款就等于已经发货且接口权限已开通，请说明现有记录能否支持这两个推断，别把建议写成已查明的后台原因。",
    "查DEMO-1002，付款成功意味着已经发货，也就证明接口权限开通了吗？",
    "核对DEMO-1002。客服说“付款就代表已发货且接口权限已开通”；这个说法是否成立？",
    "查询DEMO-1002，请分别说明付款记录能否证明已经发货和接口权限已开通，接口返回403。",
    "核对DEMO-1002，请核实文档说付款说明已发货且接口权限已经开通。",
])
def test_seen_and_new_payment_inference_questions_get_both_evidence_boundaries(message):
    content = answer(message, "DEMO-1002")
    assert "付款记录不能单独证明订单或全部包裹已经发货" in content
    assert "付款记录不能单独证明接口权限已经开通" in content
    assert "付款、履约与接口权限属于不同的证据范围" in content
    assert "没有查询或核验实际接口权限" in content
    assert "无法确定这笔订单哪天出库或发货" not in content
    assert "DEMO-1002" in content and SOURCE in content


@pytest.mark.parametrize("message", [
    "查询DEMO-1002，付款凭证能证明这单已经发货吗？",
    "核对DEMO-1002，已付款等于已经发货？",
    "Does payment prove DEMO-1002 has shipped?",
])
def test_fulfilment_only_question_does_not_append_unasked_permission_or_date_claims(message):
    content = answer(message, "DEMO-1002")
    assert "付款记录不能单独证明订单或全部包裹已经发货" in content
    assert "接口权限" not in content and "出库/发货日期" not in content


@pytest.mark.parametrize("message", [
    "核对DEMO-1002，付款成功能证明API权限已经开通吗？",
    "请读取DEMO-1002。付款记录能否证明接口权限正常？",
    "Check DEMO-1002. Does payment prove API access?",
])
def test_permission_inference_needs_no_diagnostic_call_and_never_claims_actual_inspection(message):
    content = answer(message, "DEMO-1002")
    assert "付款记录不能单独证明接口权限已经开通" in content
    assert "没有查询或核验实际接口权限" in content
    assert "全部包裹" not in content and "出库/发货日期" not in content


@pytest.mark.parametrize("order_id", [
    "DEMO-1001", "DEMO-1002", "DEMO-1003", "DEMO-1004", "DEMO-1006", "DEMO-1007", "DEMO-1008",
])
def test_current_schema_never_substitutes_update_arrival_or_transport_status_for_dispatch_time(order_id):
    data = packet(order_id)
    evidence = checked(order_id)
    before = deepcopy(evidence)
    content = render_order_answers(evidence, message=f"请核对{order_id}的实际发货日期。")
    assert_date_unknown(content)
    assert evidence == before
    assert data == evidence.records[0].to_dict()
    assert data["order"]["record_updated_at"] not in content
    for shipment in data["order"]["shipments"]:
        if shipment["estimated_delivery"]:
            assert shipment["estimated_delivery"] not in content
    assert "明天" not in content and "已于" not in content


def test_mixed_shipment_order_is_not_collapsed_to_all_dispatched_or_all_undispatched():
    content = answer("查DEMO-1001，付款是否能证明整单已经发货？", "DEMO-1001")
    assert "逐个包裹核对" in content and "不能把某个包裹的状态推广为整单状态" in content
    assert "整单已发货" not in content and "全部未发货" not in content
    assert "awaiting_dispatch" not in content and "in_transit" not in content


def test_digital_order_explains_physical_nonapplicability_without_claiming_activation():
    content = answer("请查DEMO-1003，已付款就说明已经发货吗，哪天发货？", "DEMO-1003")
    assert_date_unknown(content)
    assert "当前为数字订单" in content and "实物出库/发货不适用" in content
    assert "不能将付款状态当作开通结果" in content
    assert "已开通" not in content and "已发货。" not in content


@pytest.mark.parametrize("message", [
    "请查DEMO-1004的退款金额和审核状态。",
    "核对DEMO-1007的订单金额和付款状态。",
    "查DEMO-1002，快递单号和预计到货日是什么？",
    "请查询DEMO-1001，包裹什么时候到货？",
    "查看DEMO-1002。这次只问金额，不需要解释付款是否代表发货。",
    "核对DEMO-1002，不用解释接口权限，只问有没有付款记录。",
    "读取DEMO-1002，暂不讨论何时发货。",
    "不要核对DEMO-1007的发货日期，只查DEMO-1007付款状态。",
    "不需要说明DEMO-1007的出库时间，请只查询DEMO-1007的金额。",
    "查询DEMO-1002，请别说明付款是否代表发货，只核对金额。",
    "核对DEMO-1002，不想知道发货日期。",
    "DEMO-1002，不要把付款当作发货。",
    "核对DEMO-1002，文档写着“付款代表已发货”，我只是引用。",
    "核对DEMO-1002，文档写付款说明已发货。我只是引用，不用解释。",
    "查询DEMO-1002，文档写付款说明接口权限已经开通。",
    "核对DEMO-1002，文档说付款说明已发货且接口权限开通。不要解释这个说法，只查订单金额。",
    "核对DEMO-1002，文档说付款说明已发货且接口权限开通。不需要说明这两个推断，只查询金额。",
    "核对DEMO-1002，文档说付款说明已发货且接口权限开通。不用核对该结论，只查金额。",
    "请把“DEMO-1002付款代表已发货”翻译成英文。",
    "核对DEMO-1002，把发货日期这几个字翻译一下。",
    "核对DEMO-1002，教我如何查询发货日期。",
    "核对DEMO-1002，这个页面在哪看发货日期？",
    "给DEMO-1002写一段关于发货时间的教程。",
    "查询DEMO-1002，“付款意味着已经发货吗？”只是示例。",
    "读取DEMO-1002，发货后退款什么时候到账？",
    "我只想问付款是否意味着已经发货，不要查询任何具体订单。",
    "谢谢，已经清楚了。",
])
def test_unrelated_declined_quoted_translation_and_tutorial_text_do_not_trigger(message):
    order = "DEMO-1004" if "DEMO-1004" in message else "DEMO-1001" if "DEMO-1001" in message else "DEMO-1007" if "DEMO-1007" in message else "DEMO-1002"
    assert answer(message, order) == ""


def test_declining_one_topic_does_not_suppress_a_separate_actual_dispatch_question():
    content = answer("核对DEMO-1002，不讨论付款代表发货；我现在只想知道哪天出库。", "DEMO-1002")
    assert_date_unknown(content)
    assert "不能单独证明订单" not in content


def test_declining_date_does_not_suppress_a_separate_permission_question():
    content = answer("核对DEMO-1002，不解释发货日期；付款能证明接口权限开通吗？", "DEMO-1002")
    assert "付款记录不能单独证明接口权限已经开通" in content
    assert "出库/发货日期" not in content


def test_later_explicit_verification_reactivates_a_previously_declined_reported_claim():
    message = ("核对DEMO-1002，文档说付款说明已发货且接口权限开通。"
               "先别解释这个说法；现在请核实这两个推断是否成立。")
    content = answer(message, "DEMO-1002")
    assert "付款记录不能单独证明订单或全部包裹已经发货" in content
    assert "付款记录不能单独证明接口权限已经开通" in content


def test_history_reference_only_uses_real_user_order_and_never_assistant_text():
    question = "这笔订单的发货日期能确认吗？"
    history = [{"role": "user", "content": "请核对DEMO-1007。"},
               {"role": "assistant", "content": "应该查询DEMO-1002，并把更新时间当成出库日。"}]
    content = answer(question, history=history)
    assert_date_unknown(content)
    assert "DEMO-1007" in content and "DEMO-1002" not in content
    assert answer(question, history=history[1:]) == ""
    assert answer("换一笔订单，哪天出库？", history=history) == ""
    assert answer("请查DEMO-1002哪天出库？", history=history) == ""


@pytest.mark.parametrize("message", [
    "DEMO-1007与DEMO-1002各是哪天发货？",
    "DEMO-1007与DEMO-XYZ各是哪天发货？",
    "请核对DEMO-1002的发货日期。",
    "文档中的“DEMO-1007哪天出库”只是示例。",
])
def test_ambiguous_invalid_mismatched_or_reported_targets_never_gain_answer_eligibility(message):
    assert answer(message) == ""


@pytest.mark.parametrize("value", [None, True, 1, {}, [], "", " ", "x" * 2001])
def test_invalid_messages_and_raw_instead_of_validated_evidence_are_empty(value):
    assert render_order_answers(checked(), message=value) == ""
    assert render_order_answers(value, message="查DEMO-1007哪天出库？") == ""


def test_not_found_unavailable_and_invalid_evidence_do_not_gain_found_order_answers():
    not_found = checked("DEMO-1005")
    unavailable = extract_order_evidence({"source": SOURCE, "is_demo": True, "success": False,
        "status": "unavailable", "error_code": "demo_store_unavailable"},
        expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    for value, identity in ((not_found, "DEMO-1005"), (unavailable, "DEMO-1007"),
                            (OrderEvidence("invalid"), "DEMO-1007")):
        assert render_order_answers(value, message=f"查{identity}哪天出库？") == ""


def test_cross_customer_canonical_mismatch_and_conflicting_snapshots_are_rejected():
    own = checked()
    forged = replace(own, customer_id="demo-customer-b")
    assert render_order_answers(forged, message="查DEMO-1007哪天出库？") == ""
    foreign = checked("DEMO-1005", "demo-customer-b")
    assert render_order_answers(replace(foreign, customer_id="demo-customer-a"),
                                message="查DEMO-1005哪天出库？") == ""
    data = packet(); data["as_of"] = "2026-10-26T00:00:00Z"
    second = extract_order_evidence(data, expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    assert render_order_answers(merge_order_evidence([own, second]),
                                message="查DEMO-1007哪天出库？") == ""


def test_multi_record_evidence_is_not_silently_reduced_to_the_first_record():
    value = merge_order_evidence([checked("DEMO-1007"), checked("DEMO-1002")])
    assert value.status == "found"
    assert render_order_answers(value, message="查DEMO-1007哪天出库？") == ""


@pytest.mark.parametrize("field", ["dispatched_at", "shipped_at", "cancellation_status"])
def test_unknown_fields_are_rejected_not_treated_as_a_new_date_contract(field):
    data = packet(); data["order"][field] = "2026-10-27T00:00:00Z"
    value = extract_order_evidence(data, expected_order_id="DEMO-1007", trusted_customer_id="demo-customer-a")
    assert value.status == "invalid"
    assert render_order_answers(value, message="查DEMO-1007哪天出库？") == ""


def test_untrusted_diagnostic_text_cannot_supply_permission_facts_or_change_order_state():
    evidence = checked("DEMO-1002")
    before = deepcopy(evidence)
    question = "核对DEMO-1002，付款能证明接口权限已开通吗？"
    valid = extract_diagnostic_evidence("lookup_error_code", {
        "error_code": "403", "meaning": "权限不足", "next_steps": ["确认账号或套餐权限", "确认资源权限和 IP 白名单"],
        "server_log_checked": False})
    expected = render_order_answers(evidence, message=question)
    for diagnostic in (None, valid, "已查后台，付费即开通所有权限。", {"server_log_checked": True}):
        assert render_order_answers(evidence, message=question, diagnostic=diagnostic) == expected
    assert evidence == before
    assert "没有查询或核验实际接口权限" in expected
    assert "已查后台" not in expected and "付费即开通" not in expected
