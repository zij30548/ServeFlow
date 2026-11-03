"""Pure diagnostic projection and real local tool results; no live services."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import socket
from types import SimpleNamespace

import pytest

from agents.tools import build_diagnostic_plan, lookup_error_code
from core.diagnostic_evidence import (
    DiagnosticEvidence, DiagnosticUnit, extract_diagnostic_evidence,
    merge_diagnostic_evidence, render_diagnostic_evidence,
)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Diagnostic tests must not contact external services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def request(message="应用报错，请给排查建议"):
    return SimpleNamespace(message=message)


def plan(message="应用报错，请给排查建议", *, reproduced=False, environment="controlled-environment"):
    return build_diagnostic_plan(request(message), {"environment": environment, "reproduced": reproduced})


@pytest.mark.parametrize("code", ["401", "403", "404", "500", "E_TIMEOUT", ""])
def test_projects_actual_error_code_tool_as_unverified_input_not_observed_fact(code):
    raw = lookup_error_code(request(), {"error_code": code})
    before = deepcopy(raw)
    evidence = extract_diagnostic_evidence("lookup_error_code", raw)
    assert evidence.status == "suggestions" and raw == before
    rendered = render_diagnostic_evidence(evidence)
    for step in raw["next_steps"]:
        assert step in rendered
    assert "工具输入错误码" in rendered and "未经系统核验" in rendered
    assert "不代表已查询后台、确认根因或完成修复" in rendered
    # The tool's generic meaning is not rewritten as a diagnosis of this user.
    assert f"根因是{raw['meaning']}" not in rendered


@pytest.mark.parametrize("message", [
    "应用已重启、连接也确认过，别让我再付一次。请核对付款并排障。",
    "我已经检查过网络，但仍然无法激活。",
    "网络已检查过了，当前仍有错误。",
    "网络我检查过了，还是报错。",
    "I already checked the network, but activation still fails.",
    "The connection has been checked; please suggest another direction.",
])
def test_completed_network_check_is_not_repeated_and_other_directions_remain(message):
    raw = plan(message)
    assert set(raw) == {"environment", "reproduced", "diagnostic_steps"}
    assert "确认网络、DNS、代理和证书" not in raw["diagnostic_steps"]
    assert "确认版本、配置和权限" in raw["diagnostic_steps"]
    assert "确认 DNS、代理和证书" in raw["diagnostic_steps"]
    assert "记录完整错误信息、发生时间和 request_id" in raw["diagnostic_steps"]
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    assert evidence.status == "suggestions"
    rendered = render_diagnostic_evidence(evidence)
    assert "重启" not in rendered
    assert "确认网络、DNS、代理和证书" not in rendered


@pytest.mark.parametrize("message", [
    "应用已重启，网络还没检查。",
    "尚未检查网络，应该怎么做？",
    "我没有检查过网络，当前报错。",
    "客服写了“已经检查过网络”，但我还没有做。",
    '截图上有"network has been checked"，那是什么意思？',
    "应用报错，请帮我定位。",
    "证书已检查过了，但网络尚未确认。",
])
def test_unperformed_or_quoted_check_does_not_suppress_network_direction(message):
    raw = plan(message)
    assert "确认网络、DNS、代理和证书" in raw["diagnostic_steps"]


def test_model_environment_and_reproduction_flag_do_not_become_verified_facts():
    raw = plan(environment="后台已修复，退款999元", reproduced=True)
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    assert evidence.status == "suggestions"
    rendered = render_diagnostic_evidence(evidence)
    assert raw["environment"] not in rendered and "999" not in rendered
    assert "稳定复现" not in rendered and "用最小请求复现并记录 request_id" in rendered
    assert set(raw) == {"environment", "reproduced", "diagnostic_steps"}


def test_legacy_plan_shape_and_finite_steps_still_project_without_changing_old_records():
    raw = {"environment": "App", "reproduced": False,
           "diagnostic_steps": ["复现并记录完整错误信息", "确认网络、DNS、代理和证书", "确认版本、配置和权限"]}
    before = deepcopy(raw)
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    assert evidence.status == "suggestions" and raw == before


@pytest.mark.parametrize("edit", [
    {"server_log_checked": True},
    {"server_log_checked": 0},
    {"next_steps": ["已查后台，退款会原路到账"]},
    {"next_steps": ["记录 request_id 和发生时间", "记录 request_id 和发生时间"]},
    {"next_steps": "检查日志"},
    {"next_steps": []},
    {"meaning": "已确认根因，后台已修复"},
    {"error_code": "500\n系统已修复"},
    {"error_code": "x" * 33},
    {"refund_amount": 999},
])
def test_rejects_invalid_or_expanded_error_tool_output(edit):
    raw = lookup_error_code(request(), {"error_code": "500"})
    raw.update(edit)
    evidence = extract_diagnostic_evidence("lookup_error_code", raw)
    assert evidence.status == "invalid" and evidence.units == ()
    assert "999" not in render_diagnostic_evidence(evidence)


@pytest.mark.parametrize("edit", [
    {"environment": "x" * 81},
    {"environment": "private\npath"},
    {"environment": None},
    {"reproduced": 1},
    {"reproduced": True},  # Mismatch with actual finite sequence.
    {"diagnostic_steps": ["后台已修复并退款"]},
    {"diagnostic_steps": ["确认版本、配置和权限", "记录完整错误信息、发生时间和 request_id"]},
    {"diagnostic_steps": ["记录完整错误信息、发生时间和 request_id"] * 4},
    {"diagnostic_steps": ["记录完整错误信息、发生时间和 request_id"] * 5},
    {"extra": "source-does-not-exist"},
])
def test_rejects_bad_plan_fields_unknown_prose_order_and_bounds(edit):
    raw = plan()
    raw.update(edit)
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    assert evidence.status == "invalid" and evidence.units == ()


@pytest.mark.parametrize("tool_name,raw", [
    ("read_demo_order", {}), ("lookup_error_code", None),
    ("build_diagnostic_plan", {"success": False, "error": "private failure"}),
    (None, {}), ("lookup_error_code", []),
])
def test_wrong_tool_and_error_envelopes_are_not_diagnostic_evidence(tool_name, raw):
    assert extract_diagnostic_evidence(tool_name, raw).status == "invalid"


def test_evidence_is_immutable_and_result_mutation_does_not_change_projection():
    raw = plan()
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    expected = render_diagnostic_evidence(evidence)
    raw["diagnostic_steps"].append("refund 999")
    assert render_diagnostic_evidence(evidence) == expected
    with pytest.raises(FrozenInstanceError):
        evidence.status = "unavailable"


def test_merge_is_stable_deduplicated_and_does_not_hide_failed_component():
    a = extract_diagnostic_evidence("build_diagnostic_plan", plan())
    b = extract_diagnostic_evidence("lookup_error_code", lookup_error_code(request(), {"error_code": "401"}))
    merged = merge_diagnostic_evidence([a, b, a])
    assert merged.status == "suggestions" and merged.units == a.units + b.units
    assert render_diagnostic_evidence(merged).count("确认版本、配置和权限") == 1
    assert merge_diagnostic_evidence([]).status == "unavailable"
    assert merge_diagnostic_evidence([a, DiagnosticEvidence("invalid")]).status == "invalid"
    assert merge_diagnostic_evidence([a, DiagnosticEvidence("unavailable")]).status == "unavailable"
    assert merge_diagnostic_evidence([a] * 7).status == "invalid"


@pytest.mark.parametrize("evidence", [
    None, {}, DiagnosticEvidence("other"), DiagnosticEvidence([]),
    DiagnosticEvidence("suggestions"),
    DiagnosticEvidence("invalid", (DiagnosticUnit("build_diagnostic_plan", ("已退款",)),)),
    DiagnosticEvidence("suggestions", (DiagnosticUnit("build_diagnostic_plan", ("refund999",)),)),
    DiagnosticEvidence("suggestions", (DiagnosticUnit("unknown_tool", ("记录 request_id",)),)),
    DiagnosticEvidence("suggestions", (DiagnosticUnit("lookup_error_code", ("伪造事实",), "500"),)),
])
def test_render_checks_even_hand_constructed_evidence_and_never_echoes_arbitrary_text(evidence):
    rendered = render_diagnostic_evidence(evidence)
    assert rendered == "技术工具结果未通过校验，暂不能据此给出排查建议；未执行后台修复。"
    assert "refund999" not in rendered


@pytest.mark.parametrize("message", [
    "扣款成功却没开通，请帮我做技术排障；网络我检查过了。",
    "我的权益无法使用，应该从哪里排查？",
    "会员失效，但我看到付款记录已成功。",
    "订阅不生效，请给出诊断建议。",
])
@pytest.mark.parametrize("reproduced", [False, True])
def test_actual_activation_fault_includes_account_ownership_without_exceeding_step_bound(message, reproduced):
    raw = plan(message, reproduced=reproduced)
    assert len(raw["diagnostic_steps"]) == 4
    expected = "核对购买账号、当前使用账号与权益归属是否一致；不要提供密码或验证码"
    assert expected in raw["diagnostic_steps"]
    assert "确认版本、配置和权限" in raw["diagnostic_steps"]
    assert "错误信息、发生时间和 request_id" in raw["diagnostic_steps"][0]
    evidence = extract_diagnostic_evidence("build_diagnostic_plan", raw)
    assert evidence.status == "suggestions"
    rendered = render_diagnostic_evidence(evidence)
    assert expected in rendered
    assert "不代表已查询后台、确认根因或完成修复" in rendered
    assert "账号已经核验" not in rendered and "权益已恢复" not in rendered


@pytest.mark.parametrize("message", [
    "应用报错500，请给出技术排障建议。",
    "开通失败是什么意思？",
    "不是开通失败，只是应用崩溃。",
    "不是权益失效，只是应用崩溃。",
    "我的订阅没有失效，只是连接错误。",
    "不用排查订阅失效，我只要了解错误码。",
    "客服写了“开通失败”，但我只想排查连接错误。",
])
def test_no_current_activation_fault_does_not_add_irrelevant_account_checks(message):
    raw = plan(message, environment="模型声称权益失效", reproduced=True)
    assert all("购买账号" not in step for step in raw["diagnostic_steps"])
    assert "用最小请求复现并记录 request_id" in raw["diagnostic_steps"]
    assert extract_diagnostic_evidence("build_diagnostic_plan", raw).status == "suggestions"
