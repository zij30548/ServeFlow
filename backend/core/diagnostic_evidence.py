"""Bounded local diagnostic suggestions, never backend/financial facts.

Only the exact output shapes and finite steps of the two local technical tools
are projected. Their model-supplied input labels do not prove an environment,
a reproduced fault or an HTTP error observed on the user's system. This pure
module does not call tools, models, storage or external services.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


MAX_EVIDENCE_INPUTS = 6
MAX_UNITS = 6
MAX_STEPS = 4
_STATUSES = frozenset({"suggestions", "invalid", "unavailable"})
_ERROR_CODES = {
    "401": ("认证失败", ("确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态")),
    "403": ("权限不足", ("确认账号或套餐权限", "确认资源权限和 IP 白名单")),
    "404": ("资源或路径不存在", ("确认接口路径和环境", "确认资源标识是否正确")),
    "500": ("服务端处理异常", ("记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志")),
}
_UNKNOWN_ERROR = ("暂未识别的错误码", ("补充完整错误信息、发生时间和运行环境",))
_RECORD = "记录完整错误信息、发生时间和 request_id"
_LEGACY_RECORD = "复现并记录完整错误信息"
_NETWORK = "确认网络、DNS、代理和证书"
_NETWORK_REMAINING = "确认 DNS、代理和证书"
_VERSION = "确认版本、配置和权限"
_MINIMAL = "用最小请求复现并记录 request_id"
_REPRODUCED_RECORD = "用最小请求复现并记录完整错误信息、发生时间和 request_id"
_ENTITLEMENT = "核对购买账号、当前使用账号与权益归属是否一致；不要提供密码或验证码"


@dataclass(frozen=True)
class DiagnosticUnit:
    tool_name: str
    steps: tuple[str, ...]
    error_code: str | None = None


@dataclass(frozen=True)
class DiagnosticEvidence:
    status: str
    units: tuple[DiagnosticUnit, ...] = ()


def _text(value, maximum, *, empty=False):
    return (type(value) is str and (0 if empty else 1) <= len(value) <= maximum
            and (empty or bool(value.strip()))
            and not any(ord(ch) < 32 or ord(ch) == 127 or 0xD800 <= ord(ch) <= 0xDFFF for ch in value))


def _code(value):
    return type(value) is str and (value == "" or re.fullmatch(r"[A-Z0-9_.-]{1,32}", value) is not None)


def _steps(value):
    return (type(value) is list and 1 <= len(value) <= MAX_STEPS
            and all(_text(step, 100) for step in value) and len(set(value)) == len(value))


def _plan_sequence(steps):
    if type(steps) is not tuple or len(steps) > MAX_STEPS:
        return False
    legacy = any(steps == tuple([first] + ([network] if network is not None else []) + [_VERSION] + ([_MINIMAL] if minimal else []))
                 for first in (_RECORD, _LEGACY_RECORD)
                 for network in (_NETWORK, _NETWORK_REMAINING, None) for minimal in (True, False))
    activation = any(steps == (first, network, _VERSION, _ENTITLEMENT)
                     for first in (_RECORD, _REPRODUCED_RECORD)
                     for network in (_NETWORK, _NETWORK_REMAINING))
    return legacy or activation


def _valid_unit(unit):
    if (type(unit) is not DiagnosticUnit or type(unit.steps) is not tuple
            or type(unit.tool_name) is not str):
        return False
    if unit.tool_name == "lookup_error_code":
        return (_code(unit.error_code)
                and unit.steps == _ERROR_CODES.get(unit.error_code, _UNKNOWN_ERROR)[1])
    return unit.tool_name == "build_diagnostic_plan" and unit.error_code is None and _plan_sequence(unit.steps)


def _valid_evidence(evidence):
    return (type(evidence) is DiagnosticEvidence and type(evidence.status) is str
            and evidence.status in _STATUSES
            and type(evidence.units) is tuple and len(evidence.units) <= MAX_UNITS
            and all(_valid_unit(unit) for unit in evidence.units)
            and (evidence.status == "suggestions") == bool(evidence.units))


def extract_diagnostic_evidence(tool_name, result) -> DiagnosticEvidence:
    """Validate an actual local tool result, rejecting unknown fields/text.

    An unsupported name, an error envelope or malformed result is `invalid`;
    it is never converted into a successful generic diagnostic answer.
    """
    invalid = DiagnosticEvidence("invalid")
    if type(result) is not dict or type(tool_name) is not str:
        return invalid
    if tool_name == "lookup_error_code":
        if set(result) != {"error_code", "meaning", "next_steps", "server_log_checked"}:
            return invalid
        code = result["error_code"]
        if not _code(code) or result["server_log_checked"] is not False or not _steps(result["next_steps"]):
            return invalid
        meaning, steps = _ERROR_CODES.get(code, _UNKNOWN_ERROR)
        if result["meaning"] != meaning or tuple(result["next_steps"]) != steps:
            return invalid
        unit = DiagnosticUnit(tool_name, steps, code)
    elif tool_name == "build_diagnostic_plan":
        if set(result) != {"environment", "reproduced", "diagnostic_steps"}:
            return invalid
        if (not _text(result["environment"], 80, empty=True)
                or type(result["reproduced"]) is not bool or not _steps(result["diagnostic_steps"])):
            return invalid
        steps = tuple(result["diagnostic_steps"])
        if not _plan_sequence(steps) or ((_MINIMAL in steps or _REPRODUCED_RECORD in steps) != result["reproduced"]):
            return invalid
        # environment/reproduced intentionally do not enter the evidence unit.
        unit = DiagnosticUnit(tool_name, steps)
    else:
        return invalid
    return DiagnosticEvidence("suggestions", (unit,))


def merge_diagnostic_evidence(inputs: Iterable[DiagnosticEvidence]) -> DiagnosticEvidence:
    """Stable bounded deduplication; malformed/failed evidence is not hidden."""
    units = []
    try:
        for index, evidence in enumerate(inputs):
            if index >= MAX_EVIDENCE_INPUTS or not _valid_evidence(evidence):
                return DiagnosticEvidence("invalid")
            if evidence.status != "suggestions":
                return DiagnosticEvidence(evidence.status)
            for unit in evidence.units:
                if unit not in units:
                    units.append(unit)
                    if len(units) > MAX_UNITS:
                        return DiagnosticEvidence("invalid")
    except (TypeError, ValueError, AttributeError):
        return DiagnosticEvidence("invalid")
    return DiagnosticEvidence("suggestions", tuple(units)) if units else DiagnosticEvidence("unavailable")


def render_diagnostic_evidence(evidence: DiagnosticEvidence) -> str:
    """Fixed wording and finite local steps; no model prose or financial facts."""
    if not _valid_evidence(evidence) or evidence.status == "invalid":
        return "技术工具结果未通过校验，暂不能据此给出排查建议；未执行后台修复。"
    if evidence.status == "unavailable":
        return "尚未取得可核验的本地排查结果；未查询后台日志或执行修复。"
    lines = ["本地工具提供的排查建议，不代表已查询后台、确认根因或完成修复。"]
    seen = set()
    for unit in evidence.units:
        if unit.tool_name == "lookup_error_code":
            code = f" `{unit.error_code}`" if unit.error_code else "（未提供）"
            lines.append(f"工具输入错误码{code}，未经系统核验；以下为该输入对应的可能排查方向：")
        else:
            lines.append("可继续收集或核对：")
        for step in unit.steps:
            if step not in seen:
                seen.add(step)
                lines.append(f"- {step}")
    return "\n".join(lines)
