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

# Current-message guards only. Neither model entities nor assistant/history
# text may turn a number, a quoted command or a declined topic into a lookup.
_QUOTED = re.compile(r'“[^”\n]*”|「[^」\n]*」|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
_CODE_LABEL = re.compile(r"(?:HTTP\s*)?[45][0-9]{2}(?:\s+(?:Unauthorized|Forbidden|Not\s+Found|Internal\s+Server\s+Error))?", re.I)
_CODE = re.compile(r"(?<![A-Za-z0-9_-])(?:HTTP)?(?P<code>[45][0-9]{2})(?![A-Za-z0-9_-]|\.[0-9])", re.I)
_CODE_PREFIX = re.compile(
    r"(?:HTTP(?:\s*(?:status|error|code|状态码))*|错误(?:代码|码)?|状态码|报错?|返回|响应|提示|出现|排查|解释|查询|核对|"
    r"\b(?:returns?|responds?\s+with|error(?:\s+code)?|status(?:\s+code)?))\s*(?:为|是)?\s*[:：=]?\s*$", re.I)
_CODE_SUFFIX = re.compile(r"^\s*(?:错误|报错|故障|权限|认证|状态码|Forbidden\b|Unauthorized\b|Not\s+Found\b|Internal\s+Server\s+Error\b)", re.I)
_NON_CODE_SUFFIX = re.compile(r"^\s*(?:元|分|块|美元|人民币|页|号(?:订单|房|门)|型号|CNY\b|USD\b|dollars?\b|pages?\b)", re.I)
_NON_CODE_PREFIX = re.compile(r"(?:金额|价格|单价|型号|机型|编号|第|页码|\b(?:model|price|page))\s*$", re.I)
_TRANSLATE = re.compile(r"(?:翻译|怎么读|如何读|读音|这个词|这几个字|举例|引用|\b(?:translate|translation|pronounce|quoted?|example)\b)", re.I)
_DECLINED = re.compile(
    r"(?:不需要|不想|不用|不要|无需|无须|先别|暂不|(?:(?<!\w)|(?<=你)|(?<=您)|(?<=请)|(?<=现在)|(?<=暂时))别|不问|不查|不是(?:来)?|并非|没有|并未)\s*"
    r"(?:(?:再|先|现在|帮我|给我|做|进行|问|咨询|查|查询|核对|检查|处理|排查|诊断|修复|解决|解释|关于|返回|报错?|提示|出现)\s*){0,5}"
    r"(?:技术|故障|错误|报错|HTTP|接口|应用|客户端|服务器|(?:401|403|404|500)(?!\d))|"
    r"\b(?:do\s+not|don't|no\s+need\s+to|not\s+an?)\s+(?:(?:check|explain|diagnose|troubleshoot)\s+)?(?:HTTP(?:\s*[45][0-9]{2})?|error|API|fault|403|401|404|500)\b", re.I)
_RESOLVED = re.compile(r"(?:已经|已|早就).{0,6}(?:恢复|解决|正常)|\b(?:resolved|fixed|recovered)\b", re.I)
_REPORTED = re.compile(r"(?:文档|客服|示例|有人|对方).{0,8}(?:说|写|提到)|\b(?:document|agent|someone)\s+(?:said|wrote|mentioned)\b", re.I)
_MENTION = re.compile(r"(?:文档|示例|教程|注释).{0,8}(?:包含|写着|显示|列出)|\b(?:document|example)\s+(?:contains?|lists?)\b", re.I)
_OBSERVATION = re.compile(r"(?:报错?|返回|响应|提示|出现|\breturns?|\bresponds?\s+with)\s*(?:为|是)?\s*[:：=]?\s*(?:HTTP\s*)?$", re.I)
_ASK = re.compile(r"(?:解释|说明|核对|排查|诊断|检查|识别|怎么办|怎么查|如何|哪些|什么意思|什么含义|表示什么)|\b(?:explain|check|troubleshoot|diagnose|why|what|how)\b", re.I)
_SUBJECT = re.compile(r"(?:接口|应用|客户端|服务器|网络|登录|技术|故障|HTTP)|\b(?:API|App|server|network|login|endpoint)\b", re.I)
_FAULT = re.compile(r"(?:报错|故障|异常|失败|崩溃|无法|不能|排查|诊断|排障)|\b(?:error|fail(?:ed|ure|s)?|crash|troubleshoot|diagnose)\b", re.I)


@dataclass(frozen=True)
class DiagnosticRequest:
    required: bool = False
    error_code: str | None = None
    code_scope_unresolved: bool = False


def select_diagnostic_request(message) -> DiagnosticRequest:
    """Finite explicit technical requests, separate from order-read intent.

    Only a uniquely contextualized supported code selects the named lookup.
    Several codes or a generic fault use the existing bounded diagnostic plan;
    no code is selected by order, model entities, history or a numeric substring.
    This does not authenticate a user's reported error or diagnose its cause.
    """
    if type(message) is not str or not 1 <= len(message) <= 2000:
        return DiagnosticRequest()
    def quoted(match):
        inner = match.group()[1:-1].strip()
        return inner if _CODE_LABEL.fullmatch(inner) else " "
    text = _QUOTED.sub(quoted, message)
    clauses = []
    for clause in re.split(r"[，,。！？!?；;\n]|但是|不过|而是|只要", text):
        if (_TRANSLATE.search(clause) or _DECLINED.search(clause)
                or ((_RESOLVED.search(clause) or _REPORTED.search(clause) or _MENTION.search(clause)) and not _ASK.search(clause))):
            continue
        clauses.append(clause)
    requested = any(_ASK.search(clause) for clause in clauses)
    required, codes = False, set()
    for clause in clauses:
        current, group, previous_end = set(), [], None
        for match in _CODE.finditer(clause):
            prefix, suffix = clause[:match.start("code")], clause[match.end():]
            if _NON_CODE_PREFIX.search(prefix) or _NON_CODE_SUFFIX.search(suffix):
                continue
            contextual = bool((_CODE_PREFIX.search(prefix) or _CODE_SUFFIX.search(suffix))
                              and (requested or _FAULT.search(clause) or _OBSERVATION.search(prefix)))
            joined = previous_end is not None and re.fullmatch(r"\s*(?:和|或|及|与|、|/|and|or)\s*", clause[previous_end:match.start()], re.I)
            if not joined:
                if any(bound for _, bound in group):
                    current.update(code for code, _ in group)
                group = []
            group.append((match.group("code"), contextual))
            previous_end = match.end()
        if any(bound for _, bound in group):
            current.update(code for code, _ in group)
        codes.update(current)
        required = required or bool(current) or bool(_SUBJECT.search(clause) and _FAULT.search(clause))
    unique = next(iter(codes)) if len(codes) == 1 else None
    return DiagnosticRequest(required, unique if unique in _ERROR_CODES else None,
                             bool(codes) and (len(codes) != 1 or unique not in _ERROR_CODES))


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
            lines.append(f"参考含义：{_ERROR_CODES.get(unit.error_code, _UNKNOWN_ERROR)[0]}（本地错误码说明，不是本例根因结论）。")
        else:
            lines.append("可继续收集或核对：")
        for step in unit.steps:
            if step not in seen:
                seen.add(step)
                lines.append(f"- {step}")
    return "\n".join(lines)
