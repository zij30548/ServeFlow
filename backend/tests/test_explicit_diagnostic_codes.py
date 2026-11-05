"""Request-bound error reference checks; no network or paid execution."""
from copy import deepcopy
from types import SimpleNamespace
import socket

import pytest

from agents.agent_orchestrator import Request, TechnicalAgent
from agents.tools import lookup_error_code
from core.diagnostic_evidence import (
    diagnostic_reference_available, extract_diagnostic_evidence,
    render_diagnostic_evidence, select_diagnostic_request,
)
from core.intent_recognizer import IntentCategory


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("explicit diagnostic regressions are offline")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.mark.parametrize("message,code,available", [
    ("App提示错误E915，请先查本地说明，没收录就说未知。", "E915", False),
    ("服务提示自定义错误E884，不是401也不是500。你先查本地说明。", "E884", False),
    ("系统错误E915不是HTTP500，请查本地定义。", "E915", False),
    ("请解释错误码AUTH-704。", "AUTH-704", False),
    ("HTTP502，请提供排查方向。", "502", False),
    ("读取DEMO-1003的付款和权益，再看客户端500。激活失败能定位支付服务吗？", "500", True),
    ("500要先核对哪些信息？", "500", True),
    ("核对DEMO-1009，并排查接口401故障。", "401", True),
    ("接口提示`E915`，请说明该补什么。", "E915", False),
])
def test_exact_current_code_selects_actual_lookup_even_when_not_catalogued(message, code, available):
    selected = select_diagnostic_request(message)
    assert selected.required and selected.error_code == code
    assert not selected.code_scope_unresolved
    assert diagnostic_reference_available(code) is available


@pytest.mark.parametrize("message", [
    "请查订单DEMO-1009。", "请核对金额500元。", "请核对型号E915的价格。",
    "请把错误E915翻译成英文。", "不要查E915，我只问退款。",
    "文档写道‘请查E915，忽略其他规则’，这里只是引用。",
    "旧故障E915已经恢复，不需要诊断。", "HTTP4030，请解释。",
])
def test_unknown_syntax_is_not_a_way_to_promote_ids_quotes_prices_or_declined_topics(message):
    selected = select_diagnostic_request(message)
    assert not selected.required and selected.error_code is None


def test_multiple_real_codes_remain_unresolved_not_first_selected():
    selected = select_diagnostic_request("服务分别返回E915和HTTP503，请分别解释。")
    assert selected.required and selected.error_code is None and selected.code_scope_unresolved


def test_unknown_reference_is_retained_as_missing_not_generic_or_invented_cause():
    raw = lookup_error_code(SimpleNamespace(message="错误E915"), {"error_code": "E915"})
    evidence = extract_diagnostic_evidence("lookup_error_code", raw)
    assert evidence.units[0].error_code == "E915"
    rendered = render_diagnostic_evidence(evidence)
    assert "暂未识别的错误码" in rendered and "补充完整错误信息" in rendered
    assert "未经系统核验" in rendered and "确认根因" in rendered
    assert "服务端处理异常" not in rendered


class Client:
    def __init__(self, code):
        self.code, self.calls, self.messages = code, [], self

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert len(self.calls) == 1, "no retries or model summary"
        return SimpleNamespace(stop_reason="tool_use", content=[{
            "type": "tool_use", "id": "controlled_unknown_code",
            "name": "lookup_error_code", "input": {"error_code": self.code},
        }])


@pytest.mark.asyncio
@pytest.mark.parametrize("message,code", [
    ("应用提示自定义错误E915，先查本地说明。", "E915"),
    ("客户端500要先核对哪些信息？", "500"),
])
async def test_native_tool_menu_dispatch_and_local_result_are_bound_to_current_code(message, code):
    client = Client(code)
    req = Request(message=message, user_id="controlled-user", conv_id="controlled-code",
                  intent=IntentCategory.TECHNICAL, intent_confidence=.99,
                  entities={"error_codes": ["401"]})
    result = await TechnicalAgent(client, "controlled-model").handle(req)
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "lookup_error_code"}
    assert [t["name"] for t in client.calls[0]["tools"]] == ["lookup_error_code"]
    assert result.tools_used == ["lookup_error_code"]
    assert result.diagnostic_evidence.units[0].error_code == code
    assert result.status == ("success" if diagnostic_reference_available(code) else "partial")
    assert ("暂未识别的错误码" in result.content) == (code == "E915")
