"""Controlled diagnostics checks: no model client construction or network."""

import asyncio
import json
import logging
import socket
from types import SimpleNamespace

import pytest
import httpx
from anthropic import APITimeoutError, APIConnectionError, AuthenticationError, RateLimitError

from agents.agent_orchestrator import AgentResponse, AgentType, Request, ResponseComposer
from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.safe_diagnostics import record_llm_failure


PRIVATE = "private-customer-text-cookie-and-provider-body"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Network is forbidden in controlled diagnostics tests")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket.socket, "connect_ex", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


class FakeClient:
    def __init__(self, *, content=None, error=None):
        self.calls = 0

        async def create(**kwargs):
            self.calls += 1
            if error is not None:
                raise error
            return SimpleNamespace(content=content, stop_reason="end_turn")

        self.messages = SimpleNamespace(create=create)


def recognizer(client):
    value = IntentRecognizer.__new__(IntentRecognizer)
    value.client, value.model = client, "controlled-model"
    return value


def compose(client):
    return ResponseComposer(client, "controlled-model").compose_result(
        Request(message=PRIVATE, user_id="controlled", conv_id="controlled"),
        [AgentResponse(AgentType.TECHNICAL, "主结论", True),
         AgentResponse(AgentType.BILLING, "辅助结论", True)],
    )


def assert_safe(caplog):
    assert PRIVATE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert all(record.exc_text is None for record in caplog.records)


@pytest.mark.parametrize(("error", "phase", "code", "kind"), [
    (TimeoutError(PRIVATE), "intent.transport", "deadline_exceeded", "TimeoutError"),
    (json.JSONDecodeError(PRIVATE, PRIVATE, 0), "intent.parse", "invalid_json", "JSONDecodeError"),
    (ValueError(PRIVATE), "intent.validate", "invalid_semantics", "ValueError"),
    (TypeError(PRIVATE), "composer.extract", "invalid_structure", "TypeError"),
])
def test_diagnostics_only_log_fixed_vocabulary(caplog, error, phase, code, kind):
    result = record_llm_failure(logging.getLogger("controlled"), phase, error, 12.9)
    assert result == {"phase": phase, "error_code": code, "error_type": kind, "elapsed_ms": 12}
    assert_safe(caplog)


def test_unknown_exception_name_and_message_are_not_rendered(caplog):
    class SensitiveException(Exception):
        def __str__(self):
            raise AssertionError("Exception text must never be evaluated")
        def __repr__(self):
            raise AssertionError("Exception repr must never be evaluated")
    SensitiveException.__name__ = PRIVATE
    result = record_llm_failure(logging.getLogger("controlled"), PRIVATE, SensitiveException(), float("inf"))
    assert result == {"phase": "unknown", "error_code": "unexpected_error", "error_type": "OtherError", "elapsed_ms": 0}
    assert_safe(caplog)


@pytest.mark.parametrize(("error_kind", "expected_code"), [
    (APITimeoutError, "deadline_exceeded"),
    (APIConnectionError, "connection_failed"),
    (AuthenticationError, "authentication_failed"),
    (RateLimitError, "rate_limited"),
])
def test_provider_exceptions_do_not_log_request_response_or_body(caplog, error_kind, expected_code):
    request = httpx.Request("POST", "https://example.invalid/", headers={"Authorization": PRIVATE})
    if error_kind in (APITimeoutError, APIConnectionError):
        error = error_kind(request=request)
    else:
        response = httpx.Response(401 if error_kind is AuthenticationError else 429, request=request, text=PRIVATE)
        error = error_kind(PRIVATE, response=response, body={"private": PRIVATE})
    result = record_llm_failure(logging.getLogger("controlled"), "intent.transport", error, 1)
    assert result["error_code"] == expected_code
    assert_safe(caplog)
    assert "example.invalid" not in caplog.text


@pytest.mark.parametrize(("elapsed", "expected"), [(-1, 0), (float("nan"), 0), (True, 0), (10**12, 3_600_000)])
def test_elapsed_is_bounded(caplog, elapsed, expected):
    assert record_llm_failure(logging.getLogger("controlled"), "intent.transport", TimeoutError(), elapsed)["elapsed_ms"] == expected


@pytest.mark.parametrize(("content", "error", "phase", "code"), [
    (None, TimeoutError(PRIVATE), "intent.transport", "deadline_exceeded"),
    ([{"type": "text", "text": PRIVATE}], None, "intent.parse", "invalid_json"),
    ([{"type": "text", "text": '{"intent":"query","confidence":true}'}], None, "intent.validate", "invalid_semantics"),
    ([{"type": "text", "text": '{"confidence":0.5}'}], None, "intent.validate", "invalid_structure"),
])
def test_intent_failures_preserve_fallback_and_single_call(caplog, content, error, phase, code):
    client = FakeClient(content=content, error=error)
    result = asyncio.run(recognizer(client)._llm_recognize(PRIVATE, None))
    assert result == {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}
    assert client.calls == 1
    assert f"phase={phase}" in caplog.text and f"error_code={code}" in caplog.text
    assert_safe(caplog)


def test_intent_success_has_no_failure_log(caplog):
    client = FakeClient(content=[{"type": "text", "text": '{"intent":"query","confidence":0.8,"reasoning":"ok"}'}])
    result = asyncio.run(recognizer(client)._llm_recognize(PRIVATE, None))
    assert result["intent"] is IntentCategory.QUERY and result["confidence"] == .8
    assert client.calls == 1 and not caplog.records


@pytest.mark.parametrize(("content", "error", "phase", "code"), [
    (None, TimeoutError(PRIVATE), "composer.transport", "deadline_exceeded"),
    ([], None, "composer.extract", "empty_output"),
    (42, None, "composer.extract", "invalid_structure"),
])
def test_composer_failure_preserves_deterministic_fallback(caplog, content, error, phase, code):
    client = FakeClient(content=content, error=error)
    result = asyncio.run(compose(client))
    assert result.content == "主结论\n\n补充说明：\n辅助结论" and result.status == "success"
    assert client.calls == 1
    assert f"phase={phase}" in caplog.text and f"error_code={code}" in caplog.text
    assert_safe(caplog)


@pytest.mark.parametrize("target", ["intent", "composer"])
def test_cancellation_propagates_without_retry_or_failure_log(caplog, target):
    client = FakeClient(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(recognizer(client)._llm_recognize(PRIVATE, None) if target == "intent" else compose(client))
    assert client.calls == 1 and not caplog.records


def test_composer_success_has_no_failure_log(caplog):
    client = FakeClient(content=[{"type": "text", "text": "合并结论"}])
    result = asyncio.run(compose(client))
    assert result.content == "合并结论" and result.status == "success"
    assert client.calls == 1 and not caplog.records
