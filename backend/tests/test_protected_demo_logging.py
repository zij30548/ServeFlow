"""Operational metadata only; no services, models or user text in logs."""
import io
import logging
import sys

import pytest

from core.protected_demo import ProtectedDemoLogFilter, configure_protected_logging


@pytest.fixture
def logging_state(monkeypatch):
    root = logging.getLogger()
    factory, level = logging.getLogRecordFactory(), root.level
    existing = [(handler, list(handler.filters)) for handler in root.handlers]
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    try:
        yield
    finally:
        logging.setLogRecordFactory(factory)
        root.setLevel(level)
        for handler, filters in existing:
            handler.filters[:] = filters


class UnsafeText:
    def __str__(self):
        raise AssertionError("a raw user message must not even be formatted")


def test_filter_discards_plaintext_arguments_urls_and_exception_input(logging_state):
    try:
        raise ValueError("private-user-question https://user:secret@private.invalid/token?q=secret")
    except ValueError:
        record = logging.LogRecord("tooling.tool_manager", logging.ERROR, "private-path", 1,
            "failed message=%r key=%s", (UnsafeText(), "private-key"), sys.exc_info())
    record.stack_info = "private stack text"
    assert ProtectedDemoLogFilter().filter(record)
    output = logging.Formatter("%(name)s %(levelname)s %(message)s").format(record)
    assert output == "protected_demo.tooling ERROR protected_demo_log_event module=tooling level=error exception=ValueError"
    assert record.exc_info is record.exc_text is record.stack_info is None


def test_unknown_logger_and_exception_names_are_never_logged(logging_state):
    secret_type = type("private_secret_name", (Exception,), {})
    record = logging.LogRecord("private.user.question", logging.WARNING, "", 1,
        UnsafeText(), (), (secret_type, secret_type("private-key"), None))
    assert ProtectedDemoLogFilter().filter(record)
    assert record.getMessage() == "protected_demo_log_event module=other level=warning exception=Exception"
    assert record.name == "protected_demo.other"


def test_protected_factory_covers_future_third_party_handlers_and_is_idempotent(logging_state):
    configure_protected_logging()
    first = logging.getLogRecordFactory()
    configure_protected_logging()
    assert logging.getLogRecordFactory() is first
    assert logging.getLogger().level == logging.WARNING
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    sdk = logging.getLogger("httpx.protected-log-test")
    old_handlers, old_propagate, old_level = sdk.handlers[:], sdk.propagate, sdk.level
    try:
        sdk.handlers[:] = [handler]
        sdk.propagate = False
        sdk.setLevel(logging.WARNING)
        sdk.warning("HTTP request %s api-key=%s", "https://private.invalid/workspace", "private-key")
    finally:
        sdk.handlers[:], sdk.propagate = old_handlers, old_propagate
        sdk.setLevel(old_level)
    assert stream.getvalue() == "protected_demo_log_event module=httpx level=warning exception=none\n"


def test_info_suppressed_and_legacy_modes_keep_their_original_logs(logging_state, monkeypatch):
    filtered = ProtectedDemoLogFilter()
    info = logging.LogRecord("core.skill_loader", logging.INFO, "", 1, "private-user-question", (), None)
    assert filtered.filter(info) is False
    configure_protected_logging()
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    record = logging.getLogRecordFactory()("core.skill_loader", logging.INFO, "", 1, "legacy %s", ("unchanged",), None)
    assert filtered.filter(record) is True
    assert record.getMessage() == "legacy unchanged"


def test_mutated_record_cannot_reintroduce_raw_text(logging_state):
    configure_protected_logging()
    record = logging.getLogRecordFactory()("core.skill_loader", logging.ERROR, "", 1, "private", (), None)
    record.msg, record.args = "new private %s", ("private-secret",)
    assert ProtectedDemoLogFilter().filter(record)
    assert record.getMessage() == "protected_demo_log_event module=other level=error exception=none"
