"""Narrow perimeter contract for an explicitly protected customer demo.

The reverse proxy authenticates invited visitors. Only after the private backend
Bearer is valid may its overwritten visitor header bind a demo browser session.
This is not public registration, a merchant account system or an arbitrary
forwarded-header trust mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
import re
from urllib.parse import urlsplit


MODE = "protected-demo"
VISITOR_HEADER = "x-serveflow-demo-user"
VISITOR_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_TRACE = re.compile(r"/trace/tool/[A-Za-z0-9_-]{1,128}\Z")
_ORDER = re.compile(r"/demo/orders/DEMO-[0-9]{4}\Z")
_HOST = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*)\Z")
_LOG_SANITIZED = object()


class ProtectedDemoError(ValueError):
    def __init__(self, code="protected_demo_configuration_invalid", status_code=503):
        self.code, self.status_code = code, status_code
        super().__init__(code)


def enabled():
    return os.getenv("ECHOMIND_ENV", "development").lower() == MODE


def _safe_log_record(record):
    """Do not format user text, provider URLs or exception messages at all.

    The protected demonstration intentionally retains only coarse operational
    events. In particular, a third-party exception is not a safe log field just
    because the application did not create its message.
    """
    if (getattr(record, "_protected_demo_marker", None) is _LOG_SANITIZED
            and type(record.msg) is str and record.msg == "protected_demo_log_event module=%s level=%s exception=%s"
            and type(record.args) is tuple and len(record.args) == 3
            and all(type(item) is str for item in record.args)
            and record.args[0] in {"api", "agents", "core", "memory", "monitor", "tooling", "uvicorn", "httpx", "anthropic", "other"}
            and record.args[1] in {"warning", "error", "critical"}
            and record.args[2] in {"none", "ValueError", "TypeError", "OSError", "RuntimeError", "TimeoutError", "KeyError", "Exception"}
            and record.name == "protected_demo." + record.args[0]
            and record.levelname == record.args[1].upper()
            and record.exc_info is None and record.exc_text is None and record.stack_info is None):
        return record
    module = record.name.split(".", 1)[0] if type(record.name) is str else "other"
    if module not in {"api", "agents", "core", "memory", "monitor", "tooling", "uvicorn", "httpx", "anthropic"}:
        module = "other"
    number = record.levelno if type(record.levelno) is int else logging.WARNING
    level = "critical" if number >= logging.CRITICAL else "error" if number >= logging.ERROR else "warning"
    exception = "none"
    if type(record.exc_info) is tuple and len(record.exc_info) == 3:
        kind = record.exc_info[0]
        exception = {ValueError: "ValueError", TypeError: "TypeError", OSError: "OSError",
            RuntimeError: "RuntimeError", TimeoutError: "TimeoutError", KeyError: "KeyError"}.get(kind, "Exception") if type(kind) is type else "Exception"
    record.name = "protected_demo." + module
    record.msg = "protected_demo_log_event module=%s level=%s exception=%s"
    record.args = (module, level, exception)
    record.levelno = number
    record.levelname = level.upper()
    record.exc_info = record.exc_text = record.stack_info = None
    record.__dict__.pop("message", None)
    record._protected_demo_marker = _LOG_SANITIZED
    return record


class ProtectedDemoLogFilter(logging.Filter):
    def filter(self, record):
        if not enabled():
            return True
        if type(record.levelno) is int and record.levelno < logging.WARNING:
            return False
        _safe_log_record(record)
        return True


def configure_protected_logging():
    """Protect existing and subsequently created handlers, only in this mode.

    A record factory removes raw content before any later SDK/ASGI handler can
    format it; the root-handler filter additionally suppresses informational
    logs. Outside protected-demo the factory leaves all existing behavior alone.
    """
    if not enabled():
        return
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    current = logging.getLogRecordFactory()
    if not getattr(current, "_serveflow_protected", False):
        def factory(*args, **kwargs):
            record = current(*args, **kwargs)
            return _safe_log_record(record) if enabled() else record
        factory._serveflow_protected = True
        logging.setLogRecordFactory(factory)
    for handler in root.handlers:
        if not any(isinstance(item, ProtectedDemoLogFilter) for item in handler.filters):
            handler.addFilter(ProtectedDemoLogFilter())


def monitor_snapshot(snapshot):
    """An HTTP response projection, not merely a frontend hiding convention."""
    agents, tools, alerts = {}, {}, []
    known_tools = {"knowledge_search", "search_knowledge_base", "inspect_request_context", "suggest_required_fields",
        "lookup_error_code", "build_diagnostic_plan", "check_billing_fields", "compare_amounts", "create_handoff_summary", "read_demo_order"}
    def number(value, *, count=False, maximum=9007199254740991):
        if ((type(value) is not int if count else type(value) not in {int, float})
                or not 0 <= value <= maximum):
            return None
        return value
    source = snapshot if type(snapshot) is dict else {}
    for key, target, states, duration in (("agent_stats", agents, {"healthy", "cooldown", "probing"}, "avg_ms"),
            ("tool_stats", tools, {"closed", "open", "half_open"}, "avg_latency_ms")):
        rows = source.get(key, {})
        if type(rows) is not dict:
            continue
        for identifier, row in list(rows.items())[:100]:
            if (type(identifier) is not str or type(row) is not dict
                    or (not re.fullmatch(r"(?:general|technical|billing|escalation)_[0-9]{1,2}", identifier)
                        if key == "agent_stats" else identifier not in known_tools)):
                continue
            state_key = "recovery_state" if key == "agent_stats" else "circuit_state"
            state = row.get(state_key)
            target[identifier] = {"total": number(row.get("total"), count=True),
                "success_rate": number(row.get("success_rate"), maximum=1), duration: number(row.get(duration)),
                state_key: state if type(state) is str and state in states else "unknown"}
            if key == "tool_stats":
                target[identifier]["consecutive_fails"] = number(row.get("consecutive_fails"), count=True)
    raw_alerts = source.get("active_alerts", [])
    if type(raw_alerts) is list:
        for row in raw_alerts[-10:]:
            if type(row) is not dict or row.get("resolved") is True:
                continue
            severity, metric = row.get("severity"), row.get("metric")
            severity = severity.value if isinstance(severity, Enum) else severity
            if type(severity) is not str or severity not in {"info", "warning", "error", "critical"} or type(metric) is not str:
                continue
            kind, _, identifier = metric.partition(":")
            if kind not in {"agent_success_rate", "agent_avg_ms", "tool_success_rate", "tool_avg_ms"}:
                continue
            if identifier not in (agents if kind.startswith("agent_") else tools):
                continue
            maximum = 1 if kind.endswith("success_rate") else 9007199254740991
            value, threshold = number(row.get("value"), maximum=maximum), number(row.get("threshold"), maximum=maximum)
            if value is not None and threshold is not None:
                alerts.append({"severity": severity, "metric": metric, "value": value, "threshold": threshold, "resolved": False})
    return {"agent_stats": agents, "tool_stats": tools, "active_alerts": alerts, "suggestions": []}


def public_origin(value):
    """One canonical HTTPS origin, never a request-supplied upstream address."""
    try:
        if (type(value) is not str or not 1 <= len(value) <= 300
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
                or any(c in value for c in "?#\\")):
            raise ValueError
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        if (value != "https://" + parsed.netloc or parsed.scheme != "https" or host is None or not _HOST.fullmatch(host)
                or host.endswith(".") or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or port == 443
                or port is not None and not 1 <= port <= 65535
                or parsed.netloc != host + (f":{port}" if port is not None else "")):
            raise ValueError
        return value
    except (TypeError, ValueError, UnicodeError):
        raise ProtectedDemoError() from None


@dataclass(frozen=True)
class ProtectedDemoConfig:
    origin: str

    @classmethod
    def from_env(cls):
        if not enabled():
            return None
        return cls(public_origin(os.getenv("SERVEFLOW_DEMO_PUBLIC_ORIGIN", "")))


def allowed_route(method, path):
    if method == "GET":
        return path in {"/health", "/demo/session", "/monitor"} or bool(_ORDER.fullmatch(path) or _TRACE.fullmatch(path))
    return method == "POST" and path == "/chat"


def authorize_proxy(request, account, config):
    """Reject before any resource/model work; do not accept browser identity."""
    path, method = request.url.path, request.method
    if not allowed_route(method, path):
        raise ProtectedDemoError("protected_demo_endpoint_disabled", 403)
    if method == "GET" and path == "/health":
        # Container health checks are internal and expose no role details when
        # tokens are configured. The public Nginx location is still protected.
        return None
    if not account:
        raise ProtectedDemoError("protected_demo_proxy_required", 401)
    if account.get("role") != "user" or account.get("user_id") != "protected-demo-proxy":
        raise ProtectedDemoError("protected_demo_proxy_role_invalid", 403)
    principal = request.headers.get(VISITOR_HEADER, "")
    if (len(request.headers.getlist(VISITOR_HEADER)) != 1
            or len(request.headers.getlist("x-forwarded-proto")) != 1
            or not VISITOR_PATTERN.fullmatch(principal)
            or request.headers.get("x-forwarded-proto") != "https"):
        raise ProtectedDemoError("protected_demo_proxy_required", 401)
    origin = request.headers.get("origin")
    if (origin is not None and origin != config.origin
            or request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}):
        raise ProtectedDemoError("demo_origin_rejected", 403)
    if request.query_params:
        raise ProtectedDemoError("protected_demo_query_rejected", 400)
    return principal
