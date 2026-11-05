"""Perimeter and signed-session contracts; no server/model/database calls."""
import json
import socket

import pytest
from starlette.requests import Request

from core.access import identity
from core.protected_demo import ProtectedDemoConfig, ProtectedDemoError, allowed_route, authorize_proxy, public_origin
from demo_business.session import DemoSessionError, DemoSessionService, ProtectedDemoSessionService
from demo_business.store import DemoDatabaseConfig, DemoStoreError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k): raise AssertionError("ordinary tests cannot contact services")
    for n in ("connect", "connect_ex"): monkeypatch.setattr(socket.socket, n, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.mark.parametrize("origin", ["https://demo.example.com", "https://localhost:8443", "https://127.0.0.1:9443"])
def test_exact_https_origin(origin):
    assert public_origin(origin) == origin


@pytest.mark.parametrize("origin", ["", None, "http://demo.example.com", "HTTPS://demo.example.com", "https://DEMO.example.com",
    "https://demo.example.com/", "https://demo.example.com/a", "https://demo.example.com:443", "https://demo.example.com:08443",
    "https://demo.example.com:0", "https://demo.example.com:65536", "https://u:p@demo.example.com", "https://demo.example.com?q=1",
    "https://demo.example.com#x", "https://*.example.com", "https://例子.com", "https://a..example.com", "https://demo.example.com.",
    " https://demo.example.com", "https://demo.example.com\\evil"])
def test_origin_rejects_ambiguous_or_untrusted_form(origin):
    with pytest.raises(ProtectedDemoError): public_origin(origin)


def test_mode_is_explicit_and_tokens_never_optional(monkeypatch):
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    monkeypatch.delenv("ECHOMIND_API_TOKENS", raising=False)
    assert ProtectedDemoConfig.from_env() is None and identity("") is None
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    with pytest.raises(ValueError): identity("")
    with pytest.raises(ProtectedDemoError): ProtectedDemoConfig.from_env()
    monkeypatch.setenv("SERVEFLOW_DEMO_PUBLIC_ORIGIN", "https://demo.example.com")
    assert ProtectedDemoConfig.from_env().origin == "https://demo.example.com"
    token = "controlled-proxy-token-1234567890"
    monkeypatch.setenv("ECHOMIND_API_TOKENS", json.dumps({token: {"role": "user", "user_id": "protected-demo-proxy"}}))
    assert identity("Bearer " + token)["role"] == "user"
    assert identity("Bearer forged") == {}


def request(path="/chat", method="POST", headers=None, query=b""):
    pairs = [("x-serveflow-demo-user", "alice"), ("x-forwarded-proto", "https")]
    if headers is not None: pairs = headers
    return Request({"type": "http", "method": method, "path": path, "query_string": query,
        "scheme": "http", "server": ("api", 8816), "headers": [(k.encode(), v.encode()) for k, v in pairs]})


def test_proxy_identity_only_after_bearer_and_fixed_method_path():
    cfg = ProtectedDemoConfig("https://demo.example.com")
    account = {"role": "user", "user_id": "protected-demo-proxy"}
    assert authorize_proxy(request(), account, cfg) == "alice"
    assert authorize_proxy(request("/health", "GET", headers=[]), {}, cfg) is None
    for req, identity_value in [(request(), {}), (request(), None), (request(), {"role": "admin"}),
            (request(), {"role": "user", "user_id": "ordinary-customer"}),
            (request(headers=[]), account), (request(headers=[("x-serveflow-demo-user", "alice"), ("x-forwarded-proto", "http")]), account),
            (request(headers=[("x-serveflow-demo-user", "alice,bob"), ("x-forwarded-proto", "https")]), account),
            (request(headers=[("x-serveflow-demo-user", "alice"), ("x-serveflow-demo-user", "bob"), ("x-forwarded-proto", "https")]), account),
            (request(headers=[("x-serveflow-demo-user", "alice"), ("x-forwarded-proto", "https"), ("origin", "https://evil.example")]), account),
            (request(headers=[("x-serveflow-demo-user", "alice"), ("x-forwarded-proto", "https"), ("sec-fetch-site", "same-site")]), account),
            (request(query=b"customer_id=demo-customer-b"), account)]:
        with pytest.raises(ProtectedDemoError): authorize_proxy(req, identity_value, cfg)
    for path, method in [("/docs", "GET"), ("/openapi.json", "GET"), ("/eval/run", "POST"), ("/knowledge/upload", "POST"),
                         ("/knowledge/add", "POST"), ("/skills/reload", "POST"), ("/mcp", "POST"), ("/trace/tools", "GET"),
                         ("/search", "POST"), ("/chat", "GET"), ("/monitor", "POST")]:
        assert not allowed_route(method, path)
        with pytest.raises(ProtectedDemoError): authorize_proxy(request(path, method), account, cfg)
    assert allowed_route("GET", "/demo/orders/DEMO-1001")
    assert allowed_route("GET", "/trace/tool/a0b1_c2-d3")


def test_protected_v2_binds_visitor_origin_and_separate_browser_sessions():
    service = ProtectedDemoSessionService(b"x" * 32, "https://demo.example.com", lambda: 1000)
    first, a = service.issue(principal="alice")
    second, b = service.issue(principal="alice")
    assert a["v"] == 2 and a["owner"] == b["owner"] and a["sid"] != b["sid"]
    assert "alice" not in json.dumps(a)
    assert service.cookie_secure and service.read(first, principal="alice") == a
    assert service.workspace_id(a) != service.workspace_id(b)
    for token, principal in [(first, "bob"), (second, "bob"), (first, ""), (first, "alice,bob")]:
        with pytest.raises(DemoSessionError): service.read(token, principal=principal)
    other_origin = ProtectedDemoSessionService(b"x" * 32, "https://other.example.com", lambda: 1000)
    with pytest.raises(DemoSessionError): other_origin.read(first, principal="alice")
    old = DemoSessionService(b"x" * 32, ("http://localhost:5173",), lambda: 1000)
    old_token, _ = old.issue()
    with pytest.raises(DemoSessionError): service.read(old_token, principal="alice")
    with pytest.raises(DemoSessionError): old.read(first)


def test_protected_csrf_and_expiry_preserve_old_behavior():
    now = [1000]
    service = ProtectedDemoSessionService(b"x" * 32, "https://demo.example.com", lambda: now[0])
    token, claims = service.issue(principal="alice")
    _, other = service.issue(principal="alice")
    service.verify_write(claims, service.origins[0], service.csrf(claims))
    for origin, csrf in [(None, service.csrf(claims)), ("http://demo.example.com", service.csrf(claims)),
                          (service.origins[0], service.csrf(other))]:
        with pytest.raises(DemoSessionError): service.verify_write(claims, origin, csrf)
    now[0] += 28800
    with pytest.raises(DemoSessionError, match="expired"): service.read(token, principal="alice")


def test_protected_factory_keeps_private_file_checks_and_old_production_rejection(monkeypatch, tmp_path):
    key = tmp_path / "key"; key.write_bytes(b"x" * 32); key.chmod(0o600)
    monkeypatch.setenv("SERVEFLOW_DEMO_SESSION_KEY_FILE", str(key))
    monkeypatch.setenv("SERVEFLOW_DEMO_PUBLIC_ORIGIN", "https://demo.example.com")
    monkeypatch.setenv("SERVEFLOW_DEMO_ORIGINS", "https://ignored.example.com")
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    result = DemoSessionService.from_env()
    assert result.protected and result.cookie_secure and result.origins == ("https://demo.example.com",)
    key.chmod(0o444)
    with pytest.raises(DemoSessionError, match="key_invalid"): DemoSessionService.from_env()
    key.chmod(0o600)
    monkeypatch.setenv("ECHOMIND_ENV", "production")
    with pytest.raises(DemoSessionError, match="local_development"): DemoSessionService.from_env()


def test_mysql_service_name_only_in_protected_mode(monkeypatch):
    for mode in ["development", "production", "protected-demo"]:
        monkeypatch.setenv("ECHOMIND_ENV", mode)
        for host in ["localhost", "127.0.0.1"]:
            assert DemoDatabaseConfig.parse(f"mysql://u:p@{host}/serveflow_demo_test").host == host
        if mode == "protected-demo":
            assert DemoDatabaseConfig.parse("mysql://u:p@mysql/serveflow_demo_test").host == "mysql"
        else:
            with pytest.raises(DemoStoreError): DemoDatabaseConfig.parse("mysql://u:p@mysql/serveflow_demo_test")
        for host in ["db", "mysql.evil.example", "example.com", "10.0.0.1"]:
            with pytest.raises(DemoStoreError): DemoDatabaseConfig.parse(f"mysql://u:p@{host}/serveflow_demo_test")
