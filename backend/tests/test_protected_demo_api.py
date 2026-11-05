"""HTTP and trace-owner checks with Mock business/model and in-memory Redis."""
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio

with patch("dotenv.load_dotenv", return_value=False):
    from api import main
from core.concurrency import KeyedLocks
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core import protected_demo_quota as quota
from demo_business.session import COOKIE_NAME, ProtectedDemoSessionService
from memory.conversation_memory import MemoryManager
from test_conversation_memory import MemoryCollection

ORIGIN = "https://demo.example.com"
TOKEN = "controlled-proxy-token-1234567890"


def headers(user="alice"):
    return {"Authorization": "Bearer " + TOKEN, "X-ServeFlow-Demo-User": user, "X-Forwarded-Proto": "https"}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k): raise AssertionError("ordinary tests cannot contact services")
    for n in ("connect", "connect_ex"): monkeypatch.setattr(socket.socket, n, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})


@pytest_asyncio.fixture
async def services(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")
    monkeypatch.setenv("SERVEFLOW_DEMO_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.setenv("ECHOMIND_API_TOKENS", json.dumps({TOKEN: {"user_id": "protected-demo-proxy", "role": "user"}}))
    path = tmp_path.resolve() / "usage"
    monkeypatch.setenv("SERVEFLOW_DEMO_QUOTA_DIR", str(path))
    monkeypatch.setenv("SERVEFLOW_DEMO_MODEL_LIMIT", "1")
    usage = quota.QuotaStore(path, 1); usage.initialize()
    store = SimpleNamespace(list_sample_orders=lambda c: [{"order_id": "DEMO-1001", "scenario": "controlled"}],
        read_order=AsyncMock(return_value={"source": "serveflow-mysql-demo-v1", "is_demo": True, "as_of": "2026-10-25T00:00:00Z", "status": "not_found", "order": None}))
    sessions = ProtectedDemoSessionService(b"x" * 32, ORIGIN)
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._request_locks = KeyedLocks()
    memory._episodic, memory._profile = MemoryCollection(), MemoryCollection()
    captured, traces = [], {}
    intent = SimpleNamespace(intent=IntentCategory.ORDER_STATUS, intent_group="query", urgency=UrgencyLevel.LOW,
        confidence=.9, entities={}, source_scores={}, embedding_source="controlled")
    async def run(req):
        captured.append(req)
        traces[req.request_id] = {"request_id": req.request_id, "tool_calls": [{"tool_name": "controlled"}]}
        return SimpleNamespace(request_id=req.request_id, response="controlled HTTP wiring only", intent=req.intent,
            agent_type=SimpleNamespace(value="general"), agent_types=[], primary_agent=None, supporting_agents=[], tools_used=[], tool_traces=[],
            routing_reason="controlled", routing_confidence=.9, escalated=False, status="success", retryable=False, error_code=None)
    orc = SimpleNamespace(recognize_intent=AsyncMock(return_value=intent), run=AsyncMock(side_effect=run),
        get_tool_trace=lambda k: traces.get(k), get_stats=lambda: {})
    for name, value in {"_demo_store": store, "_demo_sessions": sessions, "_memory": memory, "_orchestrator": orc,
        "_profile_updater": None, "_monitor": SimpleNamespace(_prom={}, summary=lambda: {"agent_stats": {}, "tool_stats": {}})}.items():
        monkeypatch.setattr(main, name, value)
    try: yield SimpleNamespace(store=store, sessions=sessions, memory=memory, orc=orc, captured=captured, usage=usage, intent=intent)
    finally: await memory._redis.aclose()


def client(user="alice"):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url=ORIGIN, headers=headers(user))


async def bootstrap(http):
    response = await http.get("/demo/session")
    assert response.status_code == 200
    return response.json(), response


def body(key="one", **extra):
    return {"message": "查询DEMO-1001", "user_id": "forged-workspace", "conv_id": "controlled-conversation", "client_request_id": key, **extra}


def csrf(data): return {"Origin": ORIGIN, "X-ServeFlow-Demo-CSRF": data["csrf_token"]}


@pytest.mark.asyncio
async def test_secure_cookie_csrf_server_binding_and_idempotent_receipt(services):
    async with client() as http:
        b, initial = await bootstrap(http)
        cookie = initial.headers["set-cookie"]
        assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
        assert "alice" not in json.dumps(b)
        assert services.sessions.read(http.cookies.get(COOKIE_NAME), principal="alice")["v"] == 2
        assert (await http.post("/chat", json=body())).status_code == 403
        services.orc.recognize_intent.assert_not_called()
        good = await http.post("/chat", json=body(), headers=csrf(b))
        assert good.status_code == 200
        req = services.captured[0]
        assert req.user_id == b["workspace_user_id"] and req.demo_customer_id == "demo-customer-a"
        assert req.demo_order_reader == services.store.read_order
        assert (await http.post("/chat", json=body(), headers=csrf(b))).json() == good.json()
        assert services.orc.run.await_count == 1
        trace = await http.get("/trace/tool/" + good.json()["request_id"])
        assert trace.status_code == 200 and trace.json()["found"]
        assert services.usage.status()["used"] == 0


@pytest.mark.asyncio
async def test_monitor_http_returns_only_numeric_whitelisted_aggregate(services, monkeypatch):
    private = "PRIVATE_USER_MESSAGE key=PRIVATE_KEY https://private.invalid/token"
    raw = {"agent_stats": {"general_0": {"total": 3, "success_rate": 2 / 3, "avg_ms": 120,
        "recovery_state": "healthy", "role": private, "workflow": [private], "tool_scope": [private],
        "model": private, "last_error": private, "user_message": private}, private: {"total": 9}},
        "tool_stats": {"knowledge_search": {"total": 2, "success_rate": .5, "avg_latency_ms": 40,
            "consecutive_fails": 1, "circuit_state": "closed", "input": {"query": private}}, private: {}},
        "active_alerts": [{"severity": "warning", "metric": "tool_success_rate:knowledge_search",
            "value": .5, "threshold": .95, "message": private, "resolved": False},
            {"severity": "warning", "metric": private, "value": .5, "threshold": .95, "message": private}],
        "suggestions": [{"title": private, "action": private}], "last_error": private}
    monkeypatch.setattr(main._monitor, "summary", lambda: raw)
    async with client() as http:
        response = await http.get("/monitor")
        assert response.status_code == 200
        data = response.json()
        assert "PRIVATE" not in response.text and "private.invalid" not in response.text
        assert data["agent_stats"] == {"general_0": {"total": 3, "success_rate": 2 / 3, "avg_ms": 120, "recovery_state": "healthy"}}
        assert data["tool_stats"]["knowledge_search"]["consecutive_fails"] == 1
        assert data["active_alerts"] == [{"severity": "warning", "metric": "tool_success_rate:knowledge_search", "value": .5, "threshold": .95, "resolved": False}]
        assert data["suggestions"] == []
        assert raw["agent_stats"]["general_0"]["role"] == private


@pytest.mark.asyncio
async def test_legacy_monitor_response_is_not_reinterpreted(services, monkeypatch):
    monkeypatch.setenv("ECHOMIND_ENV", "development")
    monkeypatch.delenv("ECHOMIND_API_TOKENS")
    raw = {"agent_stats": {}, "tool_stats": {}, "suggestions": [{"action": "original local operator detail"}]}
    monkeypatch.setattr(main._monitor, "summary", lambda: raw)
    async with client() as http:
        assert (await http.get("/monitor")).json() == raw


@pytest.mark.asyncio
async def test_direct_forged_proxy_and_disabled_routes_never_reach_model_or_store(services):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url=ORIGIN) as raw:
        for h in [{}, {"X-ServeFlow-Demo-User": "alice", "X-Forwarded-Proto": "https"}, {"Authorization": "Bearer " + TOKEN},
                  {**headers(), "Authorization": "Bearer forged"}]:
            assert (await raw.post("/chat", json=body(), headers=h)).status_code == 401
    async with client() as http:
        for method, path in [("POST", "/eval/run"), ("POST", "/knowledge/add"), ("POST", "/knowledge/upload"),
            ("POST", "/skills/reload"), ("POST", "/mcp"), ("GET", "/trace/tools"), ("GET", "/docs"), ("GET", "/openapi.json"), ("POST", "/monitor")]:
            response = await http.request(method, path, content=b"untrusted body")
            assert response.status_code == 403 and response.json()["detail"] == "protected_demo_endpoint_disabled"
        assert (await http.get("/demo/session", headers={"Origin": "https://evil.example"})).status_code == 403
        assert (await http.get("/demo/orders/DEMO-1001?customer_id=demo-customer-b")).status_code == 400


@pytest.mark.asyncio
async def test_ordinary_user_bearer_cannot_assert_trusted_proxy_identity(services, monkeypatch):
    ordinary_token = "controlled-ordinary-user-token-1234567890"
    monkeypatch.setenv("ECHOMIND_API_TOKENS", json.dumps({
        TOKEN: {"user_id": "protected-demo-proxy", "role": "user"},
        ordinary_token: {"user_id": "ordinary-customer", "role": "user"}}))
    async with client() as http:
        response = await http.post("/chat", json=body(), headers={"Authorization": "Bearer " + ordinary_token})
        assert response.status_code == 403
        assert response.json()["detail"] == "protected_demo_proxy_role_invalid"
        services.orc.recognize_intent.assert_not_called()
        services.orc.run.assert_not_called()
        services.store.read_order.assert_not_called()
    services.orc.recognize_intent.assert_not_called(); services.orc.run.assert_not_called(); services.store.read_order.assert_not_called()


@pytest.mark.asyncio
async def test_visitors_cannot_reuse_cookie_csrf_or_tool_trace(services):
    async with client("alice") as a, client("bob") as b:
        aa, _ = await bootstrap(a)
        good = await a.post("/chat", json=body(), headers=csrf(aa)); assert good.status_code == 200
        request_id = good.json()["request_id"]
        stolen = a.cookies.get(COOKIE_NAME)
        assert (await b.get("/demo/orders/DEMO-1001", headers={"Cookie": f"{COOKIE_NAME}={stolen}"})).status_code == 401
        bb, _ = await bootstrap(b)
        assert aa["workspace_user_id"] != bb["workspace_user_id"]
        assert (await b.get("/trace/tool/" + request_id)).json() == {"request_id": request_id, "found": False, "trace": {}}
        assert services.orc.get_tool_trace(request_id)  # Global fallback would leak.
        assert (await b.post("/chat", json=body("b"), headers=csrf(aa))).status_code == 403
        result = await b.post("/chat", json=body("b", user_id=aa["workspace_user_id"]), headers=csrf(bb))
        assert result.status_code == 200 and services.captured[-1].user_id == bb["workspace_user_id"]
        assert not (await a.get("/trace/tool/" + result.json()["request_id"])).json()["found"]
        rotated = await a.get("/demo/session", headers={"X-ServeFlow-Demo-User": "bob"})
        assert rotated.status_code == 200 and rotated.json()["workspace_user_id"] not in {aa["workspace_user_id"], bb["workspace_user_id"]}


@pytest.mark.asyncio
async def test_exhaustion_blocks_chat_but_preserves_readonly_functions(services):
    op = services.usage.reserve(model=quota.MODEL, phase="provider", payload_sha256="a" * 64)
    services.usage.finish(op, "received", http_status=200)
    async with client() as http:
        b, _ = await bootstrap(http)
        response = await http.post("/chat", json=body(), headers=csrf(b))
        assert response.status_code == 429 and response.json()["detail"] == "demo_model_quota_exhausted"
        for path in ["/health", "/monitor", "/demo/orders/DEMO-1001"]: assert (await http.get(path)).status_code == 200
    services.orc.recognize_intent.assert_not_called(); services.orc.run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["intent", "run"])
async def test_swallowed_model_quota_error_cannot_commit_a_normal_response(services, phase):
    def fail(): quota.block_request(quota.ProtectedDemoQuotaError("demo_model_quota_exhausted"))
    if phase == "intent":
        async def recognize(*a, **k): fail(); return services.intent
        services.orc.recognize_intent.side_effect = recognize
    else:
        original = services.orc.run.side_effect
        async def run(req):
            response = await original(req); fail(); return response
        services.orc.run.side_effect = run
    async with client() as http:
        b, _ = await bootstrap(http)
        response = await http.post("/chat", json=body(), headers=csrf(b))
        assert response.status_code == 429 and response.json()["error_code"] == "demo_model_quota_exhausted"
        assert await services.memory.get_turn_receipt(b["workspace_user_id"], "controlled-conversation", "one") is None
        assert await services.memory._get_working_memory(b["workspace_user_id"], "controlled-conversation") == []
        if phase == "intent": services.orc.run.assert_not_called()


@pytest.mark.asyncio
async def test_optional_trace_owner_is_atomic_and_old_trace_is_not_authorization(services):
    memory = services.memory
    response = {"status": "success", "request_id": "controlled-owned", "response": "controlled"}
    await memory.commit_turn("alice", "c", "one", "controlled", response, {"private": "trace"}, trace_owner="alice")
    assert await memory.get_trace_for_user("controlled-owned", "alice") == {"private": "trace"}
    assert await memory.get_trace_for_user("controlled-owned", "bob") is None
    await memory._redis.delete("trace_owner:controlled-owned")
    assert await memory.get_trace_for_user("controlled-owned", "alice") is None
    assert await memory.get_trace("controlled-owned") == {"private": "trace"}
    legacy = dict(response, request_id="controlled-legacy")
    await memory.commit_turn("alice", "c", "two", "controlled", legacy, {"legacy": True})
    assert not await memory._redis.exists("trace_owner:controlled-legacy")
    assert await memory.get_trace_for_user("controlled-legacy", "alice") is None
    with pytest.raises(ValueError):
        await memory.commit_turn("alice", "c", "three", "controlled", dict(response, request_id="wrong"), {"secret": True}, trace_owner="bob")
    assert await memory.get_trace("wrong") is None
    for bad in ["../owned", "", "x" * 129]: assert await memory.get_trace_for_user(bad, "alice") is None


@pytest.mark.asyncio
async def test_exhausted_quota_still_returns_committed_same_operation_without_resending(services):
    async with client() as http:
        b, _ = await bootstrap(http)
        first = await http.post("/chat", json=body(), headers=csrf(b))
        assert first.status_code == 200 and services.orc.run.await_count == 1
        op = services.usage.reserve(model=quota.MODEL, phase="provider", payload_sha256="a" * 64)
        services.usage.finish(op, "received", http_status=200)
        again = await http.post("/chat", json=body(), headers=csrf(b))
        assert again.status_code == 200 and again.json() == first.json()
        assert services.orc.run.await_count == 1 and services.orc.recognize_intent.await_count == 1
        blocked = await http.post("/chat", json=body("two"), headers=csrf(b))
        assert blocked.status_code == 429
