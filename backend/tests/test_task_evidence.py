"""Controlled tool-evidence and one-attempt judge checks; no live services."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.tools import demo_order_tool, make_tool
from demo_business.store import load_fixture
from evaluation.task_evidence import (
    CaptureTools, EvidenceError, SingleAttemptJudge, build_judge_context,
    deterministic_checks, fixture_order_result,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tool evidence tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


class Agent:
    def __init__(self, role="general", tools=None):
        self.agent_type = SimpleNamespace(value=role)
        self.tools = tools or {"read_demo_order": demo_order_tool()}
        self.profile = SimpleNamespace(tool_scope=tuple(self.tools))

    def get_tools(self):
        return dict(self.tools)


def orc(*agents):
    pools = {}
    for agent in agents:
        pools.setdefault(agent.agent_type.value, []).append(agent)
    return SimpleNamespace(_pool=pools)


def request(customer="demo-customer-a", reader=None):
    async def fixture_reader(customer_id, order_id):
        return fixture_order_result(load_fixture(), customer_id, order_id)
    return SimpleNamespace(demo_customer_id=customer, demo_order_reader=reader or fixture_reader)


def case(order="DEMO-1004", customer="demo-customer-a"):
    return {"id": "t02", "input": {"message": f"查询{order}", "history": [], "customer_id": customer},
            "expected": {"required_agents": [], "required_tools": ["read_demo_order"],
                         "required_facts": ["退款额未知"], "forbidden_claims": ["退款额为零"]},
            "references": [{"source": "fixture", "table": "refunds", "row_id": "REF-1004", "fields": {"amount_minor": None}}]}


def result(capture, *, status="success", answer="合成演示资料，金额仍未知。", traces=None):
    if traces is None:
        traces = [{"agent_type": call["agent"], "tool_name": call["name"],
                   "input": deepcopy(call["input"]), "success": call["success"],
                   "result_success": call["result_success"], "tool_use_id": f"tool-{call['sequence']}"}
                  for call in capture.calls]
    roles = list(dict.fromkeys(call["agent"] for call in capture.calls)) or ["general"]
    return SimpleNamespace(tool_traces=traces, status=status, response=answer,
                           agent_types=[SimpleNamespace(value=role) for role in roles],
                           agent_type=SimpleNamespace(value=roles[0]))


def successful_capture(order="DEMO-1004", customer="demo-customer-a", *, reader=None):
    agent = Agent()
    capture = CaptureTools("t02", 0).install(orc(agent))
    value = asyncio.run(agent.get_tools()["read_demo_order"].handler(request(customer, reader), {"order_id": order}))
    capture.restore()
    execution = result(capture)
    return capture, execution, capture.reconcile(execution), value


def test_capture_preserves_original_result_and_get_tools_and_copies_evidence():
    agent = Agent()
    before = agent.get_tools
    capture = CaptureTools("t02", 0).install(orc(agent))
    stored = fixture_order_result(load_fixture(), "demo-customer-a", "DEMO-1004")
    reader = AsyncMock(return_value=stored)
    value = asyncio.run(agent.get_tools()["read_demo_order"].handler(request(reader=reader), {"order_id": "DEMO-1004"}))
    assert value is stored
    stored["order"]["refunds"][0]["amount_minor"] = 0
    assert capture.calls[0]["result"]["order"]["refunds"][0]["amount_minor"] is None
    assert capture.calls[0]["trusted_customer_id"] == "demo-customer-a"
    capture.restore()
    assert agent.get_tools == before and "get_tools" not in vars(agent)
    assert capture.reconcile(result(capture))["blockers"] == []


def test_parallel_same_tool_keeps_role_context_and_allows_independent_same_ids():
    active = ContextVar("controlled-role", default=None)
    @contextmanager
    def context(role):
        token = active.set(role)
        try:
            yield
        finally:
            active.reset(token)
    roles = []
    async def reader(customer, order):
        before = active.get()
        await asyncio.sleep(0)
        assert active.get() == before
        roles.append(before)
        return fixture_order_result(load_fixture(), customer, order)
    async def run():
        first, second = Agent("general"), Agent("billing")
        capture = CaptureTools("parallel", 0, call_context=context).install(orc(first, second))
        await asyncio.gather(*(a.get_tools()["read_demo_order"].handler(request(reader=reader), {"order_id": "DEMO-1001"}) for a in (first, second)))
        observed = capture.reconcile(result(capture))
        capture.restore()
        assert observed["blockers"] == []
        assert {call["agent"] for call in observed["calls"]} == {"general", "billing"}
    asyncio.run(run())
    assert set(roles) == {"general", "billing"} and active.get() is None


def test_schema_and_permission_denial_never_calls_original_handler_or_stores_unknown_input():
    reader = AsyncMock()
    agent = Agent()
    capture = CaptureTools("denied", 0).install(orc(agent))
    with pytest.raises(EvidenceError, match="tool_input_invalid"):
        asyncio.run(agent.get_tools()["read_demo_order"].handler(request(reader=reader), {"order_id": "DEMO-1001", "password": "do-not-store-this"}))
    assert reader.await_count == 0 and "do-not-store-this" not in json.dumps(capture.calls)
    capture.restore()
    technical = Agent("technical")
    denied = CaptureTools("denied", 0).install(orc(technical))
    with pytest.raises(EvidenceError, match="tool_permission_denied"):
        asyncio.run(technical.get_tools()["read_demo_order"].handler(request(reader=reader), {"order_id": "DEMO-1001"}))
    assert reader.await_count == 0
    denied.restore()


def test_tool_and_rag_budgets_are_checked_before_handler_and_callback():
    calls = []
    async def handler(req, args):
        calls.append(args)
        return {"success": True, "results": []}
    spec = make_tool("search_knowledge_base", "controlled", {"query": {"type": "string"}}, handler, ["query"])
    agent = Agent(tools={spec.name: spec})
    admission = AsyncMock()
    capture = CaptureTools("budget", 0, max_rag_calls=1, before_call=admission).install(orc(agent))
    async def run():
        await agent.get_tools()[spec.name].handler(request(), {"query": "first"})
        with pytest.raises(EvidenceError, match="tool_budget_exhausted"):
            await agent.get_tools()[spec.name].handler(request(), {"query": "second"})
    asyncio.run(run())
    assert len(calls) == admission.await_count == 1 and capture.rag_calls == 2
    capture.restore()


@pytest.mark.parametrize("payload", [{"amount": float("nan")}, {"large": "x" * 1000}, {1: "not-a-json-key"}, {"tuple": (1, 2)}])
def test_invalid_or_oversize_result_is_rejected_not_truncated(payload):
    handler = lambda req, args: payload
    spec = make_tool("inspect_request_context", "controlled", {}, handler)
    agent = Agent(tools={spec.name: spec})
    capture = CaptureTools("size", 0, max_result_bytes=64).install(orc(agent))
    with pytest.raises(EvidenceError, match="evidence_"):
        asyncio.run(agent.get_tools()[spec.name].handler(request(), {}))
    assert capture.calls[0]["result"] is None and capture.calls[0]["result_sha256"] is None
    capture.restore()


def test_cancelled_handler_restores_context_and_keeps_attempt():
    active = ContextVar("cancel-role", default=None)
    @contextmanager
    def context(role):
        token = active.set(role)
        try: yield
        finally: active.reset(token)
    async def reader(customer, order):
        assert active.get() == "general"
        raise asyncio.CancelledError()
    agent = Agent()
    capture = CaptureTools("cancel", 0, call_context=context).install(orc(agent))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.get_tools()["read_demo_order"].handler(request(reader=reader), {"order_id": "DEMO-1001"}))
    assert capture.tool_calls == 1 and capture.calls[0]["error_code"] == "tool_cancelled" and active.get() is None
    capture.restore()


@pytest.mark.parametrize("damage", ["missing", "input", "id", "name", "result_success"])
def test_trace_missing_or_mismatch_blocks_without_guessing(damage):
    capture, execution, _, _ = successful_capture()
    traces = deepcopy(execution.tool_traces)
    if damage == "missing": traces = []
    elif damage == "input": traces[0]["input"] = {"order_id": "DEMO-1001"}
    elif damage == "id": traces[0]["tool_use_id"] = ""
    elif damage == "name": traces[0]["tool_name"] = "compare_amounts"
    else: traces[0].pop("result_success")
    execution.tool_traces = traces
    evidence = capture.reconcile(execution)
    assert "tool_trace_mismatch" in evidence["blockers"]
    assert evidence["calls"][0]["result"]["order"]["refunds"][0]["amount_minor"] is None


def test_business_projection_is_strict_and_not_found_is_a_valid_denial():
    fixture = load_fixture()
    _, execution, evidence, _ = successful_capture()
    checks = deterministic_checks(case(), execution, evidence, fixture)
    assert checks["blockers"] == [] and all(checks["checks"].values())
    _, execution, evidence, value = successful_capture("DEMO-1005")
    assert value["status"] == "not_found" and value["order"] is None
    assert deterministic_checks(case("DEMO-1005"), execution, evidence, fixture)["blockers"] == []


def test_frozen_case_turns_and_prior_tool_evidence_are_supported_without_requery():
    _, execution, evidence, _ = successful_capture()
    modern = {key: value for key, value in case().items() if key != "input"}
    modern["customer_id"] = "demo-customer-a"
    modern["turns"] = ["请查DEMO-1004", "金额知道吗？"]
    checked = deterministic_checks(modern, execution, evidence, load_fixture())
    assert checked["blockers"] == []
    packet = json.loads(build_judge_context(modern, evidence).split("\n", 1)[1])
    assert packet["trusted_customer_id"] == "demo-customer-a"
    assert packet["actual_tools"][0]["turn_id"] == 0


def test_correct_other_order_fixture_cannot_satisfy_the_frozen_requested_order():
    _, execution, evidence, _ = successful_capture("DEMO-1001")
    wanted = case()
    wanted["input"]["message"] = "DEMO-1001是背景，本次只查询DEMO-1004"
    wanted["references"].append({"source": "fixture", "table": "orders", "row_id": "DEMO-1004", "fields": {}})
    checked = deterministic_checks(wanted, execution, evidence, load_fixture())
    assert not checked["checks"]["tool_arguments"]
    assert not checked["checks"]["required_tools"]


def test_wrong_case_binding_and_stale_trace_association_are_not_reused():
    capture, execution, evidence, _ = successful_capture()
    evidence["calls"][0]["case_id"] = "another-case"
    assert not deterministic_checks(case(), execution, evidence, load_fixture())["checks"]["trace_integrity"]
    execution.tool_traces = []
    assert capture.reconcile(execution)["calls"][0]["tool_use_id"] is None


def test_known_fallback_event_is_preserved_as_event_not_fabricated_tool():
    capture, execution, _, _ = successful_capture()
    execution.tool_traces.insert(0, {"event": "agent_fallback", "from_agent": "technical", "to_agent": "general", "success": True})
    evidence = capture.reconcile(execution)
    assert evidence["blockers"] == [] and len(evidence["events"]) == 1
    assert len(evidence["calls"]) == 1


@pytest.mark.parametrize("change", ["null_to_zero", "integer_to_float", "date", "foreign_owner", "identity", "argument"])
def test_business_identity_null_type_date_and_reference_errors_block(change):
    _, execution, evidence, _ = successful_capture()
    call = evidence["calls"][0]
    if change == "null_to_zero": call["result"]["order"]["refunds"][0]["amount_minor"] = 0
    elif change == "integer_to_float": call["result"]["order"]["total_amount_minor"] = 29900.0
    elif change == "date": call["result"]["as_of"] = "2026-11-03T00:00:00Z"
    elif change == "foreign_owner": call["result"] = fixture_order_result(load_fixture(), "demo-customer-b", "DEMO-1005")
    elif change == "identity": call["trusted_customer_id"] = "demo-customer-b"
    else: call["input"]["order_id"] = "DEMO-1001"
    assert deterministic_checks(case(), execution, evidence, load_fixture())["blockers"]


def test_failed_tool_missing_agent_and_literal_leak_cannot_earn_deterministic_pass():
    _, execution, evidence, _ = successful_capture()
    wanted = case(); wanted["expected"]["required_agents"] = ["technical"]
    evidence["calls"][0]["result_success"] = False
    execution.response = "演示客户B专属设备，单号DEMO-TRACK-1005"
    checked = deterministic_checks(wanted, execution, evidence, load_fixture())
    assert not checked["checks"]["required_agents"]
    assert not checked["checks"]["business_evidence"]
    assert not checked["checks"]["required_tools"]
    assert not checked["checks"]["auxiliary_disclosure_check"]


def test_judge_context_keeps_actual_tool_data_separate_from_gold_and_is_bounded():
    _, _, evidence, _ = successful_capture()
    wanted = case(); wanted["input"]["history"] = [{"role": "user", "content": "old"}]
    text = build_judge_context(wanted, evidence, history=[])
    assert "退款额未知" in text and "\\u9000" not in text
    packet = json.loads(text.split("\n", 1)[1])
    assert packet["history"] == []
    assert packet["actual_tools"][0]["result"]["order"]["refunds"][0]["amount_minor"] is None
    assert packet["references"] == wanted["references"] and packet["frozen_required_facts"] == ["退款额未知"]
    assert "input_schema" not in packet["actual_tools"][0]
    with pytest.raises(EvidenceError, match="evidence_size_limit"):
        build_judge_context(wanted, evidence, maximum_bytes=32)


def judge_client(*, text=None, stop="end_turn", error=None):
    create = AsyncMock(side_effect=error) if error else AsyncMock(return_value=SimpleNamespace(
        content=[{"type": "text", "text": text or '{"relevance":0.9,"accuracy":0.8,"completeness":0.7,"helpfulness":0.6}'}], stop_reason=stop))
    return SimpleNamespace(messages=SimpleNamespace(create=create), base_url="https://dashscope.aliyuncs.com/apps/anthropic")


def test_single_attempt_judge_reuses_rubric_and_ignores_environment(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MAX_TOKENS", "broken")
    monkeypatch.setenv("EVAL_JUDGE_TIMEOUT_S", "999")
    client = judge_client()
    judge = SingleAttemptJudge(client, "qwen3.7-flash-2026-07-15")
    scores = asyncio.run(judge.judge("退款？", "金额未知", "核验资料"))
    assert scores.overall == pytest.approx(.75) and not scores.judge_failed
    assert client.messages.create.await_count == scores.attempts == 1
    args = client.messages.create.call_args.kwargs
    assert args["max_tokens"] == 4096 and args["temperature"] == 0.0
    assert args["extra_body"]["thinking"] == {"type": "disabled"}
    assert "核验资料" in args["messages"][0]["content"] and "Few-shot" in args["messages"][0]["content"]


@pytest.mark.parametrize("text,stop", [("broken JSON", "end_turn"), ('{"relevance":1}', "end_turn"),
    ('{"relevance":true,"accuracy":1,"completeness":1,"helpfulness":1}', "end_turn"),
    ('{"relevance":1,"accuracy":1,"completeness":1,"helpfulness":1}', "max_tokens")])
def test_bad_or_truncated_judge_output_has_unknown_scores_and_no_repair(text, stop):
    client = judge_client(text=text, stop=stop)
    scored = asyncio.run(SingleAttemptJudge(client, "controlled").judge("q", "a"))
    assert scored.judge_failed and scored.overall is None and scored.relevance is None
    assert scored.attempts == client.messages.create.await_count == 1


def test_judge_request_failure_is_sanitized_and_cancel_propagates():
    client = judge_client(error=RuntimeError("Authorization: private-secret-key"))
    scores = asyncio.run(SingleAttemptJudge(client, "controlled").judge("q", "a"))
    assert scores.judge_failed and "private-secret" not in repr(scores)
    assert client.messages.create.await_count == 1
    cancelled = judge_client(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(SingleAttemptJudge(cancelled, "controlled").judge("q", "a"))
    assert cancelled.messages.create.await_count == 1


def test_judge_timeout_does_not_retry_or_fabricate_zero_scores():
    async def blocked(**kwargs):
        await asyncio.Future()
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=blocked)))
    scores = asyncio.run(SingleAttemptJudge(client, "controlled", timeout_s=1).judge("q", "a"))
    assert scores.judge_failed and scores.overall is None and scores.accuracy is None
    assert client.messages.create.await_count == scores.attempts == 1


@pytest.mark.parametrize("timeout", [True, 0, 16, float("nan"), float("inf"), "15"])
def test_judge_configuration_rejects_unbounded_deadlines(timeout):
    with pytest.raises(EvidenceError, match="invalid_judge_configuration"):
        SingleAttemptJudge(judge_client(), "controlled", timeout_s=timeout)
