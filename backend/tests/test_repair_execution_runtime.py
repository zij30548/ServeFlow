"""Real production wiring with controlled SDK/source fixtures, never live scores."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import re
import socket
from types import SimpleNamespace

from anthropic import AsyncAnthropic
import httpx
import pytest

from core.diagnostic_evidence import select_diagnostic_request
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from core.order_evidence import resolve_order_id
from demo_business.store import load_fixture
from evaluation import benchmark as bench, repair_execution_runtime as runtime
from evaluation.task_evidence import fixture_order_result
from evaluation.task_transport import CURRENT


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repair adapter tests cannot contact services")
    for key in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, key, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


class Data:
    def __init__(self, *, empty=False, corrupt=False):
        self.proof = {"kind": "controlled-fixture"}
        self.fixture, self.orders, self.searches = load_fixture(), [], []
        self.fault, self.empty, self.corrupt = None, empty, corrupt
        self.closed, self.original = False, None
        self.sources = []
        for doc in bench.read_json(bench.BACKEND_ROOT / "knowledge/serveflow-demo-v2.json"):
            self.sources.append({"title": doc["title"], "content": doc["content"], "chunk": 0, "score": .5,
                "document_id": hashlib.sha256(json.dumps(["title", doc["title"]], ensure_ascii=True).encode()).hexdigest(),
                "version": hashlib.sha256(doc["content"].encode()).hexdigest()})

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def order_reader(self, customer, order_id):
        self.orders.append((customer, order_id))
        return fixture_order_result(self.fixture, customer, order_id)

    async def knowledge_handler(self, params, tool_context):
        self.searches.append(deepcopy(params))
        self.original = self.original or params["query"]
        rows = self.sources if params["query"] == self.original else self.sources[1:]
        value = [] if self.empty else deepcopy(rows[:params["top_k"]])
        if self.corrupt and value:
            value[0]["content"] += " wrong source"
        return value


class SDK:
    base_url = "https://dashscope.aliyuncs.com/apps/anthropic"

    def __init__(self, row, *, refuse=False, ranking=None):
        self.messages, self.row = self, deepcopy(row)
        self.calls, self.refuse, self.ranking = [], refuse, ranking

    def response(self, **payload):
        frame = CURRENT.get()
        assert frame is not None
        self.calls.append((frame, deepcopy(payload)))
        phase = frame.phase
        if phase == "intent":
            intent = "technical" if self.row["id"] == "RT02" else "order_status"
            raw = json.dumps({"intent": intent, "confidence": .99, "reasoning": "controlled"})
        elif phase == "rag_rewrite":
            data = json.loads(payload["messages"][0]["content"].rsplit("\n", 1)[-1])
            raw = json.dumps(data["allowed_units"][:3], ensure_ascii=False)
        elif phase == "rag_rerank":
            indices = re.findall(r"(?m)^([0-9]+)\. ", payload["messages"][0]["content"])
            raw = self.ranking if self.ranking is not None else json.dumps([int(v) for v in indices[::-1]])
        elif phase == "agent":
            assert frame.agent in {"general", "billing", "technical"}
            name = payload.get("tool_choice", {}).get("name")
            if name and not self.refuse:
                message = self.row["case"]["input"]["message"]
                if name == "read_demo_order":
                    args = {"order_id": resolve_order_id(message, self.row["case"]["input"]["history"]).order_id}
                elif name == "lookup_error_code":
                    args = {"error_code": select_diagnostic_request(message).error_code}
                elif name == "build_diagnostic_plan":
                    args = {"environment": "App", "reproduced": False}
                elif name == "search_knowledge_base":
                    args = {"query": message, "top_k": 3}
                else:
                    raise AssertionError("unplanned native tool")
                return {"content": [{"type": "tool_use", "id": f"native_{frame.agent}_{len(self.calls)}",
                    "name": name, "input": args}], "stop_reason": "tool_use"}
            raw = "受控未核实正文，不能当业务证据。"
        elif phase == "composer":
            raw = "受控汇总，语义仍须另评。"
        else:
            raise AssertionError("Judge, remote embedding and background requests are forbidden")
        return {"content": [{"type": "text", "text": raw}], "stop_reason": "end_turn"}

    async def create(self, **kwargs):
        result = self.response(**kwargs)
        return SimpleNamespace(**result, model="qwen3.7-flash-2026-07-15",
            usage=SimpleNamespace(input_tokens=30, output_tokens=3))


def source(identity):
    return next(v for v in runtime.rows() if v["id"] == identity)


async def execute(identity="RT01", *, sdk=None, data=None, healthy=None, row=None):
    row = row or source(identity)
    sdk, data = sdk or SDK(row), data or Data()
    function = runtime.execute_task if identity.startswith("RT") else runtime.run_retrieval
    with eval_deadlines(EvaluationDeadlinePolicy()):
        result = await function(row, sdk, data, ensure_healthy=healthy or data.assert_healthy,
                                frame_case_id=identity)
    return result, sdk, data


@pytest.mark.parametrize("identity", runtime.TASK_IDS + runtime.RETRIEVAL_IDS)
def test_every_frozen_position_uses_production_paths_without_a_judge_or_semantic_auto_pass(identity):
    artifact, sdk, data = asyncio.run(execute(identity))
    assert artifact["logical_case_id"] == artifact["case_id"] == identity
    assert artifact["semantic_review"] is None and artifact["quality_pass_claim"] is False
    assert artifact["data_kind"] == "controlled-fixture"
    assert not data.closed and CURRENT.get() is None
    assert all(f.case_id == identity and f.turn_id == "turn-1" for f, _ in sdk.calls)
    assert not any(f.phase in {"judge", "embedding", "memory"} for f, _ in sdk.calls)
    if identity.startswith("RT"):
        assert artifact["judge"] is None
        assert all(artifact["shared_checks"]["checks"].values()), artifact["shared_checks"]
        assert artifact["review_status"] == "pending_review"
        assert artifact["turns"][0]["history"] == source(identity)["case"]["input"]["history"]
        for call in artifact["turns"][0]["tool_evidence"]["calls"]:
            assert call["tool_use_id"] and call["request_id"] == f"repair-task-{identity}-1"
            assert call["trusted_customer_id"] == "demo-customer-a"
    else:
        assert [f.phase for f, _ in sdk.calls] == ["rag_rewrite", "rag_rerank"]
        assert artifact["physical_requests_observed"] == 2
        assert data.orders == []
        assert artifact["baseline"]["record"]["case_id"] == source(identity)["case"]["id"]


def test_wrong_question_group_gold_or_frame_never_creates_a_runtime_or_spends():
    for edit in (lambda r: r["case"]["expected"]["required_facts"].clear(),
                 lambda r: r.update(group="other"),
                 lambda r: r["case"]["input"].update(customer_id="demo-customer-b")):
        row = source("RT01"); edit(row)
        sdk, data = SDK(row), Data()
        with pytest.raises(ValueError, match="case_changed"):
            asyncio.run(execute(row=row, sdk=sdk, data=data))
        assert sdk.calls == data.orders == []
    row, sdk, data = source("RT01"), SDK(source("RT01")), Data()
    with pytest.raises(ValueError, match="frame_identity"):
        asyncio.run(runtime.execute_task(row, sdk, data, ensure_healthy=data.assert_healthy, frame_case_id="RT02"))
    assert sdk.calls == data.orders == []


@pytest.mark.parametrize("mutation", ["history", "request", "handler_hash", "native_id", "customer", "wrapper"])
def test_recomputed_checks_reject_changed_native_evidence_or_binding(mutation):
    artifact, _, _ = asyncio.run(execute())
    changed = deepcopy(artifact)
    call = changed["turns"][0]["tool_evidence"]["calls"][0]
    if mutation == "history":
        changed["turns"][0]["history"] = [{"role": "assistant", "content": "old"}]
    elif mutation == "request":
        call["request_id"] = "old-request"
    elif mutation == "handler_hash":
        call["result"]["order"]["total_amount_minor"] += 1
        raw = json.dumps(call["result"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        call["result_sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "native_id":
        call["tool_use_id"] = "another-native-result"
    elif mutation == "customer":
        call["trusted_customer_id"] = "demo-customer-b"
    else:
        changed["wrapper_sha256"] = "0" * 64
    checks = runtime.checks(source("RT01"), changed)
    assert not all(checks["checks"].values()) and checks["blockers"]


def test_bad_actor_prose_is_not_counted_as_executed_tool_or_automatic_quality():
    artifact, _, data = asyncio.run(execute(sdk=SDK(source("RT01"), refuse=True)))
    assert artifact["review_status"] == "failed"
    assert not artifact["shared_checks"]["checks"]["required_tools"]
    assert artifact["semantic_review"] is None and data.orders == []


@pytest.mark.parametrize("identity", ["RT01", "RR01"])
def test_closed_frames_cannot_authorize_retained_old_callbacks(identity):
    _, sdk, _ = asyncio.run(execute(identity))
    with pytest.raises(ValueError, match="closed_context"):
        sdk.calls[0][0].evidence_guard()


@pytest.mark.parametrize("identity", ["RT01", "RR01"])
def test_cancelled_sdk_request_is_not_retried_and_frame_closes(identity):
    sdk, data = SDK(source(identity)), Data()
    async def cancel(**kwargs):
        sdk.calls.append((CURRENT.get(), deepcopy(kwargs)))
        raise asyncio.CancelledError
    sdk.messages = SimpleNamespace(create=cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(identity, sdk=sdk, data=data))
    assert len(sdk.calls) == 1 and CURRENT.get() is None
    with pytest.raises(ValueError, match="closed_context"):
        sdk.calls[0][0].evidence_guard()


def test_corrupt_actual_order_stays_failed_even_after_production_fallback():
    data, sdk = Data(), SDK(source("RT01"))
    original = data.order_reader
    async def corrupt(*args):
        value = await original(*args)
        value["order"]["total_amount_minor"] += 1
        return value
    data.order_reader = corrupt
    with pytest.raises(ValueError, match="order_source_changed"):
        asyncio.run(execute(sdk=sdk, data=data))
    assert len(data.orders) == 1
    assert sum(f.phase == "agent" for f, _ in sdk.calls) == 1


def test_rag_bad_source_and_transient_version_failure_cannot_be_swallowed():
    sdk, data = SDK(source("RR01")), Data(corrupt=True)
    with pytest.raises(ValueError, match="policy_source_changed"):
        asyncio.run(execute("RR01", sdk=sdk, data=data))
    assert sdk.calls == []
    data, sdk = Data(), SDK(source("RR01"))
    original, once = data.knowledge_handler, False
    async def change(*args):
        nonlocal once
        result = await original(*args)
        if len(data.searches) > 1 and not once:
            once = True
            data.fault = "controlled_version_failure"
        return result
    data.knowledge_handler = change
    with pytest.raises(ValueError, match="controlled_version_failure"):
        asyncio.run(execute("RR01", sdk=sdk, data=data))
    assert len(sdk.calls) == 1 and sdk.calls[0][0].phase == "rag_rewrite"


def test_healthy_rejection_differs_from_failed_support_decision_and_answerable_empty():
    correct, _, _ = asyncio.run(execute("RR03", sdk=SDK(source("RR03"), ranking="[]")))
    assert correct["enhanced"]["record"]["status"] == "success"
    assert correct["ranking_observations"]["enhanced"]["correct_rejection"] is True
    broken, _, _ = asyncio.run(execute("RR03", sdk=SDK(source("RR03"), ranking="[0,0]")))
    assert broken["enhanced"]["record"]["status"] == "failed"
    assert broken["ranking_observations"]["enhanced"]["correct_rejection"] is False
    missed, _, _ = asyncio.run(execute("RR01", sdk=SDK(source("RR01"), ranking="[]")))
    assert missed["ranking_observations"]["enhanced"]["hit_at_3"] is False
    assert not correct["quality_pass_claim"]


@pytest.mark.parametrize("identity", runtime.TASK_IDS + runtime.RETRIEVAL_IDS)
def test_real_sdk_native_feedback_and_query_capture_match_the_frozen_first_payloads(identity):
    from evaluation import intent_trial, retrieval_trial
    async def measure(identity):
        row, data = source(identity), Data()
        oracle = SDK(row)
        wires = []
        def send(request):
            wires.append((CURRENT.get(), request.content))
            value = oracle.response(**json.loads(request.content))
            return httpx.Response(200, json={"id": "controlled-native-wire", "type": "message", "role": "assistant",
                "model": "qwen3.7-flash-2026-07-15", **value, "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1}})
        expected = await (intent_trial._capture(row["case"], SDK.base_url) if identity.startswith("RT")
                          else retrieval_trial.capture_rewrite(row["case"], SDK.base_url))
        async with httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False) as http:
            sdk = AsyncAnthropic(api_key="controlled-not-real", base_url=SDK.base_url, max_retries=0, http_client=http)
            artifact, _, _ = await execute(identity, sdk=sdk, data=data)
        assert hashlib.sha256(wires[0][1]).hexdigest() == expected["request_bytes_sha256"]
        assert artifact["semantic_review"] is None
        for frame, raw in wires:
            wire = raw.decode()
            assert identity not in wire and "required_facts" not in wire and "relevant_titles" not in wire
            assert frame.phase != "judge"
        return artifact
    artifact = asyncio.run(measure(identity))
    if identity.startswith("RT"):
        assert all(artifact["shared_checks"]["checks"].values()), artifact["shared_checks"]
        calls = artifact["turns"][0]["tool_evidence"]["calls"]
        if source(identity)["case"]["expected"]["required_tools"]:
            assert calls and all(call["tool_use_id"] for call in calls)
    else:
        assert artifact["physical_requests_observed"] == 2
