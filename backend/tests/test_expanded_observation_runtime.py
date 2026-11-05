"""Actual production parsers/pipeline with controlled SDK and source fixtures."""
import asyncio
from copy import deepcopy
import hashlib
import json
import socket
import time
from types import SimpleNamespace

import pytest

from core.model_deadline import (EvaluationDeadlinePolicy, ModelDeadlineExceeded,
                                 assert_model_active, await_model, eval_deadlines)
from evaluation import benchmark as bench, expanded_observation_runtime as runtime
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_transport import CURRENT


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("controlled observation tests cannot contact services")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr("tooling.tool_manager.AsyncAnthropic", denied)


def case(suite, index=0):
    return bench.read_json(bench.DATASET_ROOT / f"expanded-{suite}-v1.json")["cases"][index]


class SDK:
    base_url = "https://dashscope.aliyuncs.com/apps/anthropic"

    def __init__(self, *, intent="order_status", intent_text=None, rewrite=None, rerank=None):
        self.messages = self
        self.calls, self.closed = [], False
        self.intent = intent
        self.intent_text, self.rewrite, self.rerank = intent_text, rewrite, rerank

    async def create(self, **kwargs):
        frame = CURRENT.get()
        assert frame is not None
        self.calls.append((frame, deepcopy(kwargs)))
        if frame.phase == "intent":
            raw = self.intent_text if self.intent_text is not None else json.dumps(
                {"intent": self.intent, "confidence": 0.99, "reasoning": "controlled observation"})
        elif frame.phase == "rag_rewrite":
            raw = self.rewrite if self.rewrite is not None else '["改写A", "改写B", "改写C"]'
        elif frame.phase == "rag_rerank":
            raw = self.rerank if self.rerank is not None else '[5,4,3,2,1,0]'
        else:
            raise AssertionError("Agent/Judge/background model calls are forbidden")
        return SimpleNamespace(content=[{"type": "text", "text": raw}],
                               stop_reason="end_turn", model="qwen3.7-flash-2026-07-15",
                               usage=SimpleNamespace(input_tokens=30, output_tokens=5))

    async def close(self):
        self.closed = True


class Data:
    def __init__(self, *, empty=False, corrupt=False):
        self.proof = {"kind": "controlled-fixture"}
        self.calls, self.closed, self.fault = [], False, None
        self.empty, self.corrupt = empty, corrupt
        self.rows = []
        for doc in bench.read_json(bench.BACKEND_ROOT / "knowledge/serveflow-demo-v2.json"):
            self.rows.append({"title": doc["title"], "content": doc["content"], "score": .75,
                "chunk": 0, "document_id": hashlib.sha256(json.dumps(["title", doc["title"]], ensure_ascii=True).encode()).hexdigest(),
                "version": hashlib.sha256(doc["content"].encode()).hexdigest()})

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def knowledge_handler(self, params, context):
        self.calls.append(deepcopy(params))
        if self.empty:
            return []
        rows = self.rows[1:] if params["query"].startswith("改写") else self.rows
        value = deepcopy(rows[:params["top_k"]])
        if self.corrupt and value:
            value[0]["content"] += " made up"
        return value

    async def close(self):
        self.closed = True


def run_intent(c=None, sdk=None, data=None, healthy=None):
    c, sdk, data = c or case("intents"), sdk or SDK(), data or Data()
    async def execute():
        with eval_deadlines(EvaluationDeadlinePolicy()):
            return await runtime.run_intent(c, sdk, data,
                ensure_healthy=healthy or data.assert_healthy, frame_case_id="I-"+c["id"])
    result = asyncio.run(execute())
    return result, sdk, data


def run_retrieval(c=None, sdk=None, data=None, healthy=None):
    c, sdk, data = c or case("retrieval"), sdk or SDK(), data or Data()
    async def execute():
        with eval_deadlines(EvaluationDeadlinePolicy()):
            return await runtime.run_retrieval(c, sdk, data,
                ensure_healthy=healthy or data.assert_healthy, frame_case_id="R-"+c["id"])
    result = asyncio.run(execute())
    return result, sdk, data


def test_one_intent_observation_is_shared_with_local_fusion_without_other_calls():
    c = case("intents")
    output, sdk, data = run_intent(c)
    assert output["status"] == "success" and output["shared_observation"] is True
    assert output["llm"]["intent"] == "order_status"
    assert output["fusion"]["embedding_source"] == "local-ngram"
    assert output["physical_requests_observed"] == len(sdk.calls) == 1
    frame, payload = sdk.calls[0]
    assert (frame.case_id, frame.turn_id, frame.phase, frame.agent) == ("I-xi001", "turn-1", "intent", None)
    assert payload["model"] == "qwen3.7-flash-2026-07-15"
    assert (payload["max_tokens"], payload["temperature"]) == (256, .1)
    assert output["phase_observations"][0]["sdk_arguments_sha256"] == canonical_payload_hash(payload)
    assert "payload_sha256" not in output["phase_observations"][0]
    assert output["input_sha256"] == canonical_payload_hash(bench.model_input(c))
    assert output["usage_observation"] == {"input_tokens": 30, "output_tokens": 5}
    assert data.calls == [] and not sdk.closed and not data.closed and CURRENT.get() is None


def test_labels_ids_and_review_notes_do_not_enter_the_intent_payload():
    c = case("intents", 3)
    output, sdk, _ = run_intent(c)
    wire = json.dumps(sdk.calls[0][1], ensure_ascii=False)
    assert c["input"]["message"] in wire
    assert c["id"] not in wire and c["expected"]["rationale"] not in wire
    assert c["novelty"] not in wire and "validation" not in wire
    assert output["case_sha256"] == canonical_payload_hash(c)


def test_bad_intent_json_is_preserved_as_failed_not_repaired_or_resent():
    output, sdk, _ = run_intent(sdk=SDK(intent_text="not JSON"))
    assert output["status"] == "failed" and output["llm"]["parser_failed"] is True
    assert output["phase_observations"][0]["response"]["text"] == "not JSON"
    assert len(sdk.calls) == 1 and CURRENT.get() is None


def test_retrieval_uses_original_vector_baseline_and_original_rewrite_rerank():
    c = case("retrieval")
    output, sdk, data = run_retrieval(c)
    assert [frame.phase for frame, _ in sdk.calls] == ["rag_rewrite", "rag_rerank"]
    assert all(frame.case_id == "R-er01" and frame.turn_id == "turn-1" and frame.agent is None
               for frame, _ in sdk.calls)
    assert [(p["max_tokens"], p["temperature"]) for _, p in sdk.calls] == [(512, .3), (2048, .0)]
    assert data.calls[0] == {"query": c["input"]["message"], "top_k": 3}
    assert data.calls[1] == {"query": c["input"]["message"], "top_k": 5}
    assert output["baseline"]["record"]["ranked_titles"] == [r["title"] for r in data.rows[:3]]
    assert output["enhanced"]["record"]["ranked_titles"] == [r["title"] for r in data.rows[::-1][:3]]
    assert output["method_observation"]["rewrite_queries"] == [c["input"]["message"], "改写A", "改写B", "改写C"]
    assert output["method_observation"]["rerank_input_count"] == 6
    assert output["method_observation"]["rerank_applied"] is True
    assert len(output["phase_observations"]) == output["physical_requests_observed"] == 2
    assert output["usage_observation"] == {"input_tokens": 60, "output_tokens": 10}
    assert output["semantic_review"] is None and output["quality_pass_claim"] is False
    assert not sdk.closed and not data.closed and CURRENT.get() is None


def test_retrieval_expectations_are_only_used_after_sdk_not_in_prompts():
    c = case("retrieval")
    _, sdk, _ = run_retrieval(c)
    for _, request in sdk.calls:
        wire = json.dumps(request, ensure_ascii=False)
        assert c["id"] not in wire and c["expected"]["rationale"] not in wire and c["novelty"] not in wire
        assert "relevant_titles" not in wire and "review_status" not in wire


def test_out_of_scope_nonempty_material_remains_a_false_hit():
    c = next(c for c in bench.read_json(bench.DATASET_ROOT / "expanded-retrieval-v1.json")["cases"]
             if c["expected"]["kind"] == "out_of_scope")
    output, _, _ = run_retrieval(c)
    for row in output["ranking_observations"].values():
        assert row["mis_hit"] is True and row["correct_rejection"] is False
    assert output["quality_pass_claim"] is False


def test_empty_recall_does_not_send_extra_rerank_to_fill_a_plan():
    output, sdk, _ = run_retrieval(data=Data(empty=True))
    assert output["status"] == "failed"
    assert len(sdk.calls) == 1 and sdk.calls[0][0].phase == "rag_rewrite"
    assert output["method_observation"]["rerank_applied"] is False
    assert output["enhanced"]["record"]["ranked_titles"] == []


def test_production_rerank_fallback_is_observed_not_repaired_or_retried():
    output, sdk, _ = run_retrieval(sdk=SDK(rerank="not an index array"))
    assert len(sdk.calls) == 2
    assert output["method_observation"]["rerank_applied"] is False
    assert output["enhanced"]["record"]["ranked_titles"] == output["method_observation"]["rewrite_recall_top3"]
    assert output["phase_observations"][1]["response"]["text"] == "not an index array"


def test_corrupt_source_is_rejected_before_new_model_send():
    sdk, data = SDK(), Data(corrupt=True)
    with pytest.raises(ValueError):
        run_retrieval(sdk=sdk, data=data)
    assert sdk.calls == [] and CURRENT.get() is None


def test_source_corruption_after_rewrite_cannot_reach_rerank():
    sdk, data = SDK(), Data()
    original = data.knowledge_handler
    async def search(params, tool_context):
        data.corrupt = bool(data.calls)
        return await original(params, tool_context)
    data.knowledge_handler = search
    with pytest.raises(ValueError, match="source_rejected"):
        run_retrieval(sdk=sdk, data=data)
    assert len(sdk.calls) == 1 and sdk.calls[0][0].phase == "rag_rewrite"
    assert CURRENT.get() is None


def test_transient_health_failure_in_local_tool_stays_failed_after_environment_recovers(monkeypatch):
    sdk, data = SDK(), Data()
    in_tool, failed_once = False, False
    original = runtime.MCPToolManager._run_handler
    async def handler(manager, tool, params, tool_context):
        nonlocal in_tool
        in_tool = True
        try:
            return await original(manager, tool, params, tool_context)
        finally:
            in_tool = False
    monkeypatch.setattr(runtime.MCPToolManager, "_run_handler", handler)
    def healthy():
        nonlocal failed_once
        if in_tool and not failed_once:
            failed_once = True
            raise ValueError("controlled_transient_health_failure")
        # Later checks of the underlying environment would pass. The adapter
        # must retain the original failure despite ToolManager's fallback.
    with pytest.raises(ValueError, match="controlled_transient_health_failure"):
        run_retrieval(sdk=sdk, data=data, healthy=healthy)
    assert failed_once and len(sdk.calls) == 1
    assert sdk.calls[0][0].phase == "rag_rewrite" and CURRENT.get() is None
    assert not sdk.closed and not data.closed


def test_real_sdk_first_wire_matches_existing_offline_captures():
    import httpx
    from anthropic import AsyncAnthropic
    from evaluation import intent_trial, retrieval_trial

    async def measure():
        endpoint = SDK.base_url
        intent_case, retrieval_case = case("intents"), case("retrieval")
        expected_intent = await intent_trial._capture(intent_case, endpoint)
        expected_rewrite = await retrieval_trial.capture_rewrite(retrieval_case, endpoint)
        wires = []
        async def handler(request):
            body = request.content
            wires.append((CURRENT.get().phase, body))
            phase = CURRENT.get().phase
            value = '{"intent":"order_status","confidence":0.99,"reasoning":"controlled"}' if phase == "intent" else '["改写A", "改写B", "改写C"]' if phase == "rag_rewrite" else '[5,4,3,2,1,0]'
            return httpx.Response(200, json={"id": "offline-wire", "type": "message", "role": "assistant",
                "model": "qwen3.7-flash-2026-07-15", "content": [{"type": "text", "text": value}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2}})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)
        sdk = AsyncAnthropic(api_key="offline-capture-only", base_url=endpoint, max_retries=0, http_client=http)
        data = Data()
        try:
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await runtime.run_intent(intent_case, sdk, data, ensure_healthy=data.assert_healthy, frame_case_id="I-"+intent_case["id"])
                await runtime.run_retrieval(retrieval_case, sdk, data, ensure_healthy=data.assert_healthy, frame_case_id="R-"+retrieval_case["id"])
        finally:
            await sdk.close()
        for phase, expected in [("intent", expected_intent), ("rag_rewrite", expected_rewrite)]:
            wire = next(body for label, body in wires if label == phase)
            assert hashlib.sha256(wire).hexdigest() == expected["request_bytes_sha256"]
            assert canonical_payload_hash(json.loads(wire)) == expected["payload_sha256"]
    asyncio.run(measure())


@pytest.mark.parametrize("suite,index,frame", [("intents", 0, "R-xi001"), ("retrieval", 0, "R-er25")])
def test_bad_case_identity_never_sends(suite, index, frame):
    c, sdk, data = case(suite, index), SDK(), Data()
    function = runtime.run_intent if suite == "intents" else runtime.run_retrieval
    with pytest.raises(ValueError, match="case_identity"):
        asyncio.run(function(c, sdk, data, ensure_healthy=data.assert_healthy, frame_case_id=frame))
    assert sdk.calls == [] and data.calls == []


def test_health_failure_during_model_await_prevents_followup_and_restores_context():
    sdk, data = SDK(), Data()
    original = sdk.create
    async def create(**kwargs):
        response = await original(**kwargs)
        data.fault = "controlled_version_changed"
        return response
    sdk.messages = SimpleNamespace(create=create)
    with pytest.raises(ValueError, match="controlled_version_changed"):
        run_retrieval(sdk=sdk, data=data)
    assert len(sdk.calls) == 1 and CURRENT.get() is None and not sdk.closed


def test_cancelled_model_request_is_not_retried_and_caller_resources_remain_owned():
    sdk, data = SDK(), Data()
    async def create(**kwargs):
        sdk.calls.append((CURRENT.get(), deepcopy(kwargs)))
        raise asyncio.CancelledError
    sdk.messages = SimpleNamespace(create=create)
    with pytest.raises(asyncio.CancelledError):
        run_intent(sdk=sdk, data=data)
    assert len(sdk.calls) == 1 and CURRENT.get() is None and not sdk.closed and not data.closed


def test_slow_full_local_audits_do_not_spend_active_model_time_or_pause_network():
    async def measure():
        sdk, observations, faults = SDK(), [], []
        entered = []
        async def create(**kwargs):
            assert_model_active()  # Proves actual SDK await is outside overhead.
            entered.append(True)
            await asyncio.sleep(.004)
            assert_model_active()
            return SimpleNamespace(content=[{"type": "text", "text": "controlled"}],
                stop_reason="end_turn", model="qwen3.7-flash-2026-07-15", usage=None)
        sdk.messages = SimpleNamespace(create=create)
        def audit(): time.sleep(.035)
        client = runtime._ObservedClient(sdk, "intent", observations, audit, faults)
        with eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=.25, cleanup_grace_s=.02)), runtime.context(runtime.Frame("I-xi001", "turn-1")):
            value = await await_model(client.create(model="qwen3.7-flash-2026-07-15", max_tokens=256,
                temperature=.1, messages=[{"role": "user", "content": "controlled"}]), timeout_s=.02)
        assert value.content[0]["text"] == "controlled" and entered == [True]
        assert len(observations) == 1 and faults == [] and not sdk.closed
    asyncio.run(measure())


@pytest.mark.parametrize("suite", ["intents", "retrieval"])
def test_audit_overhead_limit_stops_before_sdk_even_when_production_helper_catches(suite):
    async def measure():
        c, sdk, data = case(suite), SDK(), Data()
        calls = 0
        # Intent: initial guard then first SDK audit. RAG additionally has the
        # baseline pre/post/after and rewrite pre guard before the first audit.
        slow_at = 2 if suite == "intents" else 6
        def healthy():
            nonlocal calls
            calls += 1
            if calls == slow_at:
                time.sleep(.035)
        function = runtime.run_intent if suite == "intents" else runtime.run_retrieval
        with eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=.01, cleanup_grace_s=.02)):
            with pytest.raises(ModelDeadlineExceeded) as error:
                await function(c, sdk, data, ensure_healthy=healthy,
                    frame_case_id=("I-" if suite == "intents" else "R-")+c["id"])
            assert error.value.reason == "overhead"
        assert sdk.calls == [] and not sdk.closed and CURRENT.get() is None
    asyncio.run(measure())


def test_cancel_during_local_audit_never_enters_sdk_or_followup():
    sdk, observations, faults = SDK(), [], []
    async def measure():
        def cancel(): asyncio.current_task().cancel()
        client = runtime._ObservedClient(sdk, "intent", observations, cancel, faults)
        with eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=.2, cleanup_grace_s=.02)), runtime.context(runtime.Frame("I-xi001", "turn-1")):
            await await_model(client.create(model="qwen3.7-flash-2026-07-15", max_tokens=256,
                temperature=.1, messages=[{"role": "user", "content": "controlled"}]), timeout_s=.05)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(measure())
    assert sdk.calls == [] and observations == [] and faults and CURRENT.get() is None


def test_model_wait_remains_active_and_cannot_hide_inside_administrative_scope():
    async def measure():
        sdk, observations, faults = SDK(), [], []
        entered = []
        async def create(**kwargs):
            assert_model_active()
            entered.append(True)
            await asyncio.sleep(.05)
            raise AssertionError("network/model wait should have timed out")
        sdk.messages = SimpleNamespace(create=create)
        client = runtime._ObservedClient(sdk, "intent", observations, lambda: None, faults)
        with eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=.2, cleanup_grace_s=.02)), runtime.context(runtime.Frame("I-xi001", "turn-1")):
            with pytest.raises(ModelDeadlineExceeded) as error:
                await await_model(client.create(model="qwen3.7-flash-2026-07-15", max_tokens=256,
                    temperature=.1, messages=[{"role": "user", "content": "controlled"}]), timeout_s=.01)
            assert error.value.reason == "active"
        assert entered == [True] and len(observations) == 1
    asyncio.run(measure())
