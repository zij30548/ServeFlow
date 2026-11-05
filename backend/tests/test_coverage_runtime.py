"""Production loops with MockTransport and source fixtures; no live scores."""
import asyncio
from copy import deepcopy
import json
import os
import re
import socket
import threading
from types import SimpleNamespace

from anthropic import AsyncAnthropic
import httpx
import pytest

from core.diagnostic_evidence import select_diagnostic_request
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from core.order_evidence import resolve_order_id
from evaluation import coverage_runtime as runtime
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_evidence import fixture_order_result
from evaluation.task_transport import CURRENT


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("coverage runtime tests forbid external services")
    for key in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, key, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


def case(identity):
    snapshot = runtime.source_snapshot()
    return next(row for suite in ("tasks", "retrieval") for row in snapshot["cases"][suite] if row["id"] == identity)


class Data:
    def __init__(self, *, only=None, empty=False, corrupt_order=False, corrupt_source=False):
        self.snapshot = runtime.source_snapshot()
        self.proof = runtime.data_proof(self.snapshot, knowledge={"kind": "controlled-source-rows-no-index"})
        self.source_root = self.snapshot["source_root"]
        self.orders, self.searches, self.fault = [], [], None
        self.empty, self.corrupt_order, self.corrupt_source = empty, corrupt_order, corrupt_source
        self.rows = [{key: row[key] for key in ("document_id", "title", "content", "version", "chunk")} | {"score": .5}
                     for row in self.snapshot["chunks"].values()
                     if only is None or row["explicit_document_id"] in only]

    def assert_healthy(self):
        if self.fault:
            raise ValueError(self.fault)

    async def order_reader(self, customer, order_id):
        self.orders.append((customer, order_id))
        value = fixture_order_result(self.snapshot["fixture"], customer, order_id)
        if self.corrupt_order and value["order"]:
            value["order"]["total_amount_minor"] += 1
        return value

    async def knowledge_handler(self, params, tool_context):
        self.searches.append(deepcopy(params))
        rows = [] if self.empty else deepcopy(self.rows[:params["top_k"]])
        if self.corrupt_source and rows:
            rows[0]["content"] += " corrupted"
        return rows


class Replies:
    """Mock provider response plans, never pass/fail semantics."""

    def __init__(self, row, *, ranking=None, no_tools=False, cancel=False):
        self.row, self.ranking = deepcopy(row), ranking
        self.no_tools, self.cancel = no_tools, cancel
        self.calls, self.native = [], set()

    def __call__(self, request):
        frame, payload = CURRENT.get(), json.loads(request.content)
        assert frame is not None
        self.calls.append((frame, payload))
        if self.cancel:
            raise asyncio.CancelledError()
        phase = frame.phase
        if phase == "intent":
            raw = json.dumps({"intent": "order_status" if self.row["id"].startswith("CT") else "query",
                              "confidence": .99, "reasoning": "controlled fixture"})
        elif phase == "rag_rewrite":
            query = json.loads(payload["messages"][0]["content"].rsplit("\n", 1)[-1])
            raw = json.dumps(query["allowed_units"][:3], ensure_ascii=False)
        elif phase == "rag_rerank":
            indices = re.findall(r"(?m)^([0-9]+)\. ", payload["messages"][0]["content"])
            raw = self.ranking if self.ranking is not None else json.dumps([int(i) for i in indices])
        elif phase == "agent":
            names = {tool["name"] for tool in payload.get("tools", [])}
            name = payload.get("tool_choice", {}).get("name")
            selection = resolve_order_id(self.row["input"]["message"], self.row["input"]["history"])
            if (name is None and not self.no_tools and "read_demo_order" in names
                    and (frame.agent, "read_demo_order") not in self.native and selection.status == "resolved"):
                name = "read_demo_order"  # A chooses its native tool autonomously.
            if name and not self.no_tools:
                if name == "read_demo_order":
                    args = {"order_id": selection.order_id}
                elif name == "lookup_error_code":
                    args = {"error_code": select_diagnostic_request(self.row["input"]["message"]).error_code}
                elif name == "build_diagnostic_plan":
                    args = {"environment": "App", "reproduced": False}
                elif name == "search_knowledge_base":
                    args = {"query": self.row["input"]["message"], "top_k": 3}
                else:
                    raise AssertionError("unplanned native tool")
                self.native.add((frame.agent, name))
                return self.response([{"type": "tool_use", "id": "cx_native_" + str(len(self.calls)),
                                       "name": name, "input": args}], stop="tool_use")
            raw = "受控模型自由正文；未证明语义正确，也不能当真实交易证据。"
        elif phase == "composer":
            raw = "受控汇总文字；尚须逐项语义复核。"
        else:
            raise AssertionError("Judge, remote embedding and memory requests are forbidden")
        return self.response([{"type": "text", "text": raw}])

    @staticmethod
    def response(content, *, stop="end_turn"):
        return httpx.Response(200, json={"id": "msg_controlled", "type": "message", "role": "assistant",
            "model": "qwen3.7-flash-2026-07-15", "content": content, "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 30, "output_tokens": 4}})


async def execute(identity="CT-D01", arm="D", *, data=None, replies=None, healthy=None):
    row, data = case(identity), data or Data()
    replies = replies or Replies(row)
    sdk = AsyncAnthropic(api_key="controlled-test-key", base_url="https://coverage.invalid",
        max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(replies)))
    try:
        with eval_deadlines(EvaluationDeadlinePolicy()):
            if identity.startswith("CT"):
                artifact = await runtime.execute_task(row, arm, sdk, data,
                    ensure_healthy=healthy or data.assert_healthy, frame_case_id=arm + "-" + identity)
            else:
                artifact = await runtime.run_retrieval(row, sdk, data,
                    ensure_healthy=healthy or data.assert_healthy, frame_case_id="R-" + identity)
        return artifact, replies, data
    finally:
        await sdk.close()


TASK_IDS = tuple(f"CT-{split}{i:02d}" for split, count in (("D", 12), ("V", 28)) for i in range(1, count + 1))
RETRIEVAL_IDS = tuple(f"CR-{split}{i:02d}" for split, count in (("D", 16), ("V", 32)) for i in range(1, count + 1))


@pytest.mark.parametrize("identity", TASK_IDS)
@pytest.mark.parametrize("arm", ["A", "D"])
def test_native_production_task_paths_keep_arms_sources_and_gold_boundaries(identity, arm):
    artifact, replies, data = asyncio.run(execute(identity, arm))
    assert artifact["case_id"] == arm + "-" + identity
    assert artifact["logical_case_id"] == identity and artifact["data_kind"] == runtime.DATA_KIND
    assert artifact["judge"] is artifact["semantic_review"] is None
    assert artifact["quality_pass_claim"] is False
    assert artifact["physical_requests_observed"] == len(replies.calls)
    assert CURRENT.get() is None
    assert artifact["method_observation"]["recalls"] == [] or all("sources" in r for r in artifact["method_observation"]["recalls"])
    for frame, payload in replies.calls:
        assert frame.case_id == arm + "-" + identity
        assert frame.phase in {"intent", "agent", "composer", "rag_rewrite", "rag_rerank"}
        raw = json.dumps(payload, ensure_ascii=False)
        assert "required_fact_ids" not in raw and "relevant_document_ids" not in raw
        assert case(identity)["expected"]["rationale"] not in raw
    for call in artifact["turns"][0]["tool_evidence"]["calls"]:
        assert call["case_id"] == identity and call["turn_id"] == 1
        assert call["request_id"] == "coverage-task-" + identity + "-1"
        assert call["tool_use_id"] and call["result_sha256"]
    if identity == "CT-D01":
        assert all(artifact["tool_checks"]["checks"].values()), artifact["tool_checks"]
        assert data.orders == [("demo-customer-a", "DEMO-2001")]
        if arm == "D":
            assert "754" in artifact["answer"] and "700" in artifact["answer"]
        else:
            assert "受控模型自由正文" in artifact["answer"]


def test_source_case_frame_arm_and_data_tampering_fail_before_sdk_or_handler():
    async def run():
        data, row = Data(), case("CT-D01")
        forbidden_sdk = SimpleNamespace(base_url="https://coverage.invalid", messages=SimpleNamespace(create=None))
        for arm, frame, changed in [("B", "B-CT-D01", row), ("D", "D-CT-D02", row),
                ("D", "D-CT-D01", {**row, "expected": {}})]:
            with pytest.raises(ValueError):
                await runtime.execute_task(changed, arm, forbidden_sdk, data,
                    ensure_healthy=data.assert_healthy, frame_case_id=frame)
        data.proof["fixture_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="data_contract"):
            await runtime.execute_task(row, "D", forbidden_sdk, data,
                ensure_healthy=data.assert_healthy, frame_case_id="D-CT-D01")
        assert data.orders == data.searches == []
    asyncio.run(run())


@pytest.mark.parametrize("edit", ["result", "request", "native", "history", "customer", "case_hash"])
def test_task_checks_recompute_native_association_and_fixture_evidence(edit):
    artifact, _, _ = asyncio.run(execute())
    changed = deepcopy(artifact)
    call = changed["turns"][0]["tool_evidence"]["calls"][0]
    if edit == "result":
        call["result"]["order"]["total_amount_minor"] += 1
        raw = json.dumps(call["result"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        import hashlib
        call["result_sha256"] = hashlib.sha256(raw).hexdigest()
    elif edit == "request":
        changed["turns"][0]["result"]["request_id"] = "other-request"
    elif edit == "native":
        call["tool_use_id"] = "other-native-id"
    elif edit == "history":
        changed["turns"][0]["history"] = [{"role": "assistant", "content": "not the source history"}]
    elif edit == "customer":
        call["trusted_customer_id"] = "demo-customer-b"
    else:
        changed["case_sha256"] = "0" * 64
    assert runtime.task_checks(case("CT-D01"), "D", changed)["blockers"]


@pytest.mark.parametrize("argument", ["corrupt_order", "corrupt_source"])
def test_source_corruption_is_sticky_across_production_fallback(argument):
    identity = "CT-D01" if argument == "corrupt_order" else "CR-V15"
    with pytest.raises((ValueError, RuntimeError), match="source_changed|tool_handler_failed"):
        asyncio.run(execute(identity, data=Data(**{argument: True})))
    assert CURRENT.get() is None


def test_same_index_full_chunks_and_raw_rerank_are_observed_without_title_shortcut():
    artifact, replies, data = asyncio.run(execute("CR-V15", data=Data(only={"CX-V15"})))
    assert artifact["baseline"]["rows"] == artifact["enhanced"]["rows"]
    assert len(artifact["enhanced"]["rows"]) == 2
    assert [f.phase for f, _ in replies.calls] == ["rag_rewrite", "rag_rerank"]
    assert artifact["method_observation"]["rewrites"][0]["original_query"] == case("CR-V15")["input"]["message"]
    assert artifact["method_observation"]["rewrites"][0]["queries"][0] == data.searches[0]["query"]
    assert artifact["method_observation"]["reranks"][0]["merged_sources"] == artifact["baseline"]["rows"]
    assert artifact["phase_observations"][1]["response"]["text"] == "[0, 1]"
    assert artifact["ranking_observations"]["enhanced"]["all_required_facts_at_3"]
    # Same document, first chunk alone: document hit is true but tail exception
    # fact coverage is incomplete; no semantic pass follows from the title.
    partial = runtime.preparation.retrieval_metrics(case("CR-V15"), artifact["enhanced"]["rows"][:1], "success")
    assert partial["document_hit_at_3"] and not partial["all_required_facts_at_3"]


@pytest.mark.parametrize("identity", RETRIEVAL_IDS)
def test_every_planned_retrieval_keeps_the_original_query_and_actual_two_phase_observations(identity):
    artifact, replies, data = asyncio.run(execute(identity))
    assert artifact["case_id"] == "R-" + identity
    assert artifact["input_sha256"] == canonical_payload_hash(runtime.preparation.model_input(case(identity)))
    assert artifact["method_observation"]["rewrites"][0]["queries"][0] == case(identity)["input"]["message"]
    assert data.searches[0] == {"query": case(identity)["input"]["message"], "top_k": 3}
    assert [frame.phase for frame, _ in replies.calls] == ["rag_rewrite", "rag_rerank"]
    assert artifact["quality_pass_claim"] is False and artifact["semantic_review"] is None


@pytest.mark.parametrize("identity,raw,status,rejection", [
    ("CR-V25", "[]", "success", True),
    ("CR-V25", "[0]", "failed", False),
    ("CR-V15", "[]", "success", False),
])
def test_healthy_empty_failure_empty_and_answerable_miss_remain_distinct(identity, raw, status, rejection):
    replies = Replies(case(identity), ranking=raw)
    artifact, _, _ = asyncio.run(execute(identity, data=Data(only={"CX-V15"}), replies=replies))
    assert artifact["enhanced"]["status"] == status
    assert artifact["enhanced"]["rows"] == []
    assert artifact["ranking_observations"]["enhanced"]["healthy_empty_rejection"] is rejection
    assert artifact["semantic_review"] is None and artifact["quality_pass_claim"] is False


def test_cancelled_sdk_is_not_retried_and_frame_is_closed():
    replies = Replies(case("CT-D01"), cancel=True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(replies=replies))
    assert len(replies.calls) == 1 and CURRENT.get() is None


@pytest.mark.parametrize("malformed", ["json", "shape"])
def test_known_http200_sdk_parse_failure_fails_its_position_and_next_position_still_runs(malformed):
    class FirstMalformed(Replies):
        def __call__(self, request):
            if not self.calls:
                self.calls.append((CURRENT.get(), json.loads(request.content)))
                if malformed == "json":
                    return httpx.Response(200, content=b"{", headers={"content-type": "application/json"})
                return httpx.Response(200, json={"id": "missing-required-message-fields"})
            return super().__call__(request)

    async def run():
        data, replies = Data(), FirstMalformed(case("CT-D01"))
        http = httpx.AsyncClient(transport=httpx.MockTransport(replies))
        sdk = AsyncAnthropic(api_key="controlled-only", base_url="https://coverage.invalid", max_retries=0,
            _strict_response_validation=True, http_client=http)
        try:
            artifacts = []
            for identity in ("CT-D01", "CT-D02"):
                replies.row = case(identity)
                with eval_deadlines(EvaluationDeadlinePolicy()):
                    artifacts.append(await runtime.execute_task(replies.row, "D", sdk, data,
                        ensure_healthy=data.assert_healthy, frame_case_id="D-" + identity))
                assert http.event_hooks["response"] == []
            failed, following = artifacts
            failures = [r["structure_failure"] for r in failed["phase_observations"] if "structure_failure" in r]
            assert failures == [{"error_type": "JSONDecodeError" if malformed == "json" else "APIResponseValidationError", "http_status": 200}]
            assert failed["review_status"] == "failed"
            assert not failed["tool_checks"]["checks"]["model_response_structure_valid"]
            assert following["tool_checks"]["checks"]["model_response_structure_valid"]
            assert following["review_status"] == "pending_review"
            assert sum(frame.phase == "intent" and frame.case_id == "D-CT-D01" for frame, _ in replies.calls) == 1
            assert sum(frame.phase == "intent" and frame.case_id == "D-CT-D02" for frame, _ in replies.calls) == 1
            assert CURRENT.get() is None
        finally:
            await sdk.close()
    asyncio.run(run())


def test_http200_bad_json_does_not_hide_a_simultaneous_administrative_fault():
    data = Data()
    class FaultedReply(Replies):
        def __call__(self, request):
            self.calls.append((CURRENT.get(), json.loads(request.content)))
            data.fault = "source_admin_changed"
            return httpx.Response(200, content=b"{", headers={"content-type": "application/json"})
    replies = FaultedReply(case("CT-D01"))
    with pytest.raises(ValueError, match="source_admin_changed"):
        asyncio.run(execute(data=data, replies=replies))
    assert len(replies.calls) == 1 and CURRENT.get() is None


def test_parse_error_without_http200_evidence_remains_sticky_and_is_not_retried():
    class Non200Reply(Replies):
        def __call__(self, request):
            self.calls.append((CURRENT.get(), json.loads(request.content)))
            return httpx.Response(201, content=b"{", headers={"content-type": "application/json"})
    replies = Non200Reply(case("CT-D01"))
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(execute(replies=replies))
    assert len(replies.calls) == 1 and CURRENT.get() is None


def test_local_data_uses_exact_new_fixture_and_only_owned_collection_cleanup():
    snapshot = runtime.source_snapshot()
    closed = []
    knowledge = SimpleNamespace(proof={"kind": "controlled-index"}, close=lambda: closed.append("owned-only"))
    data = runtime.CoverageData(snapshot, knowledge)
    result = asyncio.run(data.order_reader("demo-customer-a", "DEMO-2116"))
    assert result["order"]["total_amount_minor"] == 140600 and data.proof["database_connections"] == 0
    assert asyncio.run(data.order_reader("demo-customer-a", "DEMO-2112"))["status"] == "not_found"
    asyncio.run(data.close()); asyncio.run(data.close())
    assert closed == ["owned-only"]
    with pytest.raises(ValueError, match="data_closed"):
        data.assert_healthy()


def test_missing_embedding_cache_stops_before_chroma_creation(monkeypatch):
    def unavailable():
        raise runtime.task_data.TaskDataError("evaluation_embedding_cache_unavailable")
    monkeypatch.setattr(runtime.task_data, "model_cache_proof", unavailable)
    with pytest.raises(runtime.task_data.TaskDataError, match="cache_unavailable"):
        asyncio.run(runtime.build_data())


def test_explicit_index_factory_uses_ephemeral_cached_embedding_and_exact_chunks(monkeypatch):
    import chromadb

    class Collection:
        def __init__(self):
            self.rows = {}

        def get(self, where=None, include=None):
            rows = [(key, content, meta) for key, (content, meta) in self.rows.items()
                    if where is None or all(meta.get(k) == v for k, v in where.items())]
            return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                    "metadatas": [r[2] for r in rows]}

        def upsert(self, *, ids, documents, metadatas):
            self.rows.update({key: (content, meta) for key, content, meta in zip(ids, documents, metadatas)})

        def delete(self, ids):
            for key in ids:
                self.rows.pop(key)

        def count(self):
            return len(self.rows)

    class Client:
        def __init__(self):
            self.collections, self.deleted = {"unrelated": object()}, []

        def create_collection(self, *, name, embedding_function, metadata):
            assert name.startswith("serveflow_coverage_eval_") and embedding_function is sentinel
            self.collections[name] = Collection()
            return self.collections[name]

        def delete_collection(self, name):
            self.deleted.append(name)
            self.collections.pop(name)

    client, sentinel, embeddings = Client(), object(), []
    cache = {"model": "controlled-existing-cache", "downloads": "forbidden"}
    monkeypatch.setattr(runtime.task_data, "model_cache_proof", lambda: cache)
    monkeypatch.setattr(runtime.task_data, "_cached_only_embedding",
                        lambda base, proof: embeddings.append(proof) or sentinel)
    monkeypatch.setattr(chromadb, "EphemeralClient", lambda **kwargs: client)
    monkeypatch.setattr(chromadb, "HttpClient", lambda **kwargs: pytest.fail("HTTP Chroma forbidden"))
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: pytest.fail("persistent Chroma forbidden"))
    data = asyncio.run(runtime.build_data())
    assert embeddings == [cache]
    assert data.proof["knowledge"]["chunk_count"] == 28
    assert data.proof["knowledge"]["downloads"] == "forbidden"
    actual = data._knowledge.kb._collection.get()
    assert len(actual["ids"]) == 28
    assert set(meta["chunk_index"] for meta in actual["metadatas"]) == {0, 1}
    asyncio.run(data.close())
    assert list(client.collections) == ["unrelated"] and len(client.deleted) == 1


def test_retained_closed_frame_cannot_send_again():
    artifact, replies, _ = asyncio.run(execute())
    frame = replies.calls[0][0]
    with pytest.raises(ValueError, match="closed_context"):
        frame.evidence_guard()
    assert len(replies.calls) == artifact["physical_requests_observed"]


def test_cancelled_local_factory_cleans_resource_that_finishes_in_its_thread(monkeypatch):
    started, release, closed = threading.Event(), threading.Event(), []
    def build(_snapshot):
        started.set()
        assert release.wait(2)
        return SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(runtime, "_build_knowledge", build)
    async def run():
        task = asyncio.create_task(runtime.build_data())
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel(); release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert closed == [True]
