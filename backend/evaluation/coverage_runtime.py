"""Observed production execution on the new synthetic coverage sources.

Importing this module creates no SDK, index, database or authorization. The
caller owns the counted SDK and execution deadlines. Orders are controlled
fixture projections, never MySQL observations; the explicit data factory uses
only an owned ephemeral Chroma collection and the byte-checked cached MiniLM.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from contextvars import ContextVar
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

from agents.agent_orchestrator import Request
from core.order_evidence import extract_order_evidence, resolve_order_id
from evaluation import benchmark as bench, coverage_expansion as preparation
from evaluation import intent_trial as base, quality_arm_runtime, task_data
from evaluation.expanded_observation_runtime import _bounded, _guard, _usage
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_evidence import CaptureTools, deterministic_checks, fixture_order_result
from evaluation.task_runtime import _result
from evaluation.task_transport import BoundClient, CURRENT, Frame, context
from tooling.tool_manager import MCPToolManager, Tool


SCHEMA = "serveflow-coverage-execution-artifact-v1"
DATA_KIND = "controlled-orders-and-isolated-chroma"
ARMS = frozenset({"A", "D"})
_AUDIT_REQUEST = ContextVar("serveflow_coverage_sdk_response_observation", default=None)


def _fail(code):
    raise ValueError("coverage_runtime_" + code)


def source_snapshot(*, source_root=bench.BACKEND_ROOT):
    """Read and strictly validate new public sources; no index or client."""
    root = Path(source_root)
    corpus, records, cases, _ = preparation.load_validated(root)
    return {"source_root": root, "documents": deepcopy(corpus["documents"]),
        "fixture": deepcopy(records["fixture"]), "cases": deepcopy(cases),
        "chunks": preparation.source_chunks(corpus["documents"]),
        "source_files": {path: bench.sha256(root / path) for path in preparation.FILES}}


def data_proof(snapshot, *, knowledge=None):
    """The same source declaration is used by real-index and test adapters."""
    return {"schema_version": "serveflow-coverage-data-proof-v1", "kind": DATA_KIND,
        "orders": "controlled-order-fixture; fixture_order_result, no database connection",
        "fixture_sha256": canonical_payload_hash(snapshot["fixture"]),
        "source_files": deepcopy(snapshot["source_files"]),
        "document_count": len(snapshot["documents"]), "chunk_count": len(snapshot["chunks"]),
        "knowledge": deepcopy(knowledge), "database_connections": 0,
        "remote_embedding_requests": 0}


def _rows(rows, snapshot):
    """Bind every full chunk, including repeated titles at different indices."""
    return preparation.validate_source_rows(rows, root=snapshot["source_root"])


def _source(case, suite, snapshot):
    if type(case) is not dict or type(case.get("id")) is not str:
        _fail("case_identity")
    original = next((row for row in snapshot["cases"][suite] if row["id"] == case["id"]), None)
    if original is None or original != case:
        _fail("case_changed")
    return deepcopy(original)


def _data_contract(data, snapshot):
    proof = getattr(data, "proof", None)
    expected = data_proof(snapshot)
    if (type(proof) is not dict or any(proof.get(key) != value for key, value in expected.items() if key != "knowledge")
            or not callable(getattr(data, "assert_healthy", None))):
        _fail("data_contract")
    data.assert_healthy()


class CoverageData:
    """One owned local index plus immutable synthetic order projections."""

    def __init__(self, snapshot, knowledge):
        self.source_root = snapshot["source_root"]
        self._snapshot, self._knowledge = snapshot, knowledge
        self.proof = data_proof(snapshot, knowledge=knowledge.proof)
        self._proof_hash = canonical_payload_hash(self.proof)
        self._closed, self._fault = False, None

    def assert_healthy(self):
        if self._fault is not None:
            raise self._fault
        try:
            if self._closed:
                _fail("data_closed")
            if canonical_payload_hash(self.proof) != self._proof_hash:
                _fail("data_proof_changed")
            if canonical_payload_hash(self._snapshot["fixture"]) != self.proof["fixture_sha256"]:
                _fail("fixture_changed")
            if any(bench.sha256(self.source_root / path) != digest
                   for path, digest in self.proof["source_files"].items()):
                _fail("sources_changed")
        except BaseException as error:
            self._fault = error
            raise

    async def order_reader(self, customer_id, order_id):
        try:
            self.assert_healthy()
            value = fixture_order_result(self._snapshot["fixture"], customer_id, order_id)
            self.assert_healthy()
            return value
        except BaseException as error:
            self._fault = error
            raise

    async def knowledge_handler(self, params, tool_context):
        try:
            self.assert_healthy()
            if (type(params) is not dict or not set(params) <= {"query", "top_k"}
                    or type(params.get("query")) is not str or not 1 <= len(params["query"]) <= 2000
                    or type(params.get("top_k", 5)) is not int or not 1 <= params.get("top_k", 5) <= 20):
                _fail("knowledge_arguments")
            rows = await task_data._owned(self._knowledge.kb.search_async(
                params["query"], top_k=params.get("top_k", 5)))
            self.assert_healthy()
            return _rows(rows, self._snapshot)
        except BaseException as error:
            self._fault = error
            raise

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await task_data._owned(asyncio.to_thread(self._knowledge.close))


def _build_knowledge(snapshot):
    """Explicit local-only construction. Never invoke KnowledgeBase.__init__."""
    client = None
    name = "serveflow_coverage_eval_" + uuid.uuid4().hex
    try:
        if metadata.version("chromadb") != task_data.CHROMA_VERSION:
            _fail("chroma_version")
        cache = task_data.model_cache_proof()
        import chromadb
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        from tooling.knowledge_base import KnowledgeBase
        embedding = task_data._cached_only_embedding(ONNXMiniLM_L6_V2, cache)
        client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
        collection = client.create_collection(name=name, embedding_function=embedding,
            metadata={"description": "New ServeFlow synthetic coverage corpus, isolated evaluation only"})
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._write_lock = threading.RLock()
        kb._client, kb._collection, kb._use_server = client, collection, False
        documents = [{key: doc[key] for key in ("document_id", "title", "content")}
                     for doc in snapshot["documents"]]
        if kb.add_documents(documents) != len(snapshot["chunks"]) or kb.doc_count != len(snapshot["chunks"]):
            _fail("index_count")
        indexed = collection.get(include=["documents", "metadatas"])
        rows = [{"document_id": meta["document_id"], "title": meta["title"], "content": content,
                 "version": meta["version"], "chunk": meta["chunk_index"], "score": 0.0}
                for content, meta in zip(indexed["documents"], indexed["metadatas"])]
        # Validate in bounded batches; an index is larger than one search result.
        for offset in range(0, len(rows), 20):
            _rows(rows[offset:offset + 20], snapshot)
        if (len(rows) != len(snapshot["chunks"])
                or {(r["document_id"], r["chunk"], r["version"]) for r in rows} != set(snapshot["chunks"])):
            _fail("index_binding")
        proof = {"kind": "isolated-ephemeral-chroma", "chromadb_version": task_data.CHROMA_VERSION,
            "embedding": cache, "embedding_implementation_sha256": hashlib.sha256(
                Path(inspect.getfile(ONNXMiniLM_L6_V2)).read_bytes()).hexdigest(),
            "document_count": len(documents), "chunk_count": len(rows),
            "storage": "ephemeral-memory-only; close deletes only this owned collection",
            "downloads": "forbidden", "remote_embedding_requests": 0}
        return task_data._KnowledgeRuntime(kb, client, name, proof)
    except BaseException:
        if client is not None:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        raise


async def build_data(*, source_root=bench.BACKEND_ROOT):
    """Caller-authorized local inference only; SDK and database requests are 0."""
    snapshot = source_snapshot(source_root=source_root)
    worker = asyncio.create_task(asyncio.to_thread(_build_knowledge, snapshot))
    try:
        knowledge = await asyncio.shield(worker)
    except asyncio.CancelledError:
        # A thread cannot be cancelled; retain ownership until its resource can
        # be cleaned instead of abandoning an eventual successful collection.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled() and worker.exception() is None:
            worker.result().close()
        raise
    data = CoverageData(snapshot, knowledge)
    try:
        data.assert_healthy()
        return data
    except BaseException:
        await data.close()
        raise


def _scope(data, ensure_healthy):
    if not callable(ensure_healthy):
        _fail("health_guard")
    faults, state = [], {"open": True}

    def current():
        if not state["open"]:
            _fail("closed_context")
        data.assert_healthy()
        ensure_healthy()

    healthy = _guard(current, faults)

    def reject(code):
        error = ValueError("coverage_runtime_" + code)
        if not faults:
            faults.append(error)
        raise error

    return healthy, faults, lambda: state.update(open=False), reject


class _AuditClient:
    """Observe every existing phase, retaining only whitelisted response data."""

    def __init__(self, sdk, healthy, observations, faults):
        self.base_url, self.messages = sdk.base_url, SimpleNamespace(create=self.create)
        self._bound = BoundClient(sdk)
        self._healthy, self._observations, self._faults = healthy, observations, faults

    async def create(self, **kwargs):
        from core.model_deadline import assert_model_active, overhead
        row = None
        try:
            async with overhead("admission"):
                self._healthy()
            assert_model_active()
            frame = CURRENT.get()
            if frame is None or frame.phase not in {"intent", "agent", "composer", "rag_rewrite", "rag_rerank"}:
                _fail("phase")
            # Native follow-up messages contain SDK Pydantic content blocks.
            # Normalize only this observation hash; pass the original objects
            # unchanged to the SDK. Canonical/wire send hashes belong to the
            # outer counted transport after the SDK's own serialization.
            row = {"phase": frame.phase, "agent": frame.agent,
                   "sdk_arguments_sha256": canonical_payload_hash(_argument_json(kwargs)),
                   "request_model": kwargs.get("model"), "response": None}
            self._observations.append(row)
            response = await self._request(**kwargs)
            async with overhead("admission"):
                self._healthy()
            row["response"] = base._response_metadata(response)
            return response
        except BaseException as error:
            if row is not None and _ordinary_structure_error(error):
                row["structure_failure"] = {"error_type": type(error).__name__, "http_status": 200}
            elif not self._faults:
                self._faults.append(error)
            raise

    async def _request(self, **kwargs):
        """Observe status without changing SDK arguments, wire bytes or retry.

        JSONDecodeError carries no response, so a request-local httpx response
        hook establishes the HTTP status before the SDK parses it. Hooks retain
        no headers/body and are removed on every exit. Concurrent role calls
        share a client but an opaque ContextVar identifies only this call.
        """
        from anthropic import APIResponseValidationError
        from core.model_deadline import assert_model_active, overhead
        status, identity = {"http_status": None}, object()
        token = _AUDIT_REQUEST.set(identity)
        http = getattr(self._bound._sdk, "_client", None)
        event_hooks = getattr(http, "event_hooks", None)
        hooks = event_hooks.get("response") if type(event_hooks) is dict else None

        async def response_status(response):
            if _AUDIT_REQUEST.get() is identity:
                status["http_status"] = response.status_code

        if type(hooks) is list:
            hooks.append(response_status)
        try:
            return await self._bound.messages.create(**kwargs)
        except (json.JSONDecodeError, APIResponseValidationError) as error:
            response = getattr(error, "response", None)
            code = getattr(response, "status_code", status["http_status"])
            if code == 200:
                # Parent transport has already sealed the received response.
                # Administrative faults still take priority over a known
                # response-shape failure; unknown errors never gain this path.
                async with overhead("admission"):
                    self._healthy()
                assert_model_active()
                error._coverage_structure_http_status = 200
            raise
        finally:
            if type(hooks) is list:
                hooks.remove(response_status)
            _AUDIT_REQUEST.reset(token)


def _ordinary_structure_error(error):
    from anthropic import APIResponseValidationError
    return (isinstance(error, (json.JSONDecodeError, APIResponseValidationError))
            and getattr(error, "_coverage_structure_http_status", None) == 200)


def _argument_json(value, depth=0):
    from pydantic import BaseModel
    if depth > 30:
        _fail("sdk_arguments_depth")
    if isinstance(value, BaseModel):
        return _argument_json(value.model_dump(mode="json", exclude_none=True), depth + 1)
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _argument_json(item, depth + 1) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_argument_json(item, depth + 1) for item in value]
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    _fail("sdk_arguments_type")


async def build_runtime(arm, sdk, data):
    if type(arm) is not str or arm not in ARMS:
        _fail("arm")
    return await quality_arm_runtime.build_arm_runtime(arm, sdk, data)


def _observe_manager(manager, snapshot, healthy, method):
    rewrite, rerank = manager.rewrite_query, manager._rerank_with_status

    async def observed_rewrite(query, *args, **kwargs):
        healthy()
        value = await rewrite(query, *args, **kwargs)
        healthy()
        method["rewrites"].append({"original_query": query, "queries": list(value)})
        return value

    async def observed_rerank(query, items, top_k, **kwargs):
        healthy()
        event = {"original_query": query, "merged_sources": _rows(items, snapshot),
                 "top_k": top_k, "require_support": kwargs.get("require_support", False)}
        method["reranks"].append(event)
        result, applied = await rerank(query, items, top_k, **kwargs)
        healthy()
        event.update({"sources": _rows(result, snapshot), "applied": applied})
        return result, applied

    manager.rewrite_query, manager._rerank_with_status = observed_rewrite, observed_rerank


def task_checks(case, arm, artifact, *, source_root=bench.BACKEND_ROOT):
    """Recompute actual tool/source checks; never infer semantic correctness."""
    snapshot = source_snapshot(source_root=source_root)
    case = _source(case, "tasks", snapshot)
    if arm not in ARMS:
        _fail("arm")
    inp, trusted = preparation.model_input(case), preparation.trusted_input(case)
    identity, expected_frame = case["id"], arm + "-" + case["id"]
    turn = artifact["turns"][0]
    output, captured = turn["result"], deepcopy(turn["tool_evidence"])
    replay = CaptureTools(identity, 1)
    replay.calls = deepcopy(captured["calls"])
    associated = replay.reconcile(SimpleNamespace(**output))
    d = deterministic_checks(case, SimpleNamespace(**output), captured, snapshot["fixture"])["checks"]
    selection = resolve_order_id(inp["message"], inp["history"])
    order_valid, policy_valid, call_binding = True, True, True
    for call in captured["calls"]:
        call_binding &= bool(call.get("case_id") == identity and call.get("turn_id") == 1
            and call.get("request_id") == f"coverage-task-{identity}-1"
            and call.get("trusted_customer_id") == trusted["customer_id"])
        if call["name"] == "read_demo_order":
            args = call.get("input")
            order_valid &= bool(type(args) is dict and selection.status == "resolved"
                and args == {"order_id": selection.order_id})
            if order_valid:
                order_valid &= (canonical_payload_hash(call["result"]) == canonical_payload_hash(
                    fixture_order_result(snapshot["fixture"], trusted["customer_id"], args["order_id"])))
                order_valid &= extract_order_evidence(call["result"], expected_order_id=args["order_id"],
                    trusted_customer_id=trusted["customer_id"]).status in {"found", "not_found"}
        elif call["name"] == "search_knowledge_base":
            value = call.get("result")
            if type(value) is not dict or value.get("success") is not True or value.get("query") != inp["message"].strip():
                policy_valid = False
            else:
                try:
                    _rows(value.get("results"), snapshot)
                except (ValueError, TypeError, KeyError):
                    policy_valid = False
    eligible = output["status"] in {"success", "partial"} and bool(output["response"].strip())
    values = {"artifact_binding": bool(artifact["schema_version"] == SCHEMA
            and artifact["case_id"] == expected_frame and artifact["logical_case_id"] == identity
            and artifact["arm"] == arm and artifact["case_sha256"] == canonical_payload_hash(case)
            and artifact["input_sha256"] == canonical_payload_hash(inp)
            and artifact["trusted_context_sha256"] == canonical_payload_hash(trusted)
            and artifact["data_kind"] == DATA_KIND and artifact["data_source_files"] == snapshot["source_files"]
            and len(artifact["turns"]) == 1 and turn["question"] == inp["message"]
            and turn["history"] == inp["history"] and turn["turn_id"] == "turn-1"
            and turn["request_id"] == output["request_id"] == f"coverage-task-{identity}-1"
            and artifact["answer"] == output["response"] and artifact["execution_status"] == output["status"]),
        "execution_eligible": bool(eligible),
        "actual_trace_integrity": bool(d["trace_integrity"] and not associated["blockers"]
                                       and associated["calls"] == captured["calls"]),
        "actual_tool_permissions": d["tool_permissions"], "actual_trusted_identity": d["trusted_identity"],
        "actual_tool_arguments": d["tool_arguments"], "actual_business_evidence": d["business_evidence"],
        "required_tools": d["required_tools"], "order_results_valid": bool(order_valid),
        "policy_results_valid": bool(policy_valid), "trusted_request_binding": bool(call_binding),
        "model_response_structure_valid": all(row.get("response") is not None
            and "structure_failure" not in row for row in artifact["phase_observations"]),
        "auxiliary_disclosure_check": d["auxiliary_disclosure_check"],
        "expected_status": bool(eligible and (case["expected"]["expected_status"] == "not-prescribed"
                                             or output["status"] == case["expected"]["expected_status"]))}
    return {"checks": values, "blockers": list(dict.fromkeys([*captured["blockers"],
        *(name for name, value in values.items() if not value)])),
        "mechanisms": {"configured_fresh_read": quality_arm_runtime.policy_for(arm).fresh_read,
                       "configured_typed_projection": quality_arm_runtime.policy_for(arm).typed_projection,
                       "native_order_read_observed": any(call.get("name") == "read_demo_order"
                           and call.get("success") and call.get("tool_use_id") for call in captured["calls"])},
        "semantic_review_required": True, "quality_pass_claim": False}


async def execute_task(case, arm, sdk, data, *, ensure_healthy, frame_case_id):
    snapshot = source_snapshot(source_root=getattr(data, "source_root", bench.BACKEND_ROOT))
    case = _source(case, "tasks", snapshot)
    if type(arm) is not str or arm not in ARMS or frame_case_id != arm + "-" + case["id"]:
        _fail("frame_identity")
    _data_contract(data, snapshot)
    healthy, faults, close, reject = _scope(data, ensure_healthy)
    observations, method = [], {"rewrites": [], "reranks": [], "recalls": []}
    inp, trusted = preparation.model_input(case), preparation.trusted_input(case)
    started, capture = time.perf_counter(), None

    async def reader(customer, order_id):
        healthy()
        try:
            value = await data.order_reader(customer, order_id)
        except asyncio.CancelledError as error:
            if not faults:
                faults.append(error)
            raise
        except Exception:
            reject("order_source_unavailable")
        healthy()
        if canonical_payload_hash(value) != canonical_payload_hash(fixture_order_result(snapshot["fixture"], customer, order_id)):
            reject("order_source_changed")
        return value

    async def search(params, tool_context):
        healthy()
        try:
            value = await data.knowledge_handler(deepcopy(params), tool_context)
        except asyncio.CancelledError as error:
            if not faults:
                faults.append(error)
            raise
        except Exception:
            reject("policy_source_unavailable")
        healthy()
        try:
            rows = _rows(value, snapshot)
        except (ValueError, TypeError, KeyError):
            reject("policy_source_changed")
        method["recalls"].append({"query": params["query"], "top_k": params.get("top_k"), "sources": rows})
        return rows

    try:
        healthy()
        proxy = SimpleNamespace(order_reader=reader, knowledge_handler=search)
        orchestrator, manager = await build_runtime(arm, _AuditClient(sdk, healthy, observations, faults), proxy)
        _observe_manager(manager, snapshot, healthy, method)
        history = deepcopy(inp["history"])
        request = Request(message=inp["message"], history=history or None,
            user_id="coverage-task-" + trusted["customer_id"], conv_id="coverage-task-" + case["id"],
            request_id=f"coverage-task-{case['id']}-1", conversation_state={},
            context=("[本次对话历史]\n" + "\n".join(f"{v['role']}: {v['content']}" for v in history[-6:])) if history else "",
            demo_customer_id=trusted["customer_id"], demo_order_reader=reader)
        selection = resolve_order_id(request.message, request.history)
        capture = CaptureTools(case["id"], 1, before_call=lambda _meta: healthy(),
            call_context=lambda role: context(agent=role),
            expected_order_id=selection.order_id if selection.status == "resolved" else None)
        capture.install(orchestrator)

        def admission():
            healthy()
            capture.assert_send_allowed()

        with context(Frame(frame_case_id, "turn-1", evidence_guard=admission)):
            actual = await orchestrator.run(request)
            observed = capture.reconcile(actual)
            capture.assert_healthy()
            healthy()
            output = _result(actual)
        artifact = {"schema_version": SCHEMA, "suite": "tasks", "case_id": frame_case_id,
            "logical_case_id": case["id"], "arm": arm, "case_sha256": canonical_payload_hash(case),
            "input_sha256": canonical_payload_hash(inp), "trusted_context_sha256": canonical_payload_hash(trusted),
            "data_kind": DATA_KIND, "data_source_files": snapshot["source_files"],
            "order_data_mode": "controlled-order-fixture; no MySQL connection",
            "execution_scope": "single-turn production Orchestrator; supplied scripted history; no HTTP or Redis",
            "arm_comparison": "A autonomous/free order text versus D fresh-read/typed protection on shared current safety foundations; not original EchoMind",
            "turns": [{"turn_id": "turn-1", "question": inp["message"], "history": inp["history"],
                       "request_id": request.request_id, "result": output, "tool_evidence": observed}],
            "answer": actual.response, "execution_status": actual.status,
            "wall_latency_ms": (time.perf_counter()-started)*1000, "actor_path_latency_ms": actual.latency_ms,
            "method_observation": method, "phase_observations": observations,
            "physical_requests_observed": len(observations), "usage_observation": _usage(observations),
            "judge": None, "semantic_review": None, "quality_pass_claim": False}
        artifact["tool_checks"] = task_checks(case, arm, artifact, source_root=snapshot["source_root"])
        artifact["review_status"] = "pending_review" if not artifact["tool_checks"]["blockers"] else "failed"
        return _bounded(artifact)
    finally:
        if capture is not None:
            capture.restore()
        close()


async def run_retrieval(case, sdk, data, *, ensure_healthy, frame_case_id):
    snapshot = source_snapshot(source_root=getattr(data, "source_root", bench.BACKEND_ROOT))
    case = _source(case, "retrieval", snapshot)
    if frame_case_id != "R-" + case["id"]:
        _fail("frame_identity")
    _data_contract(data, snapshot)
    healthy, faults, close, reject = _scope(data, ensure_healthy)
    observations, method = [], {"rewrites": [], "reranks": [], "recalls": []}
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._model, manager._tools, manager._cache, manager._cache_versions = base.MODEL, {}, {}, {}
    manager._client = BoundClient(_AuditClient(sdk, healthy, observations, faults), "rag_rewrite")

    async def handler(params, tool_context):
        healthy()
        try:
            value = await data.knowledge_handler(deepcopy(params), tool_context)
        except asyncio.CancelledError as error:
            if not faults:
                faults.append(error)
            raise
        except Exception:
            reject("policy_source_unavailable")
        healthy()
        try:
            rows = _rows(value, snapshot)
        except (ValueError, TypeError, KeyError):
            reject("policy_source_changed")
        method["recalls"].append({"query": params["query"], "top_k": params.get("top_k"), "sources": rows})
        return rows

    manager.register(Tool("knowledge_search", "New frozen synthetic local policy collection", handler,
        schema={"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"],
            "additionalProperties": False}, cache_ttl=300, supports_rerank=True, timeout_s=8))
    original_rerank = manager._rerank_with_status

    async def phase_rerank(*args, **kwargs):
        with context(phase="rag_rerank"):
            # A fixed rewrite facade must not overwrite the rerank phase.
            manager._client = BoundClient(_AuditClient(sdk, healthy, observations, faults), "rag_rerank")
            return await original_rerank(*args, **kwargs)

    manager._rerank_with_status = phase_rerank
    _observe_manager(manager, snapshot, healthy, method)
    question, started = preparation.model_input(case)["message"], time.perf_counter()
    try:
        with context(Frame(frame_case_id, "turn-1", evidence_guard=healthy)):
            began = time.perf_counter()
            baseline = {"status": "success", "rows": await handler({"query": question, "top_k": 3}, None),
                        "latency_ms": (time.perf_counter()-began)*1000}
            began = time.perf_counter()
            result = await manager.search_with_rewrite("knowledge_search", question, top_k=3,
                                                     context={"original_query": question})
            healthy()
            returned_rows = _rows(result.data, snapshot)
            valid_structure = not any("structure_failure" in row for row in observations)
            enhanced = {"status": "success" if result.success and valid_structure else "failed",
                        "rows": returned_rows if valid_structure else [],
                        "latency_ms": (time.perf_counter()-began)*1000,
                        "reranked": result.reranked, "error": result.error}
            if not valid_structure:
                enhanced.update(returned_rows=returned_rows, error="model_response_structure_failed")
        metrics = {name: preparation.retrieval_metrics(case, part["rows"], part["status"],
                    root=snapshot["source_root"]) for name, part in (("baseline", baseline), ("enhanced", enhanced))}
        return _bounded({"schema_version": SCHEMA, "suite": "retrieval", "case_id": frame_case_id,
            "logical_case_id": case["id"], "arm": "same-index-vector-and-enhanced",
            "case_sha256": canonical_payload_hash(case), "input_sha256": canonical_payload_hash(preparation.model_input(case)),
            "data_kind": DATA_KIND, "data_source_files": snapshot["source_files"],
            "execution_scope": "same local index, original-query vector Top3 versus current production extract/recall/rerank; no generated answer",
            "baseline": baseline, "enhanced": enhanced, "ranking_observations": metrics,
            "method_observation": method, "phase_observations": observations,
            "physical_requests_observed": len(observations), "usage_observation": _usage(observations),
            "wall_latency_ms": (time.perf_counter()-started)*1000, "status": enhanced["status"],
            "semantic_review": None, "quality_pass_claim": False})
    finally:
        close()
