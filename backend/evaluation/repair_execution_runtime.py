"""Adapters for the sixteen frozen repair positions; no sender or Judge.

The caller owns the counted SDK, data, execution clock and durable ledger.  This
module runs the production Orchestrator for tasks and the production RAG chain
on the same index as a vector baseline.  It creates no real SDK, database,
memory manager or remote embedding client and never assigns a semantic score.
"""
from __future__ import annotations

from copy import deepcopy
import time
from types import SimpleNamespace

from agents.agent_orchestrator import Request
from core.order_evidence import (
    extract_order_evidence, merge_order_evidence, render_order_evidence, resolve_order_id,
)
from demo_business.store import load_fixture
from evaluation import benchmark as bench, repair_validation as preparation
from evaluation import intent_trial as base
from evaluation.benchmark_metrics import score_retrieval
from evaluation.expanded_observation_runtime import (
    _ObservedClient, _bounded, _guard, _record, _usage,
)
from evaluation.intent_transport import canonical_payload_hash
from evaluation.retrieval_baseline import _rows as source_rows
from evaluation.task_evidence import CaptureTools, deterministic_checks, fixture_order_result
from evaluation.task_runtime import build_runtime, _result
from evaluation.task_transport import Frame, context
from tooling.tool_manager import MCPToolManager, Tool


SCHEMA = "serveflow-repair-execution-artifact-v1"
DATASET_SHA256 = "1962adb23bb85d054aa0a0129fe0f0621df4e38382419f9725335ef0be0a6707"
TASK_IDS = tuple(f"RT{i:02d}" for i in range(1, 9))
RETRIEVAL_IDS = tuple(f"RR{i:02d}" for i in range(1, 9))
_ACTUAL = ("trace_integrity", "tool_permissions", "trusted_identity", "tool_arguments", "business_evidence")
_GATES = ("artifact_binding", "execution_eligible", "actual_trace_integrity", "actual_tool_permissions",
          "actual_trusted_identity", "actual_tool_arguments", "actual_business_evidence", "required_tools",
          "order_results_valid", "auxiliary_disclosure_check", "tool_task_correct", "expected_status")


def _fail(code):
    raise ValueError("repair_runtime_" + code)


def rows(suite=None):
    """Read immutable question wrappers only; no results or execution config."""
    path = bench.BACKEND_ROOT / preparation.DATASET
    if bench.sha256(path) != DATASET_SHA256:
        _fail("dataset_changed")
    data = bench.read_json(path)
    if (data.get("schema_version") != "serveflow-repair-validation-cases-v1"
            or [v.get("id") for v in data.get("tasks", [])] != list(TASK_IDS)
            or [v.get("id") for v in data.get("retrieval", [])] != list(RETRIEVAL_IDS)):
        _fail("dataset_identity")
    if suite is not None and suite not in {"tasks", "retrieval"}:
        _fail("suite")
    return deepcopy(data[suite] if suite else data["tasks"] + data["retrieval"])


def _source(row, suite, frame_case_id=None):
    if type(row) is not dict or row.get("id") not in (TASK_IDS if suite == "tasks" else RETRIEVAL_IDS):
        _fail("case_identity")
    original = next(v for v in rows(suite) if v["id"] == row["id"])
    if row != original:
        _fail("case_changed")
    if frame_case_id is not None and frame_case_id != original["id"]:
        _fail("frame_identity")
    return original


def _data_kind(data):
    proof = getattr(data, "proof", None)
    kind = proof.get("kind") if type(proof) is dict else None
    if kind is None and type(proof) is dict and proof.get("schema_version") == "serveflow-task-live-data-v1":
        kind = "live-mysql-and-isolated-chroma"
    if kind not in {"controlled-fixture", "live-mysql-and-isolated-chroma"}:
        _fail("data_kind")
    return kind


def _scope(ensure_healthy):
    if not callable(ensure_healthy):
        _fail("health_guard")
    faults, state = [], {"open": True}

    def current():
        if not state["open"]:
            _fail("closed_context")
        ensure_healthy()

    healthy = _guard(current, faults)

    def reject(code):
        error = ValueError("repair_runtime_" + code)
        if not faults:
            faults.append(error)
        raise error

    return healthy, faults, lambda: state.update(open=False), reject


class _TaskClient(_ObservedClient):
    """Use the existing bounded audit clock without limiting a task to one phase.

    The parent transport accounts every physical request. Unlike the separate
    RAG observation adapter, a task may legitimately call several role stages.
    No response, phase, budget or actor text is repaired here.
    """

    def __init__(self, sdk, healthy, faults):
        super().__init__(sdk, None, [], healthy, faults)

    async def create(self, **kwargs):
        await self._audit()
        response = await self._bound.messages.create(**kwargs)
        await self._audit()
        return response


def _invalid():
    return {"checks": dict.fromkeys(_GATES, False), "blockers": ["invalid_repair_task_artifact"],
            "mechanisms": {"fresh_read_each_relevant_turn": None, "exact_projection": None,
                           "required_agents": None, "original_execution_complete": None},
            "semantic_review_required": True}


def checks(row, artifact):
    """Recheck actual native/handler identity and source, not prose quality."""
    try:
        row = _source(row, "tasks")
        case, identity = row["case"], row["id"]
        inp, trusted = bench.model_input(case), bench.fixture_context(case)
        request_id = f"repair-task-{identity}-1"
        turns = artifact.get("turns")
        if type(turns) is not list or len(turns) != 1 or type(turns[0]) is not dict:
            return _invalid()
        turn = turns[0]
        output, captured = turn.get("result"), turn.get("tool_evidence")
        if type(output) is not dict or type(captured) is not dict:
            return _invalid()
        calls = captured.get("calls")
        if type(calls) is not list or len(calls) > 12 or type(captured.get("blockers")) is not list:
            return _invalid()
        bound = bool(artifact.get("schema_version") == SCHEMA and artifact.get("suite") == "tasks"
            and artifact.get("case_id") == artifact.get("logical_case_id") == identity
            and artifact.get("source_case_id") == case["id"] and artifact.get("group") == row["group"]
            and artifact.get("wrapper_sha256") == canonical_payload_hash(row)
            and artifact.get("case_sha256") == canonical_payload_hash(case)
            and artifact.get("data_kind") in {"controlled-fixture", "live-mysql-and-isolated-chroma"}
            and artifact.get("input_sha256") == canonical_payload_hash(inp)
            and artifact.get("trusted_context_sha256") == canonical_payload_hash(trusted)
            and turn.get("turn_id") == "turn-1" and turn.get("question") == inp["message"]
            and turn.get("history") == inp["history"] and turn.get("request_id") == request_id
            and output.get("request_id") == request_id
            and captured.get("case_id") == identity and captured.get("turn_id") == 1
            and artifact.get("answer") == output.get("response")
            and artifact.get("execution_status") == output.get("status"))
        replay = CaptureTools(identity, 1)
        replay.calls = deepcopy(calls)
        joined = replay.reconcile(SimpleNamespace(**output))
        associated = not joined["blockers"] and joined["calls"] == calls
        actual_case = {"id": identity, "input": {**inp, **trusted},
            "expected": {"required_agents": [], "required_tools": case["expected"]["required_tools"]},
            "references": case["references"]}
        observed = deterministic_checks(actual_case, SimpleNamespace(**output), captured, load_fixture())
        d = observed["checks"]
        selection = resolve_order_id(inp["message"], inp["history"])
        records, fresh, valid_results = [], False, True
        for call in calls:
            if type(call) is not dict:
                return _invalid()
            bound &= (call.get("case_id") == identity and call.get("turn_id") == 1
                      and call.get("request_id") == request_id)
            if call.get("name") != "read_demo_order":
                continue
            args = call.get("input")
            if (type(args) is not dict or set(args) != {"order_id"}
                    or selection.status != "resolved" or args["order_id"] != selection.order_id):
                valid_results = False
                continue
            value = extract_order_evidence(call.get("result"), expected_order_id=args["order_id"],
                                           trusted_customer_id=trusted["customer_id"])
            valid_results &= value.status in {"found", "not_found"}
            records.append(value)
            fresh |= bool(call.get("success") and call.get("tool_use_id") and value.status in {"found", "not_found"})
        merged = merge_order_evidence(records) if records else None
        has_order = "read_demo_order" in case["expected"]["required_tools"]
        eligible = bool(output.get("status") in {"success", "partial"}
                        and type(output.get("response")) is str and output["response"].strip())
        status = eligible and (case["expected"]["expected_status"] == "not-prescribed"
                              or output["status"] == case["expected"]["expected_status"])
        actual_safe = bool(associated and all(d[name] for name in _ACTUAL))
        result = {"artifact_binding": bound, "execution_eligible": eligible,
            "actual_trace_integrity": bool(d["trace_integrity"] and associated),
            "actual_tool_permissions": d["tool_permissions"], "actual_trusted_identity": d["trusted_identity"],
            "actual_tool_arguments": d["tool_arguments"], "actual_business_evidence": d["business_evidence"],
            "required_tools": d["required_tools"], "order_results_valid": bool(valid_results),
            "auxiliary_disclosure_check": d["auxiliary_disclosure_check"],
            "tool_task_correct": bool(bound and actual_safe and valid_results and d["required_tools"]),
            "expected_status": bool(status)}
        return {"checks": result,
            "blockers": list(dict.fromkeys([*captured["blockers"], *(k for k, passed in result.items() if not passed)])),
            "mechanisms": {"fresh_read_each_relevant_turn": fresh if has_order else None,
                "exact_projection": bool(merged and merged.status in {"found", "not_found"}
                    and render_order_evidence(merged) in output["response"]) if has_order else None,
                "required_agents": None, "original_execution_complete": d["execution_complete"]},
            "execution_observation": {"actual_status": output.get("status"),
                "expected_status": case["expected"]["expected_status"], "scripted_history_turn_count": len(inp["history"]),
                "provided_history_preserved": turn.get("history") == inp["history"], "semantic_review_required": True},
            "tool_task_applicable": case["expected"]["tool_task"], "semantic_review_required": True}
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
        return _invalid()


async def execute_task(row, sdk, data, *, ensure_healthy, frame_case_id=None):
    """One fresh production Orchestrator turn; no Judge or background memory.

    The caller must enter eval_deadlines(EvaluationDeadlinePolicy()), as for the
    existing counted transport. Input history is supplied test data, not a claim
    of live multi-turn memory. Request/position IDs stay transport metadata.
    """
    row = _source(row, "tasks", frame_case_id)
    kind = _data_kind(data)
    healthy, faults, close, reject = _scope(ensure_healthy)
    healthy()
    case, identity = row["case"], row["id"]
    inp, trusted = bench.model_input(case), bench.fixture_context(case)
    started = time.perf_counter()
    fixture = load_fixture()

    async def order_reader(customer, order_id):
        healthy()
        result = await data.order_reader(customer, order_id)
        healthy()
        wanted = fixture_order_result(fixture, customer, order_id)
        if canonical_payload_hash(result) != canonical_payload_hash(wanted):
            reject("order_source_changed")
        return result

    async def knowledge_handler(params, tool_context):
        healthy()
        result = await data.knowledge_handler(deepcopy(params), tool_context)
        healthy()
        try:
            if type(result) is not list or len(result) > 20:
                raise ValueError
            validated = [source_rows([v])[0] for v in result]
            if len({v["document_id"] for v in validated}) != len(validated):
                raise ValueError
        except Exception:
            reject("policy_source_changed")
        return deepcopy(result)

    capture = None
    try:
        proxy = SimpleNamespace(order_reader=order_reader, knowledge_handler=knowledge_handler)
        orchestrator, _manager = await build_runtime(_TaskClient(sdk, healthy, faults), proxy)
        healthy()
        history = deepcopy(inp["history"])
        request = Request(message=inp["message"], history=history or None,
            user_id="repair-task-" + trusted["customer_id"], conv_id="repair-task-" + identity,
            request_id=f"repair-task-{identity}-1", context=("[本次对话历史]\n" + "\n".join(
                f"{t['role']}: {t['content']}" for t in history[-6:])) if history else "",
            conversation_state={}, demo_customer_id=trusted["customer_id"], demo_order_reader=order_reader)
        selection = resolve_order_id(request.message, request.history)
        capture = CaptureTools(identity, 1, before_call=lambda _metadata: healthy(),
            call_context=lambda role: context(agent=role),
            expected_order_id=selection.order_id if selection.status == "resolved" else None)
        capture.install(orchestrator)

        def admission():
            healthy()
            capture.assert_send_allowed()

        with context(Frame(identity, "turn-1", evidence_guard=admission)):
            actual = await orchestrator.run(request)
            observed = capture.reconcile(actual)
            capture.assert_healthy()
            healthy()
            output = _result(actual)
        artifact = {"schema_version": SCHEMA, "suite": "tasks", "case_id": identity,
            "logical_case_id": identity, "source_case_id": case["id"], "group": row["group"],
            "wrapper_sha256": canonical_payload_hash(row), "case_sha256": canonical_payload_hash(case),
            "data_kind": kind, "input_sha256": canonical_payload_hash(inp),
            "trusted_context_sha256": canonical_payload_hash(trusted),
            "turns": [{"turn_id": "turn-1", "question": inp["message"], "history": inp["history"],
                       "request_id": request.request_id, "result": output, "tool_evidence": observed}],
            "answer": actual.response, "execution_status": actual.status,
            "wall_latency_ms": (time.perf_counter()-started)*1000, "actor_path_latency_ms": actual.latency_ms,
            "judge": None, "semantic_review": None, "quality_pass_claim": False}
        artifact["shared_checks"] = checks(row, artifact)
        artifact["review_status"] = ("pending_review" if all(artifact["shared_checks"]["checks"].values())
                                     and not artifact["shared_checks"]["blockers"] else "failed")
        return _bounded(artifact)
    finally:
        if capture is not None:
            capture.restore()
        close()


async def run_retrieval(row, sdk, data, *, ensure_healthy, frame_case_id=None):
    """Same-index vector Top3 versus the current extract/recall/rerank chain."""
    row = _source(row, "retrieval", frame_case_id)
    kind = _data_kind(data)
    healthy, faults, close, reject = _scope(ensure_healthy)
    healthy()
    case, identity = row["case"], row["id"]
    observations, recalls = [], []
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._model = base.MODEL
    manager._tools, manager._cache, manager._cache_versions = {}, {}, {}
    manager._client = _ObservedClient(sdk, "rag_rewrite", observations, healthy, faults)
    method = {"rewrite_queries": [], "merged_sources": [], "rewrite_recall_top3": [],
              "rerank_input_count": None, "rerank_applied": False}

    async def handler(params, tool_context):
        healthy()
        actual = await data.knowledge_handler(deepcopy(params), tool_context)
        healthy()
        try:
            if type(actual) is not list or len(actual) > 20:
                raise ValueError
            validated = [source_rows([v])[0] for v in actual]
            if len({v["document_id"] for v in validated}) != len(validated):
                raise ValueError
        except Exception:
            reject("policy_source_changed")
        recalls.append({"query": params["query"], "top_k": params.get("top_k"),
                        "titles": [v["title"] for v in validated]})
        return deepcopy(actual)

    manager.register(Tool("knowledge_search", "Frozen local policy collection", handler,
        schema={"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["query"], "additionalProperties": False},
        cache_ttl=300, supports_rerank=True, timeout_s=8))
    original_rewrite, original_rerank = manager.rewrite_query, manager._rerank_with_status

    async def rewrite(*args, **kwargs):
        healthy()
        value = await original_rewrite(*args, **kwargs)
        healthy()
        method["rewrite_queries"] = list(value)
        return value

    async def rerank(query, items, top_k, **kwargs):
        healthy()
        if len(items) > 20:
            reject("source_limit")
        method["merged_sources"] = [source_rows([v])[0] for v in items]
        method["rewrite_recall_top3"] = [v["title"] for v in method["merged_sources"][:3]]
        method["rerank_input_count"] = len(items)
        manager._client = _ObservedClient(sdk, "rag_rerank", observations, healthy, faults)
        value, applied = await original_rerank(query, items, top_k, **kwargs)
        healthy()
        method["rerank_applied"] = applied
        return value, applied

    manager.rewrite_query, manager._rerank_with_status = rewrite, rerank
    question = bench.model_input(case)["message"]
    began = time.perf_counter()
    try:
        with context(Frame(identity, "turn-1", evidence_guard=healthy)):
            baseline_started = time.perf_counter()
            baseline_rows = source_rows(await handler({"query": question, "top_k": 3}, None))
            baseline_ms = (time.perf_counter()-baseline_started)*1000
            healthy()
            enhanced_started = time.perf_counter()
            with base._quiet_clients():
                result = await manager.search_with_rewrite("knowledge_search", question, top_k=3)
            healthy()
            enhanced_rows = source_rows(result.data if type(result.data) is list else [])
            enhanced_ms = (time.perf_counter()-enhanced_started)*1000
        baseline = _record(case["id"], status="success", latency_ms=baseline_ms,
                           titles=[v["title"] for v in baseline_rows])
        enhanced = _record(case["id"], status="success" if result.success else "failed", latency_ms=enhanced_ms,
                           titles=[v["title"] for v in enhanced_rows])
        healthy()
        return _bounded({"schema_version": SCHEMA, "suite": "retrieval", "case_id": identity,
            "logical_case_id": identity, "source_case_id": case["id"], "group": row["group"],
            "wrapper_sha256": canonical_payload_hash(row), "case_sha256": canonical_payload_hash(case),
            "input_sha256": canonical_payload_hash(bench.model_input(case)), "data_kind": kind,
            "status": enhanced["status"], "wall_latency_ms": (time.perf_counter()-began)*1000,
            "baseline": {"record": baseline, "sources": baseline_rows},
            "enhanced": {"record": enhanced, "sources": enhanced_rows},
            "ranking_observations": {"baseline": score_retrieval([case], [baseline])["per_case"][0],
                                     "enhanced": score_retrieval([case], [enhanced])["per_case"][0]},
            "method_observation": {**method, "recalls": recalls}, "phase_observations": observations,
            "usage_observation": _usage(observations), "physical_requests_observed": len(observations),
            "semantic_review": None, "quality_pass_claim": False})
    finally:
        close()
