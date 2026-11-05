"""Evaluation adapters for new intent/retrieval questions, without a sender.

The caller supplies its counted SDK, immutable data and synchronous health
guard. These functions do not read credentials, construct/close SDK clients,
open databases, authorize sends or write an execution ledger. Labels are used
only after retrieval to calculate deterministic ranking observations.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time
from types import SimpleNamespace

from core.model_deadline import assert_model_active, overhead
from evaluation import benchmark as bench, intent_trial as base
from evaluation.benchmark_metrics import score_retrieval
from evaluation.intent_transport import canonical_payload_hash
from evaluation.retrieval_baseline import _rows as source_rows
from evaluation.task_transport import BoundClient, Frame, context
from tooling.tool_manager import MCPToolManager, Tool


INTENT_IDS = tuple(f"xi{i:03d}" for i in range(1, 61))
RETRIEVAL_IDS = tuple(f"er{i:02d}" for i in range(1, 25))
SCHEMA = "serveflow-expanded-observation-artifact-v1"
ARTIFACT_LIMIT = 2 * 1024 * 1024


def _fail(code):
    raise ValueError("expanded_observation_" + code)


def _identity(case, suite, frame_case_id):
    ids, prefix = (INTENT_IDS, "I-") if suite == "intent" else (RETRIEVAL_IDS, "R-")
    if type(case) is not dict or case.get("id") not in ids or frame_case_id != prefix + case["id"]:
        _fail("case_identity")
    bench._exact(case.get("input"), {"message", "history"}, "expanded_observation_input")
    bench._text(case["input"]["message"])
    history = case["input"]["history"]
    if type(history) is not list or len(history) > 12:
        _fail("history")
    for item in history:
        bench._exact(item, {"role", "content"}, "expanded_observation_history")
        if item["role"] not in ("user", "assistant"):
            _fail("history")
        bench._text(item["content"])


def _bounded(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError):
        _fail("artifact_structure")
    if len(encoded) > ARTIFACT_LIMIT:
        _fail("artifact_limit")
    return value


class _ObservedClient:
    """Observe one existing SDK path without taking ownership of its client."""

    def __init__(self, sdk, phase, observations, ensure_healthy, audit_faults):
        self.base_url = sdk.base_url
        self.messages = SimpleNamespace(create=self.create)
        self._bound = BoundClient(sdk, phase=phase)
        self._phase, self._observations, self._healthy = phase, observations, ensure_healthy
        self._audit_faults = audit_faults

    async def _audit(self):
        try:
            # Only full local source/ledger validation is administrative work.
            # The actual SDK/network await below must remain outside this scope.
            async with overhead("admission"):
                self._healthy()
            assert_model_active()
        except BaseException as error:
            # Existing model helpers catch ordinary exceptions for their parser
            # fallback. Keep administrative failure sticky across that fallback
            # so it cannot authorize the next stage of this evaluation job.
            if not self._audit_faults:
                self._audit_faults.append(error)
            raise

    async def create(self, **kwargs):
        await self._audit()
        if any(row["phase"] == self._phase for row in self._observations):
            _fail("duplicate_phase")
        # kwargs may contain SDK-only extra_body; this is deliberately not the
        # canonical/wire payload hash persisted by the outer HTTP transport.
        row = {"phase": self._phase, "sdk_arguments_sha256": canonical_payload_hash(kwargs),
               "request_model": kwargs.get("model"), "response": None}
        self._observations.append(row)
        response = await self._bound.messages.create(**kwargs)
        await self._audit()
        row["response"] = base._response_metadata(response)
        return response


def _record(case_id, *, status, latency_ms, titles):
    return {"case_id": case_id, "status": status, "latency_ms": latency_ms,
            # The caller joins actual physical records to count use once.
            "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None},
            "ranked_titles": list(titles)}


def _usage(observations):
    return {field: sum(row["response"]["usage"][field] for row in observations)
            if all(row["response"] is not None and row["response"]["usage"][field] is not None
                   for row in observations) else None
            for field in ("input_tokens", "output_tokens")}


def _guard(ensure_healthy, audit_faults):
    def healthy():
        try:
            if audit_faults:
                raise audit_faults[0]
            ensure_healthy()
        except BaseException as error:
            if not audit_faults:
                audit_faults.append(error)
            raise
    return healthy


async def run_intent(case, sdk, data, *, ensure_healthy, frame_case_id):
    """One unchanged LLM observation plus the existing local fusion decision."""
    _identity(case, "intent", frame_case_id)
    ensure_healthy()
    observations, audit_faults = [], []
    ensure_healthy = _guard(ensure_healthy, audit_faults)
    client = _ObservedClient(sdk, "intent", observations, ensure_healthy, audit_faults)
    recognizer = base._recognizer(client)
    started = time.perf_counter()
    with context(Frame(frame_case_id, "turn-1")):
        with base._quiet_clients():
            llm = await recognizer._llm_recognize(**bench.model_input(case))
        ensure_healthy()
        if len(observations) != 1 or observations[0]["response"] is None:
            _fail("intent_observation_missing")
        fusion = await base._local_fusion(recognizer, case, llm)
    ensure_healthy()
    return _bounded({"schema_version": SCHEMA, "suite": "intents", "case_id": case["id"],
        "logical_case_id": frame_case_id, "input_sha256": canonical_payload_hash(bench.model_input(case)),
        "case_sha256": canonical_payload_hash(case), "data_kind": "not-used-for-intent",
        "status": "failed" if llm.get("failed") else "success",
        "llm": {"intent": llm["intent"].value, "confidence": llm.get("confidence", 0.0),
                "parser_failed": bool(llm.get("failed"))}, "fusion": fusion,
        "wall_latency_ms": (time.perf_counter()-started)*1000,
        "shared_observation": True, "phase_observations": observations,
        "usage_observation": _usage(observations), "physical_requests_observed": len(observations),
        "score_scope": "classification only; one shared model observation, no Agent task quality claim"})


async def run_retrieval(case, sdk, data, *, ensure_healthy, frame_case_id):
    """Original-query vector Top3 versus the unchanged complete RAG pipeline."""
    _identity(case, "retrieval", frame_case_id)
    ensure_healthy()
    observations, recalls, source_fault, audit_faults = [], [], [], []
    ensure_healthy = _guard(ensure_healthy, audit_faults)
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._model = base.MODEL
    manager._tools, manager._cache, manager._cache_versions = {}, {}, {}
    manager._client = _ObservedClient(sdk, "rag_rewrite", observations, ensure_healthy, audit_faults)
    method = {"rewrite_queries": [], "merged_sources": [], "rewrite_recall_top3": [],
              "rerank_input_count": None, "rerank_applied": False}

    async def handler(params, tool_context):
        ensure_healthy()
        actual = await data.knowledge_handler(deepcopy(params), tool_context)
        ensure_healthy()
        try:
            if type(actual) is not list or len(actual) > 20:
                _fail("source_limit")
            # Preserve original segment/source identity; no invented rows or truncation.
            validated = [source_rows([row])[0] for row in actual]
            if len({row["document_id"] for row in validated}) != len(validated):
                _fail("duplicate_source")
        except Exception:
            source_fault.append("source_rejected")
            raise
        recalls.append({"query": params["query"], "top_k": params.get("top_k"),
                        "titles": [row["title"] for row in validated]})
        # Production manager receives its original exact source shape.
        return deepcopy(actual)

    manager.register(Tool("knowledge_search", "Frozen local policy collection", handler,
        schema={"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["query"], "additionalProperties": False},
        cache_ttl=300, supports_rerank=True, timeout_s=8))
    original_rewrite, original_rerank = manager.rewrite_query, manager._rerank_with_status

    async def rewrite(*args, **kwargs):
        ensure_healthy()
        value = await original_rewrite(*args, **kwargs)
        ensure_healthy()
        method["rewrite_queries"] = list(value)
        return value

    async def rerank(query, items, top_k):
        ensure_healthy()
        if source_fault:
            _fail("source_rejected")
        if len(items) > 20:
            _fail("merged_limit")
        method["merged_sources"] = [source_rows([item])[0] for item in items]
        method["rewrite_recall_top3"] = [row["title"] for row in method["merged_sources"][:3]]
        method["rerank_input_count"] = len(items)
        manager._client = _ObservedClient(sdk, "rag_rerank", observations, ensure_healthy, audit_faults)
        value, applied = await original_rerank(query, items, top_k)
        ensure_healthy()
        method["rerank_applied"] = applied
        return value, applied

    manager.rewrite_query, manager._rerank_with_status = rewrite, rerank
    question = bench.model_input(case)["message"]
    began = time.perf_counter()
    with context(Frame(frame_case_id, "turn-1")):
        baseline_started = time.perf_counter()
        baseline_rows = source_rows(await handler({"query": question, "top_k": 3}, None))
        baseline_ms = (time.perf_counter()-baseline_started)*1000
        ensure_healthy()
        enhanced_started = time.perf_counter()
        with base._quiet_clients():
            result = await manager.search_with_rewrite("knowledge_search", question, top_k=3)
        ensure_healthy()
        if source_fault:
            _fail("source_rejected")
        enhanced_rows = source_rows(result.data if type(result.data) is list else [])
        enhanced_ms = (time.perf_counter()-enhanced_started)*1000
    baseline = _record(case["id"], status="success", latency_ms=baseline_ms,
                       titles=[row["title"] for row in baseline_rows])
    enhanced = _record(case["id"], status="success" if result.success else "failed", latency_ms=enhanced_ms,
                       titles=[row["title"] for row in enhanced_rows])
    ensure_healthy()
    return _bounded({"schema_version": SCHEMA, "suite": "retrieval", "case_id": case["id"],
        "logical_case_id": frame_case_id, "input_sha256": canonical_payload_hash(bench.model_input(case)),
        "case_sha256": canonical_payload_hash(case), "data_kind": (
            "live-mysql-and-isolated-chroma" if getattr(data, "proof", {}).get("schema_version") == "serveflow-task-live-data-v1"
            else deepcopy(getattr(data, "proof", {})).get("kind", "caller-owned-source-data")),
        "status": enhanced["status"], "wall_latency_ms": (time.perf_counter()-began)*1000,
        "baseline": {"record": baseline, "sources": baseline_rows},
        "enhanced": {"record": enhanced, "sources": enhanced_rows},
        "ranking_observations": {"baseline": score_retrieval([case], [baseline])["per_case"][0],
                                 "enhanced": score_retrieval([case], [enhanced])["per_case"][0]},
        "method_observation": {**method, "recalls": recalls}, "phase_observations": observations,
        "usage_observation": _usage(observations), "physical_requests_observed": len(observations),
        "semantic_review": None, "quality_pass_claim": False,
        "score_scope": "title-level retrieval only; insufficient evidence remains insufficient; nonempty out-of-scope is a false hit"})
