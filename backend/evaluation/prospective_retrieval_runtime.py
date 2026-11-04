"""Evaluation-only paired execution of the unchanged production RAG methods.

This module creates no ledger, authorization, network transport, index or model
client for sending. Its caller owns the isolated knowledge collection and supplies
the protocol-bound, counted SDK facades. An ordinary production parser fallback
is preserved; an unhealthy transport/accounting boundary is re-raised before a
fallback can be mistaken for successful execution. Labels never enter prompts.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import time

from evaluation import intent_trial as base
from evaluation.benchmark_metrics import score_retrieval
from evaluation.retrieval_baseline import _rows as _source_rows
from tooling.tool_manager import MCPToolManager, Tool


CASE_IDS = tuple(f"q{number:02d}" for number in range(1, 13))
STAGES = ("rewrite", "rerank")
TOP_K = 3
RECALL_K = 5
MODEL_REQUEST_UPPER_BOUND = len(CASE_IDS) * len(STAGES)


def _sources(value, limit):
    """Validate full source snapshots without changing production algorithm input."""
    if type(value) is not list or len(value) > limit:
        raise ValueError("prospective_retrieval_sources_invalid")
    rows = [_source_rows([row])[0] for row in value]
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("prospective_retrieval_sources_duplicated")
    return rows


def _input(case):
    if (type(case) is not dict or case.get("id") not in CASE_IDS
            or type(case.get("input")) is not dict
            or set(case["input"]) != {"message", "history"}
            or type(case["input"]["message"]) is not str
            or not 1 <= len(case["input"]["message"]) <= 2000
            or case["input"]["history"] != []):
        raise ValueError("prospective_retrieval_case_invalid")
    # Do not copy expected/references/related_prior_cases into any actor input.
    return case["id"], case["input"]["message"]


def _record(case_id, status, started, sources, *, model_arm):
    # Only the caller's physical ledger can supply B's actual sends or usage.
    usage = {key: None if model_arm else 0
             for key in ("input_tokens", "output_tokens", "physical_requests")}
    return {"case_id": case_id, "status": status,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "usage": usage, "ranked_titles": [row["title"] for row in sources]}


async def execute_retrieval_case(case, *, client_factory, knowledge, ensure_healthy,
                                 arm_order=("A", "B"), model=base.MODEL,
                                 on_observation=None):
    """Run one new question once per arm, in the caller's frozen arm order.

    ``client_factory(case_id, stage)`` is synchronous and returns a counted SDK
    facade such as BoundClient(sdk, phase="rag_" + stage). The caller installs
    the immutable Frame and eval_deadlines scope, verifies all frozen inputs and
    owns authorization, persistence, clients and collection cleanup. The runtime
    never closes injected clients/knowledge and never classifies a result live.

    A uses one local Top3 search. B invokes the actual production
    search_with_rewrite method, including original-query order, per-query Top5,
    content deduplication, the 200-character rerank window and existing fallback.
    Ordinary method failure returns a failed record; cancellation, source drift,
    client/accounting failures or ensure_healthy failures raise. The callback can
    durably seal partial observations before the enclosing trial pauses.
    """
    case_id, query = _input(case)
    if (model != base.MODEL or type(arm_order) not in (list, tuple)
            or tuple(arm_order) not in (("A", "B"), ("B", "A"))
            or not callable(client_factory) or not callable(ensure_healthy)
            or on_observation is not None and not callable(on_observation)):
        raise ValueError("prospective_retrieval_configuration_invalid")
    handler = getattr(getattr(knowledge, "kb", None), "search_handler", None)
    if not callable(handler):
        raise ValueError("prospective_retrieval_knowledge_invalid")

    arms, events, fatal = {}, [], []
    method = {"rewrite_queries": [], "recalls": [], "merged_sources": [],
              "rewrite_recall_top3": [], "rerank_applied": False,
              "rerank_input_count": None, "stage_wall_ms": {}}

    def healthy():
        ensure_healthy()
        if fatal:
            raise ValueError(fatal[0])

    async def observe(kind, **fields):
        value = {"event": kind, "case_id": case_id, **deepcopy(fields)}
        events.append(value)
        if on_observation is not None:
            pending = on_observation(deepcopy(value))
            if inspect.isawaitable(pending):
                await pending

    async def read(params, context):
        healthy()
        try:
            rows = await handler(params, context)
            validated = _sources(rows, params["top_k"])
        except asyncio.CancelledError:
            raise
        except Exception:
            # ToolManager.call has a normal fallback for handler failures.
            # Actual source/adapter drift must additionally stop this trial.
            fatal.append("prospective_retrieval_local_source_failure")
            raise ValueError(fatal[0]) from None
        healthy()
        observed = {"query_index": method["rewrite_queries"].index(params["query"]),
                    "query": params["query"], "top_k": params["top_k"],
                    "sources": validated}
        method["recalls"].append(deepcopy(observed))
        await observe("recall_completed", arm="B", **observed)
        return rows

    async def vector_arm():
        began = time.perf_counter()
        healthy()
        rows = await asyncio.wait_for(handler({"query": query, "top_k": TOP_K}, None), 8)
        healthy()
        sources = _sources(rows, TOP_K)
        result = {"record": _record(case_id, "success", began, sources, model_arm=False),
                  "sources": sources, "local_queries": [{"query": query, "top_k": TOP_K}]}
        return result

    async def enhanced_arm():
        began = time.perf_counter()
        healthy()
        # Construction has no sends. Close its unused SDK before injecting the
        # caller's counted SDK facade; never read .env or use this dummy client.
        manager = MCPToolManager("unused-replaced-before-use",
                                 base_url="https://dashscope.aliyuncs.com/apps/anthropic", model=model)
        await manager._client.close()
        manager._client = None
        manager.register(Tool("knowledge_search", "Frozen local policy collection", read,
            schema={"type": "object", "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["query"], "additionalProperties": False},
            cache_ttl=300, supports_rerank=True, timeout_s=8))
        production_rewrite = manager.rewrite_query
        production_rerank = manager._rerank_with_status

        def client(stage):
            healthy()
            sdk = client_factory(case_id, stage)
            if inspect.isawaitable(sdk):
                if inspect.iscoroutine(sdk):
                    sdk.close()
                raise ValueError("prospective_retrieval_sync_client_factory_required")
            if (not callable(getattr(getattr(sdk, "messages", None), "create", None))
                    or not getattr(sdk, "base_url", None)):
                raise ValueError("prospective_retrieval_counted_client_required")
            return sdk

        async def rewrite(*args, **kwargs):
            manager._client = client("rewrite")
            started = time.perf_counter()
            value = await production_rewrite(*args, **kwargs)
            healthy()  # Production fallback cannot hide a failed send.
            method["stage_wall_ms"]["rewrite"] = (time.perf_counter() - started) * 1000
            method["rewrite_queries"] = deepcopy(value)
            await observe("rewrite_completed", arm="B", queries=value)
            return value

        async def rerank(question, items, top_k):
            healthy()
            method["rerank_input_count"] = len(items)
            method["merged_sources"] = _sources(items, 20)
            method["rewrite_recall_top3"] = [row["title"] for row in method["merged_sources"][:TOP_K]]
            await observe("rerank_candidates", arm="B", sources=method["merged_sources"],
                          same_candidate_pool_top3=method["rewrite_recall_top3"])
            manager._client = client("rerank")
            started = time.perf_counter()
            value, applied = await production_rerank(question, items, top_k)
            healthy()
            method["stage_wall_ms"]["rerank"] = (time.perf_counter() - started) * 1000
            method["rerank_applied"] = applied
            await observe("rerank_completed", arm="B", applied=applied,
                          sources=_sources(value, TOP_K))
            return value, applied

        manager.rewrite_query, manager._rerank_with_status = rewrite, rerank
        with base._quiet_clients():
            result = await manager.search_with_rewrite("knowledge_search", query, top_k=TOP_K)
        healthy()
        method["recalls"].sort(key=lambda row: row["query_index"])
        sources = _sources(result.data, TOP_K)
        return {"record": _record(case_id, "success" if result.success else "failed",
                                  began, sources, model_arm=True), "sources": sources,
                "local_queries": [{"query": row["query"], "top_k": row["top_k"]}
                                  for row in method["recalls"]]}

    for arm in arm_order:
        healthy()
        await observe("arm_started", arm=arm)
        arms[arm] = await (vector_arm() if arm == "A" else enhanced_arm())
        healthy()
        await observe("arm_completed", arm=arm, result=arms[arm])
    return {"schema_version": "serveflow-prospective-retrieval-observation-v1",
            "case_id": case_id, "measurement_kind": "runtime-observation-not-live-classified",
            "arm_order": list(arm_order), "arms": arms,
            "method_observation": method, "events": events,
            "limitations": ["same-pool rerank observation is not component ablation",
                            "retrieval hits do not establish answer correctness",
                            "physical usage and live/controlled classification require caller ledger"]}


def summarize_retrieval(cases, samples):
    """Reuse shared metric definitions; absent observations stay unexecuted."""
    if type(samples) is not list:
        raise ValueError("prospective_retrieval_samples_invalid")
    seen, records = set(), {"A": [], "B": []}
    expected = {case["id"] for case in cases}
    for sample in samples:
        case_id = sample.get("case_id")
        if case_id not in expected or case_id in seen:
            raise ValueError("prospective_retrieval_sample_identity_invalid")
        seen.add(case_id)
        for arm in ("A", "B"):
            if arm in sample.get("arms", {}):
                record = sample["arms"][arm]["record"]
                if record.get("case_id") != case_id:
                    raise ValueError("prospective_retrieval_sample_identity_invalid")
                records[arm].append(record)
    return {arm: score_retrieval(cases, records[arm]) for arm in ("A", "B")}
