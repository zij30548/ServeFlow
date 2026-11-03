"""Fresh, observed production Orchestrator instances for a task trial.

This scope is Orchestrator.run, not the HTTP app, Redis persistence, profile
maintenance or live merchant transactions. Actor inputs never include gold.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
import time

from agents.agent_orchestrator import AgentOrchestrator, Request
from agents.tools import build_shared_rag_tools
from core.skill_loader import SkillManager
from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation.task_evidence import CaptureTools, SingleAttemptJudge, build_judge_context, deterministic_checks
from evaluation.task_transport import AGENT_PARAMETERS, BoundClient, Frame, context
from tooling.tool_manager import MCPToolManager, Tool


def _result(value):
    return {"response": value.response, "status": value.status,
        "agent_type": value.agent_type.value, "agent_types": [a.value for a in value.agent_types],
        "primary_agent": value.primary_agent.value if value.primary_agent else None,
        "supporting_agents": [a.value for a in value.supporting_agents],
        "intent": value.intent.value if value.intent else None, "tools_used": value.tools_used,
        "tool_traces": value.tool_traces, "routing_reason": value.routing_reason,
        "routing_confidence": value.routing_confidence, "latency_ms": value.latency_ms,
        "request_id": value.request_id, "error_code": value.error_code}


async def build_runtime(sdk, data):
    skills = SkillManager(str(bench.BACKEND_ROOT / "skills"), max_prompt_chars=5000)
    skills.load()
    if skills.errors:
        raise ValueError("eval_skills_invalid")
    orc = AgentOrchestrator("unused-client-replaced-before-use", base_url=str(sdk.base_url),
                            model=base.MODEL, skill_manager=skills)
    originals = {id(orc._intent_recognizer.client): orc._intent_recognizer.client,
                 id(orc._composer._client): orc._composer._client}
    try:
        orc._intent_recognizer.client = BoundClient(sdk, "intent")
        orc._intent_recognizer._embedding.enabled = False
        orc._intent_recognizer._embedding.provider = "local"
        orc._composer._client = BoundClient(sdk, "composer")
        for agents in orc._pool.values():
            for agent in agents:
                originals[id(agent._client)] = agent._client
                if agent._model != base.MODEL:
                    raise ValueError("eval_model_override_not_allowed")
                if agent.agent_type.value in AGENT_PARAMETERS:
                    if (agent.profile.max_tokens, agent.profile.temperature) != AGENT_PARAMETERS[agent.agent_type.value]:
                        raise ValueError("eval_agent_parameters_changed")
                agent._client = BoundClient(sdk, "agent", agent.agent_type.value)
        manager = MCPToolManager("unused-client-replaced-before-use", base_url=str(sdk.base_url), model=base.MODEL)
        originals[id(manager._client)] = manager._client
        manager._client = BoundClient(sdk)
        original_rewrite, original_rerank = manager.rewrite_query, manager._rerank_with_status
        async def rewrite(*args, **kwargs):
            with context(phase="rag_rewrite"):
                return await original_rewrite(*args, **kwargs)
        async def rerank(*args, **kwargs):
            with context(phase="rag_rerank"):
                return await original_rerank(*args, **kwargs)
        manager.rewrite_query, manager._rerank_with_status = rewrite, rerank
        manager.register(Tool(name="knowledge_search", description="Frozen synthetic policy collection",
            handler=data.knowledge_handler,
            schema={"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["query"], "additionalProperties": False},
            cache_ttl=300, supports_rerank=True, timeout_s=8))
        orc.set_shared_tools(build_shared_rag_tools(manager))
        return orc, manager
    finally:
        # These clients never send; no unmanaged client survives injection.
        await asyncio.gather(*(client.close() for client in originals.values()))


async def execute_case(case, sdk, data, *, ensure_healthy):
    """Run all actual user turns, carrying actual assistant replies and state."""
    from demo_business.store import load_fixture
    started = time.perf_counter()
    orc, manager = await build_runtime(sdk, data)
    history, state, turns, calls, blockers, roles, tools, traces = [], {}, [], [], [], [], [], []
    judge = SingleAttemptJudge(BoundClient(sdk, "judge"), base.MODEL, timeout_s=8)
    for number, question in enumerate(case["turns"], 1):
        ensure_healthy()
        req = Request(message=question, user_id="task-eval-" + case["customer_id"],
            conv_id="task-eval-" + case["id"], request_id=f"task-eval-{case['id']}-{number}",
            context="[本次实际对话历史]\n" + "\n".join(f"{v['role']}: {v['content']}" for v in history[-6:]) if history else "",
            history=history[-6:] or None, conversation_state=state,
            demo_customer_id=case["customer_id"], demo_order_reader=data.order_reader)
        capture = CaptureTools(case["id"], number, call_context=lambda role: context(agent=role))
        frame = Frame(case["id"], f"turn-{number}", evidence_guard=capture.assert_send_allowed)
        capture.install(orc)
        try:
            with context(frame):
                actual = await orc.run(req)
                observed = capture.reconcile(actual)
                capture.assert_healthy()
                ensure_healthy()  # Production fallback must not hide transport/data failures.
                calls.extend(observed["calls"])
                blockers.extend(observed["blockers"])
                roles.extend(actual.agent_types)
                tools.extend(actual.tools_used)
                traces.extend(actual.tool_traces)
                aggregate = {"calls": list(calls), "blockers": list(blockers)}
                final = number == len(case["turns"])
                judge_case = case if final else {**case, "expected": {**case["expected"], "required_facts": [],
                    "rationale": "Intermediate actual turn; evaluate only this turn's request, not the future final question."}, "references": []}
                judge_context = build_judge_context(judge_case, aggregate, history)
                quality = None
                if actual.status != "failed":
                    quality = await judge.judge(question, actual.response, context=judge_context)
                    ensure_healthy()
                turns.append({"turn_id": frame.turn_id, "question": question,
                    "request_id": req.request_id, "result": _result(actual), "tool_evidence": observed,
                    "judge": asdict(quality) if quality else None,
                    "judge_overall": quality.overall if quality else None,
                    "judge_skipped": "actor_failed" if quality is None else None})
                history += [{"role": "user", "content": question}, {"role": "assistant", "content": actual.response}]
                state = req.conversation_state or {}
        finally:
            capture.restore()
    all_success = all(turn["result"]["status"] == "success" for turn in turns)
    combined = SimpleNamespace(status="success" if all_success else "partial",
        response=turns[-1]["result"]["response"], agent_types=list(dict.fromkeys(roles)),
        agent_type=roles[-1] if roles else None,
        tools_used=list(dict.fromkeys(tools)), tool_traces=traces)
    checks = deterministic_checks(case, combined, {"calls": calls, "blockers": blockers}, load_fixture())
    kind = data.proof.get("kind")
    if kind is None and data.proof.get("schema_version") == "serveflow-task-live-data-v1":
        kind = "live-mysql-and-isolated-chroma"
    if kind not in {"controlled-fixture", "live-mysql-and-isolated-chroma"}:
        raise ValueError("invalid_evaluation_data_kind")
    return {"schema_version": "serveflow-task-case-result-v1", "case_id": case["id"],
        "data_kind": kind, "turns": turns, "deterministic": checks,
        "wall_latency_ms": (time.perf_counter() - started) * 1000,
        "actor_path_latency_ms": sum(turn["result"]["latency_ms"] for turn in turns),
        "execution_status": "success" if all_success else "failed",
        "judge_ready": all(turn["judge"] is not None and not turn["judge"]["judge_failed"] for turn in turns),
        "semantic_review": None, "answer": turns[-1]["result"]["response"]}
