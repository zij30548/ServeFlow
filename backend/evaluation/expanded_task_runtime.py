"""Source-bound A/D task adapter for the expanded question set.

The caller owns the SDK, durable allowance, transport, data and deadlines. This
module performs one real Orchestrator turn, captures actual native tool evidence
and checks its association. It never sends to a Judge or grades prose semantics.
Scripted input history remains history, not proof of an earlier live execution.
"""
from __future__ import annotations

from copy import deepcopy
import time
from types import SimpleNamespace

from agents.agent_orchestrator import Request
from core.order_evidence import extract_order_evidence, merge_order_evidence, render_order_evidence, resolve_order_id
from demo_business.store import load_fixture
from evaluation import benchmark as bench
from evaluation.intent_transport import canonical_payload_hash
from evaluation.quality_arm_runtime import build_arm_runtime
from evaluation.task_evidence import CaptureTools, deterministic_checks
from evaluation.task_runtime import _result
from evaluation.task_transport import Frame, context


ARMS = ("A", "D")
SCHEMA = "serveflow-expanded-task-artifact-v1"
CASE_IDS = tuple(f"et{i:02d}" for i in range(1, 25))
_ACTUAL = ("trace_integrity", "tool_permissions", "trusted_identity", "tool_arguments", "business_evidence")
_GATES = ("artifact_binding", "execution_eligible", "actual_trace_integrity", "actual_tool_permissions",
          "actual_trusted_identity", "actual_tool_arguments", "actual_business_evidence", "required_tools",
          "order_results_valid", "auxiliary_disclosure_check", "tool_task_correct", "expected_status")


def cases():
    """Read frozen source questions only; no outputs, clients or configuration."""
    data = bench.read_json(bench.DATASET_ROOT / "expanded-tasks-v1.json")
    if (data.get("schema_version") != "serveflow-expanded-tasks-v1"
            or [c.get("id") for c in data.get("cases", [])] != list(CASE_IDS)):
        raise ValueError("expanded_task_cases_changed")
    return deepcopy(data["cases"])


def _source(case):
    if type(case) is not dict or case.get("id") not in CASE_IDS:
        raise ValueError("expanded_task_case_required")
    original = next(row for row in cases() if row["id"] == case["id"])
    if case != original:
        raise ValueError("expanded_task_case_changed")
    return original


def _identity(arm, case, frame_case_id):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("expanded_task_arm_required")
    original = _source(case)
    if frame_case_id != f"{arm}-{original['id']}":
        raise ValueError("expanded_task_frame_mismatch")
    return original


def _invalid():
    return {"checks": dict.fromkeys(_GATES, False), "blockers": ["invalid_expanded_task_artifact"],
            "mechanisms": {"fresh_read_each_relevant_turn": None, "exact_projection": None,
                           "required_agents": None, "original_execution_complete": None},
            "semantic_review_required": True}


def checks(case, artifact):
    """Recompute source/native/handler gates; never infer answer correctness.

    Unlike the old single-turn adapter, order binding uses the original supplied
    history. It is not regenerated from model prose or started from an empty
    history. No gate rewards a named role or exact D-rendered answer.
    """
    try:
        case = _source(case)
        if type(artifact) is not dict:
            return _invalid()
        arm, case_id = artifact.get("arm"), case["id"]
        expected_request = f"expanded-task-{case_id}-1"
        inp, trusted = bench.model_input(case), bench.fixture_context(case)
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
        bound = bool(artifact.get("schema_version") == SCHEMA and arm in ARMS
            and artifact.get("case_id") == f"{arm}-{case_id}" and artifact.get("source_case_id") == case_id
            and artifact.get("data_kind") in {"controlled-fixture", "live-mysql-and-isolated-chroma"}
            and artifact.get("input_sha256") == canonical_payload_hash(inp)
            and artifact.get("trusted_context_sha256") == canonical_payload_hash(trusted)
            and turn.get("turn_id") == "turn-1" and turn.get("question") == inp["message"]
            and turn.get("history") == inp["history"] and turn.get("request_id") == expected_request
            and output.get("request_id") == expected_request
            and captured.get("case_id") == case_id and captured.get("turn_id") == 1
            and artifact.get("answer") == output.get("response")
            and artifact.get("execution_status") == output.get("status"))
        replay = CaptureTools(case_id, 1)
        replay.calls = deepcopy(calls)
        joined = replay.reconcile(SimpleNamespace(**output))
        associated = not joined["blockers"] and joined["calls"] == calls
        actual_case = {"id": case_id, "input": {**inp, **trusted},
                       "expected": {"required_agents": [], "required_tools": case["expected"]["required_tools"]},
                       "references": case["references"]}
        observed = deterministic_checks(actual_case, SimpleNamespace(**output), captured, load_fixture())
        d = observed["checks"]
        selection = resolve_order_id(inp["message"], inp["history"])
        records, fresh, valid_results = [], False, True
        for call in calls:
            if type(call) is not dict:
                return _invalid()
            bound &= (call.get("case_id") == case_id and call.get("turn_id") == 1
                      and call.get("request_id") == expected_request)
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
        result = {"artifact_binding": bool(bound), "execution_eligible": eligible,
            "actual_trace_integrity": bool(d["trace_integrity"] and associated),
            "actual_tool_permissions": d["tool_permissions"], "actual_trusted_identity": d["trusted_identity"],
            "actual_tool_arguments": d["tool_arguments"], "actual_business_evidence": d["business_evidence"],
            "required_tools": d["required_tools"], "order_results_valid": bool(valid_results),
            "auxiliary_disclosure_check": d["auxiliary_disclosure_check"],
            "tool_task_correct": bool(bound and actual_safe and valid_results and d["required_tools"]),
            "expected_status": bool(status)}
        return {"checks": result,
            "blockers": list(dict.fromkeys([*captured["blockers"], *(k for k, value in result.items() if not value)])),
            "mechanisms": {"fresh_read_each_relevant_turn": fresh if has_order else None,
                "exact_projection": bool(merged and merged.status in {"found", "not_found"}
                    and render_order_evidence(merged) in output["response"]) if has_order else None,
                "required_agents": None, "original_execution_complete": d["execution_complete"]},
            "execution_observation": {"actual_status": output.get("status"),
                "expected_status": case["expected"]["expected_status"], "scripted_history_turn_count": len(inp["history"]),
                "provided_history_preserved": turn.get("history") == inp["history"],
                "semantic_review_required": True},
            "tool_task_applicable": case["expected"]["tool_task"], "semantic_review_required": True}
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
        return _invalid()


async def execute_task(arm, case, sdk, data, *, ensure_healthy, frame_case_id):
    """One source-bound turn on a new role pool and evidence cache for each arm."""
    case = _identity(arm, case, frame_case_id)
    ensure_healthy()
    proof = getattr(data, "proof", None)
    kind = proof.get("kind") if type(proof) is dict else None
    if kind is None and type(proof) is dict and proof.get("schema_version") == "serveflow-task-live-data-v1":
        kind = "live-mysql-and-isolated-chroma"
    if kind not in {"controlled-fixture", "live-mysql-and-isolated-chroma"}:
        raise ValueError("invalid_evaluation_data_kind")
    started = time.perf_counter()
    orchestrator, _manager = await build_arm_runtime(arm, sdk, data)
    inp, trusted = bench.model_input(case), bench.fixture_context(case)
    history = deepcopy(inp["history"])
    request = Request(message=inp["message"], history=history or None,
        user_id="expanded-task-" + trusted["customer_id"], conv_id="expanded-task-" + case["id"],
        request_id=f"expanded-task-{case['id']}-1",
        context="[本次对话历史]\n" + "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:]) if history else "",
        conversation_state={}, demo_customer_id=trusted["customer_id"], demo_order_reader=data.order_reader)
    selection = resolve_order_id(request.message, request.history)
    capture = CaptureTools(case["id"], 1, call_context=lambda role: context(agent=role),
                           expected_order_id=selection.order_id if selection.status == "resolved" else None)
    frame = Frame(frame_case_id, "turn-1", evidence_guard=capture.assert_send_allowed)
    capture.install(orchestrator)
    try:
        with context(frame):
            ensure_healthy()
            actual = await orchestrator.run(request)
            observed = capture.reconcile(actual)
            capture.assert_healthy()
            ensure_healthy()
            output = _result(actual)
    finally:
        capture.restore()
    artifact = {"schema_version": SCHEMA, "case_id": frame_case_id,
        "source_case_id": case["id"], "arm": arm, "data_kind": kind,
        "input_sha256": canonical_payload_hash(inp), "trusted_context_sha256": canonical_payload_hash(trusted),
        "turns": [{"turn_id": "turn-1", "question": inp["message"], "history": inp["history"],
                   "request_id": request.request_id, "result": output, "tool_evidence": observed}],
        "answer": actual.response, "execution_status": actual.status,
        "wall_latency_ms": (time.perf_counter() - started) * 1000,
        "actor_path_latency_ms": actual.latency_ms, "judge": None,
        "judge_policy": "not-run; seven-dimensional source/native/handler review required",
        "semantic_review": None}
    artifact["shared_checks"] = checks(case, artifact)
    artifact["review_status"] = ("pending_review" if all(artifact["shared_checks"]["checks"].values())
                                  and not artifact["shared_checks"]["blockers"] else "failed")
    return artifact
