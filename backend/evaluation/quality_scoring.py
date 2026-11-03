"""Common structural gates for the four-arm order-evidence experiment.

No model, storage or configuration is opened here. These checks establish
execution and source identity, not relevance, faithful prose or completeness.
Those require the same seven-dimension evidence review in every arm. A correct
natural summary is eligible without matching the program's rendered text.
"""
from __future__ import annotations

from copy import deepcopy
import re
from types import SimpleNamespace

from core.order_evidence import extract_order_evidence, merge_order_evidence, render_order_evidence, resolve_order_id
from demo_business.store import load_fixture
from evaluation.task_business_regression import business_execution_checks
from evaluation.task_evidence import CaptureTools, deterministic_checks


SCORING_POLICY = {
    "version": "quality-common-source-review-v1",
    "main_gate": "complete bound execution, actual valid sources and tool permissions, seven semantic dimensions, no blocker",
    "free_summary": "may pass if directly supported, complete, source-faithful and correctly scoped; no verbatim requirement",
    "history": "same arm/case actual validated earlier tool results may support a later turn on static fixtures; assistant text is not evidence",
    "mechanisms": "per-turn fresh read and exact program projection are observations only, never main gates",
    "roles": "original required_agents labels retained as architecture diagnostics, not semantic quality gates",
    "partial": "read plus explicit refusal of a requested unsupported mutation is eligible; never call it a completed refund",
    "tool_metric": "actual tool choice, arguments, trusted identity, execution and source association; prose fact errors are graded separately",
    "source_support": "structural integrity is not proof that selected evidence answers all legitimate subquestions",
    "legacy": "original success, required-agent and Judge thresholds remain separate, never rewrite prior scores",
}
_TOOLS = ("trace_integrity", "tool_permissions", "trusted_identity", "tool_arguments",
          "business_evidence", "required_tools")
_GATES = ("artifact_binding", "execution_eligible", "actual_trace_integrity", "actual_tool_permissions",
          "actual_trusted_identity", "actual_tool_arguments", "actual_business_evidence",
          "required_tools", "order_results_valid", "auxiliary_disclosure_check", "tool_task_correct")


def _invalid():
    return {"checks": dict.fromkeys(_GATES, False), "blockers": ["invalid_quality_artifact"],
            "mechanisms": {"fresh_read_each_relevant_turn": None, "exact_projection": None,
                           "required_agents": None, "original_execution_complete": None},
            "semantic_review_required": True}


def shared_checks(case, artifact):
    """Recompute from actual per-turn evidence; do not trust stored pass flags.

The original case ID stays in requests and capture records. An arm-prefixed
ledger ID may exist outside model inputs. No arm-specific branch affects any
gate or semantic expectation.
"""
    try:
        if type(case) is not dict or type(artifact) is not dict:
            return _invalid()
        source_id = case["id"]
        identity = artifact.get("case_id")
        arm = artifact.get("arm")
        bound = identity == source_id or (arm in {"A", "B", "C", "D"}
            and identity == f"{arm}-{source_id}" and artifact.get("source_case_id") == source_id)
        turns = artifact["turns"]
        if type(turns) is not list or not 1 <= len(turns) <= len(case["turns"]) <= 2:
            return _invalid()
        execution = business_execution_checks(case, artifact)
        calls, blocks, roles, names, traces, history = [], [], [], [], [], []
        per_turn_safe = True
        per_turn_disclosure_safe = True
        valid_order_results = True
        retained, fresh_observations, projection_observations = {}, [], []
        has_order_context = "read_demo_order" in case["expected"].get("required_tools", [])
        for index, turn in enumerate(turns, 1):
            result, capture = turn["result"], turn["tool_evidence"]
            if type(result) is not dict or type(capture) is not dict:
                return _invalid()
            actual_calls, actual_blocks = capture["calls"], capture["blockers"]
            if (type(actual_calls) is not list or len(actual_calls) > 12
                    or type(actual_blocks) is not list or len(actual_blocks) > 30
                    or type(result.get("response")) is not str
                    or type(result.get("agent_types")) is not list
                    or type(result.get("tools_used")) is not list
                    or type(result.get("tool_traces")) is not list):
                return _invalid()
            bound &= (turn.get("turn_id") == f"turn-{index}"
                      and turn.get("question") == case["turns"][index - 1]
                      and capture.get("case_id") == source_id and capture.get("turn_id") == index
                      and turn.get("request_id") == result.get("request_id"))
            bound &= (type(turn.get("request_id")) is str
                      and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", turn["request_id"]) is not None)
            association = CaptureTools(source_id, index)
            association.calls = deepcopy(actual_calls)
            joined = association.reconcile(SimpleNamespace(**result))
            per_turn_safe &= not joined["blockers"] and joined["calls"] == actual_calls
            selection = resolve_order_id(turn["question"], history=history)
            current_order_results, current_read = [], False
            for call in actual_calls:
                if type(call) is not dict:
                    return _invalid()
                bound &= call.get("turn_id") == index and call.get("request_id") == turn["request_id"]
                if call.get("name") == "read_demo_order":
                    args = call.get("input")
                    if type(args) is not dict or set(args) != {"order_id"}:
                        valid_order_results = False
                    else:
                        valid_order_results &= (selection.status == "resolved"
                                                and args["order_id"] == selection.order_id)
                        value = extract_order_evidence(call.get("result"), expected_order_id=args["order_id"],
                                                       trusted_customer_id=case["customer_id"])
                        valid_order_results &= value.status in {"found", "not_found"}
                        current_read |= bool(call.get("success") and call.get("tool_use_id")
                                             and args["order_id"] == selection.order_id)
                        current_order_results.append(value)
            if has_order_context and selection.status == "resolved":
                if current_order_results:
                    retained.pop(selection.order_id, None)
                    merged = merge_order_evidence(current_order_results)
                    if merged.status in {"found", "not_found"}:
                        retained[selection.order_id] = merged
                previous = retained.get(selection.order_id)
                fresh_observations.append(current_read)
                projection_observations.append(bool(previous is not None
                    and render_order_evidence(previous) in result["response"]))
            # Check every turn, including intermediate source leaks. Gold is
            # used only by this evaluator, never sent to an answering Agent.
            local_case = {"id": source_id, "input": {"message": turn["question"],
                "history": history, "customer_id": case["customer_id"]},
                "expected": {"required_tools": [], "required_agents": []},
                "references": case.get("references", [])}
            local = deterministic_checks(local_case, SimpleNamespace(**result),
                {"calls": actual_calls, "blockers": actual_blocks}, load_fixture())
            per_turn_safe &= all(local["checks"][name] for name in _TOOLS if name != "required_tools")
            per_turn_disclosure_safe &= local["checks"]["auxiliary_disclosure_check"]
            calls.extend(actual_calls)
            blocks.extend(actual_blocks)
            roles.extend(result["agent_types"])
            names.extend(result["tools_used"])
            traces.extend(result["tool_traces"])
            history += [{"role": "user", "content": turn["question"]},
                        {"role": "assistant", "content": result["response"]}]
        bound &= artifact.get("answer") == turns[-1]["result"]["response"]
        combined = SimpleNamespace(status="success" if all(t["result"]["status"] == "success" for t in turns) else "partial",
            response=artifact["answer"], agent_types=roles, agent_type=roles[-1] if roles else None,
            tools_used=names, tool_traces=traces)
        original = deterministic_checks(case, combined, {"calls": calls, "blockers": blocks}, load_fixture())
        d = original["checks"]
        tool_correct = bool(bound and per_turn_safe and valid_order_results and all(d[name] for name in _TOOLS)
                            and execution["checks"]["actual_turns_complete"])
        checks = {"artifact_binding": bool(bound),
            "execution_eligible": all(execution["checks"].values()),
            "actual_trace_integrity": d["trace_integrity"] and per_turn_safe,
            "actual_tool_permissions": d["tool_permissions"],
            "actual_trusted_identity": d["trusted_identity"],
            "actual_tool_arguments": d["tool_arguments"],
            "actual_business_evidence": d["business_evidence"],
            "required_tools": d["required_tools"],
            "order_results_valid": bool(valid_order_results),
            "auxiliary_disclosure_check": d["auxiliary_disclosure_check"] and per_turn_disclosure_safe,
            "tool_task_correct": tool_correct}
        mechanisms = {
            "fresh_read_each_relevant_turn": all(fresh_observations) if fresh_observations else None,
            "exact_projection": all(projection_observations) if projection_observations else None,
            "required_agents": d["required_agents"],
            "original_execution_complete": d["execution_complete"],
        }
        return {"checks": checks,
                "blockers": list(dict.fromkeys([*blocks, *(name for name, ok in checks.items() if not ok)])),
                "mechanisms": mechanisms, "semantic_review_required": True}
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
        return _invalid()
