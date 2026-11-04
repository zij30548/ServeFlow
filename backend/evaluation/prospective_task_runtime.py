"""Single-turn A/D evaluation adapter for the frozen prospective questions.

Reuse the real role factory, native tool loop, handler capture and source checks.
There is no sender, authorization, database setup, Judge or semantic auto-pass.
The caller owns the frozen protocol, durable shared quota, SDK and deadlines.
"""
from __future__ import annotations

from copy import deepcopy
import time

from agents.agent_orchestrator import Request
from core.order_evidence import resolve_order_id
from evaluation import benchmark as bench
from evaluation import prospective_validation as design
from evaluation.quality_arm_runtime import build_arm_runtime
from evaluation.quality_scoring import shared_checks
from evaluation.task_evidence import CaptureTools
from evaluation.task_runtime import _result
from evaluation.task_transport import Frame, context


ARMS = ("A", "D")
CASE_IDS = tuple(f"p{i:02d}" for i in range(1, 13))
# Conservative physical envelope, enforced by the caller's persistent sender.
# Up to three initial model-backed roles plus two specialist -> General fallbacks,
# each with the unchanged three-round loop. No additional Judge requests.
PHASE_CAPS = {"intent": 1, "agent": 15, "composer": 1,
              "rag_rewrite": 3, "rag_rerank": 3}
PER_CASE_ARM_MAX = sum(PHASE_CAPS.values())
MAX_TASK_REQUESTS = len(CASE_IDS) * len(ARMS) * PER_CASE_ARM_MAX


def cases():
    value = design._cases(bench.BACKEND_ROOT, "tasks")
    design.validate_suite(value)
    return deepcopy(value["cases"])


def _case(case_id):
    if type(case_id) is not str or case_id not in CASE_IDS:
        raise ValueError("prospective_task_case_required")
    return next(case for case in cases() if case["id"] == case_id)


def _identity(arm, case_id, frame_case_id):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("prospective_task_arm_required")
    if type(case_id) is not str or case_id not in CASE_IDS:
        raise ValueError("prospective_task_case_required")
    if frame_case_id != f"{arm}-{case_id}":
        raise ValueError("prospective_task_frame_mismatch")


def _source_case(case):
    """Adapt representation only; no changed questions, gold or identities.

    The legacy checker takes turns/customer_id and treats required-agent labels
    as diagnostic. Prospective tasks impose actual tool permissions instead of
    forcing one named route; all gold remains outside the actor Request.
    """
    return {"id": case["id"], "turns": [case["input"]["message"]],
        "customer_id": case["input"]["customer_id"],
        "expected": {"required_agents": [], **deepcopy(case["expected"])},
        "references": deepcopy(case["references"])}


def checks(case_id, artifact):
    """Observed execution/source gates, never proof of a correct answer.

    The old checker only admitted partial results for mutation requests. This
    set explicitly permits a truthful partial for not-prescribed status. Keep
    that old observation but apply this set's pre-frozen status rules openly.
    Neither expected status nor vacuously safe p11 tool omission proves coverage
    or refusal correctness; all seven dimensions still require semantic review.
    """
    case = _case(case_id)
    value = shared_checks(_source_case(case), artifact)
    result = deepcopy(value)
    old_eligible = result["checks"]["execution_eligible"]
    artifact_bound = bool(type(artifact) is dict
        and artifact.get("schema_version") == "serveflow-prospective-task-artifact-v1"
        and artifact.get("source_case_id") == case_id
        and artifact.get("arm") in ARMS
        and artifact.get("case_id") == f"{artifact.get('arm')}-{case_id}"
        and artifact.get("data_kind") in {"controlled-fixture", "live-mysql-and-isolated-chroma"})
    result["checks"]["artifact_binding"] &= artifact_bound
    turns = artifact.get("turns", []) if type(artifact) is dict else []
    complete = type(turns) is list and len(turns) == 1
    turn = turns[0] if complete and type(turns[0]) is dict else {}
    output = turn.get("result", {})
    output = output if type(output) is dict else {}
    observed_status = output.get("status")
    expected_status = case["expected"]["expected_status"]
    eligible = bool(complete and artifact.get("execution_status") == observed_status
                    and turn.get("request_id") == f"prospective-task-{case_id}-1"
                    and observed_status in {"success", "partial"}
                    and type(output.get("response")) is str and output["response"].strip())
    status_matches = bool(eligible and (expected_status == "not-prescribed"
                                       or observed_status == expected_status))
    result["checks"]["execution_eligible"] = eligible
    result["checks"]["expected_status"] = status_matches
    result["blockers"] = list(dict.fromkeys([
        *(name for name in value["blockers"] if name != "execution_eligible"),
        *(name for name in ("artifact_binding", "execution_eligible", "expected_status")
          if not result["checks"][name]),
    ]))
    result["execution_observation"] = {"actual_status": observed_status,
        "expected_status": expected_status, "original_mutation_only_gate": old_eligible,
        "prospective_eligible": eligible, "semantic_review_required": True}
    result["tool_task_applicable"] = case["expected"]["tool_task"]
    result["semantic_review_required"] = True
    return result


async def execute_arm_case(arm, case_id, sdk, data, *, ensure_healthy, frame_case_id):
    """One real Orchestrator turn using the caller's bounded, counted SDK.

    No Judge is created; labels, references and reviewer expectations never
    enter a model request. A/D gets a separate fresh runtime and evidence cache.
    Frame identity is accounting metadata; both arms have identical actor IDs.
    """
    _identity(arm, case_id, frame_case_id)
    case = _case(case_id)
    ensure_healthy()
    proof = getattr(data, "proof", None)
    kind = proof.get("kind") if type(proof) is dict else None
    if kind is None and type(proof) is dict and proof.get("schema_version") == "serveflow-task-live-data-v1":
        kind = "live-mysql-and-isolated-chroma"
    if kind not in {"controlled-fixture", "live-mysql-and-isolated-chroma"}:
        raise ValueError("invalid_evaluation_data_kind")
    started = time.perf_counter()
    orchestrator, _manager = await build_arm_runtime(arm, sdk, data)
    actor_input = bench.model_input(case)
    trusted = bench.fixture_context(case)
    request = Request(message=actor_input["message"], history=None,
        user_id="prospective-task-" + trusted["customer_id"],
        conv_id="prospective-task-" + case_id, request_id=f"prospective-task-{case_id}-1",
        context="", conversation_state={}, demo_customer_id=trusted["customer_id"],
        demo_order_reader=data.order_reader)
    selection = resolve_order_id(request.message, request.history)
    capture = CaptureTools(case_id, 1, call_context=lambda role: context(agent=role),
        expected_order_id=selection.order_id if selection.status == "resolved" else None)
    frame = Frame(frame_case_id, "turn-1", evidence_guard=capture.assert_send_allowed)
    capture.install(orchestrator)
    try:
        with context(frame):
            ensure_healthy()
            actual = await orchestrator.run(request)
            observed = capture.reconcile(actual)
            capture.assert_healthy()
            ensure_healthy()  # Never let production fallback hide provider/data failures.
            output = _result(actual)
    finally:
        # Closing capture also makes an old retained frame unable to spend.
        capture.restore()
    artifact = {"schema_version": "serveflow-prospective-task-artifact-v1",
        "case_id": frame_case_id, "source_case_id": case_id, "arm": arm,
        "data_kind": kind, "turns": [{"turn_id": "turn-1", "question": request.message,
            "request_id": request.request_id, "result": output, "tool_evidence": observed}],
        "answer": actual.response, "execution_status": actual.status,
        "wall_latency_ms": (time.perf_counter() - started) * 1000,
        "actor_path_latency_ms": actual.latency_ms, "judge": None,
        "judge_policy": "not-run; seven-dimensional source/native/handler review required",
        "semantic_review": None}
    artifact["shared_checks"] = checks(case_id, artifact)
    artifact["review_status"] = ("pending_review"
        if all(artifact["shared_checks"]["checks"].values())
        and not artifact["shared_checks"]["blockers"] else "failed")
    return artifact
