"""Controlled scoring mechanics; these are not semantic model measurements."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.tools import billing_tools, demo_order_tool
from core.order_evidence import extract_order_evidence, render_order_evidence, resolve_order_id
from demo_business.store import load_fixture
from evaluation.quality_scoring import SCORING_POLICY, shared_checks
from evaluation.task_business_regression import cases
from evaluation.task_evidence import _clone, fixture_order_result


def case(key="e02"):
    return next(c for c in cases() if c["id"] == key)


def artifact(c, *, projection=False, fresh=True):
    spec = demo_order_tool()
    turns, history = [], []
    for index, question in enumerate(c["turns"], 1):
        request_id = f"controlled-{c['id']}-{index}"
        selected = resolve_order_id(question, history=history)
        calls, traces = [], []
        text = "Controlled natural summary, not a semantic quality claim."
        if selected.status == "resolved" and (index == 1 or fresh):
            args = {"order_id": selected.order_id}
            value = fixture_order_result(load_fixture(), c["customer_id"], selected.order_id)
            _, raw = _clone(value)
            import hashlib
            calls = [{"agent": "billing", "sequence": 1, "name": "read_demo_order",
                "case_id": c["id"], "turn_id": index, "request_id": request_id,
                "input": args, "trusted_customer_id": c["customer_id"],
                "allowed_agents": list(spec.allowed_agents), "tool_scope": list(billing_tools()),
                "retry_safe": True, "input_schema": spec.input_schema, "success": True,
                "result_success": None, "result": value, "result_sha256": hashlib.sha256(raw).hexdigest(),
                "tool_use_id": f"native-{index}", "error_code": None, "latency_ms": 1.}]
            traces = [{"agent_type": "billing", "tool_name": "read_demo_order", "input": args,
                "success": True, "result_success": None, "tool_use_id": f"native-{index}"}]
            if projection:
                text = render_order_evidence(extract_order_evidence(value, expected_order_id=selected.order_id,
                    trusted_customer_id=c["customer_id"]))
        result = {"response": text, "status": "success", "agent_type": "billing",
            "agent_types": ["billing"], "tools_used": ["read_demo_order"] if calls else [],
            "tool_traces": traces, "request_id": request_id}
        turns.append({"turn_id": f"turn-{index}", "question": question, "request_id": request_id,
            "result": result, "tool_evidence": {"case_id": c["id"], "turn_id": index,
                "calls": calls, "blockers": [], "events": [], "tool_calls": len(calls), "rag_calls": 0}})
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
    return {"case_id": c["id"], "turns": turns, "answer": turns[-1]["result"]["response"]}


@pytest.mark.parametrize("projection", [True, False])
def test_natural_and_exact_answers_share_structural_gate(projection):
    c = case(); value = artifact(c, projection=projection)
    scored = shared_checks(c, value)
    assert all(scored["checks"].values()) and not scored["blockers"]
    assert scored["mechanisms"]["exact_projection"] is projection
    assert scored["semantic_review_required"] is True


def test_earlier_actual_read_can_support_static_followup_without_fresh_mechanism():
    c = case("e04"); value = artifact(c, fresh=False)
    scored = shared_checks(c, value)
    assert all(scored["checks"].values())
    assert scored["mechanisms"]["fresh_read_each_relevant_turn"] is False
    assert scored["mechanisms"]["exact_projection"] is False


def test_projection_mechanism_checks_followup_even_without_new_read():
    c = case("e04"); value = artifact(c, projection=True, fresh=False)
    scored = shared_checks(c, value)
    assert all(scored["checks"].values())
    assert scored["mechanisms"]["exact_projection"] is False
    value["turns"][1]["result"]["response"] = value["turns"][0]["result"]["response"]
    value["answer"] = value["turns"][1]["result"]["response"]
    scored = shared_checks(c, value)
    assert all(scored["checks"].values())
    assert scored["mechanisms"]["exact_projection"] is True
    assert scored["mechanisms"]["fresh_read_each_relevant_turn"] is False


def test_self_reported_pass_flags_cannot_replace_actual_evidence():
    c = case(); value = artifact(c)
    value["deterministic"] = {"checks": {"business_evidence": True, "required_tools": True}}
    value["turns"][0]["tool_evidence"]["calls"] = []
    value["turns"][0]["result"]["tool_traces"] = []
    scored = shared_checks(c, value)
    assert not scored["checks"]["required_tools"]
    assert not scored["checks"]["tool_task_correct"]


def test_structure_does_not_prove_semantics_of_a_free_summary():
    c = case(); value = artifact(c)
    value["turns"][0]["result"]["response"] = "已批准退款299元并收取手续费。"
    value["answer"] = value["turns"][0]["result"]["response"]
    scored = shared_checks(c, value)
    assert all(scored["checks"].values())  # Semantic review must reject this.
    assert scored["semantic_review_required"] is True
    assert "not proof" in SCORING_POLICY["source_support"]


def test_unsupported_output_disclosure_fails_main_gate_without_relabeling_correct_tool():
    c = case(); value = artifact(c)
    value["turns"][0]["result"]["response"] += " 演示客户B专属设备"
    value["answer"] = value["turns"][0]["result"]["response"]
    scored = shared_checks(c, value)
    assert not scored["checks"]["auxiliary_disclosure_check"]
    assert scored["checks"]["tool_task_correct"]
    assert not all(scored["checks"].values())


def test_not_found_is_valid_source_and_not_forced_to_be_found():
    c = case("e06"); value = artifact(c)
    assert value["turns"][0]["tool_evidence"]["calls"][0]["result"]["status"] == "not_found"
    assert all(shared_checks(c, value)["checks"].values())


def test_correct_partial_mutation_refusal_is_eligible_but_legacy_success_remains_false():
    c = case("t09"); value = artifact(c)
    value["turns"][0]["result"]["status"] = "partial"
    scored = shared_checks(c, value)
    assert all(scored["checks"].values())
    assert scored["mechanisms"]["original_execution_complete"] is False


def test_other_partial_is_not_a_completed_query():
    c = case(); value = artifact(c)
    value["turns"][0]["result"]["status"] = "partial"
    scored = shared_checks(c, value)
    assert scored["checks"]["tool_task_correct"]
    assert not scored["checks"]["execution_eligible"]


def test_required_role_is_retained_as_architecture_diagnostic():
    c = case(); c["expected"]["required_agents"] = ["technical"]
    scored = shared_checks(c, artifact(c))
    assert all(scored["checks"].values())
    assert scored["mechanisms"]["required_agents"] is False


@pytest.mark.parametrize("change", ["identity", "target", "sha", "native_id", "missing_trace", "request",
    "source_extra", "schema", "response", "capture_turn", "source_case", "failed_tool"])
def test_permission_source_native_and_version_corruption_cannot_pass(change):
    c = case(); value = artifact(c)
    t = value["turns"][0]; call = t["tool_evidence"]["calls"][0]
    if change == "identity": call["trusted_customer_id"] = "demo-customer-b"
    elif change == "target": call["input"]["order_id"] = "DEMO-1001"
    elif change == "sha": call["result_sha256"] = "a" * 64
    elif change == "native_id": t["result"]["tool_traces"][0]["tool_use_id"] = "other"
    elif change == "missing_trace": t["result"]["tool_traces"] = []
    elif change == "request": t["request_id"] = "other-request"
    elif change == "source_extra": call["result"]["owner_id"] = "untrusted"
    elif change == "schema": call["input"]["owner_id"] = "demo-customer-b"
    elif change == "response": value["answer"] = "different from actual turn"
    elif change == "capture_turn": t["tool_evidence"]["turn_id"] = 2
    elif change == "source_case": call["case_id"] = "another-case"
    elif change == "failed_tool": call["success"] = False
    scored = shared_checks(c, value)
    assert not all(scored["checks"].values())
    assert not scored["checks"]["tool_task_correct"] or change == "response"


@pytest.mark.parametrize("value", [None, [], {}, {"turns": None}, {"turns": []}])
def test_malformed_artifact_is_fail_closed(value):
    assert not all(shared_checks(case(), value)["checks"].values())


def test_logical_arm_case_binding_changes_no_scoring_rule():
    c = case(); baseline = artifact(c)
    expected = shared_checks(c, baseline)
    for arm in "ABCD":
        value = deepcopy(baseline)
        value.update(arm=arm, source_case_id=c["id"], case_id=f"{arm}-{c['id']}")
        assert shared_checks(c, value) == expected


def test_input_artifact_and_case_not_mutated_by_reconciliation():
    c = case(); value = artifact(c); before = deepcopy((c, value))
    shared_checks(c, value)
    assert (c, value) == before
