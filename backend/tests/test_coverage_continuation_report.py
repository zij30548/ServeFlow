"""Offline report/review tests; no grants, model sends or real storage services."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from evaluation import coverage_continuation_report as report
from evaluation import coverage_expansion as prep
from evaluation.coverage_budget import fixed_plan, job_specs
from evaluation.expanded_validation import DIMENSIONS


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("report checks must be offline")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == ".env" or path.suffix in {".key", ".pem"}:
            denied()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


def write(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def batch(tmp_path):
    _, _, cases, conversations = prep.load_validated()
    protocol = {"jobs": job_specs(), "plan": fixed_plan(), "cases": {
        **{row["id"]: row for suite in ("tasks", "retrieval") for row in cases[suite]},
        **{row["id"]: row for row in conversations["conversations"]}}}
    root = tmp_path / "batch"; root.mkdir()
    write(root, "protocol.json", protocol)
    state = {"attempts": 0, "pending": [], "active_case": None, "paused": False, "completed_cases": []}
    sealed = {"operations": {}, "cases": {}}
    return root, protocol, state, sealed


def seal_case(batch, identity, artifact, *, outcome="success"):
    root, _, state, sealed = batch
    artifact = {**deepcopy(artifact), "frame_case_id": identity,
                "evidence_kind": artifact.get("evidence_kind", "controlled-transport")}
    if artifact.get("schema_version") != report.FAILURE_SCHEMA and "phase_observations" not in artifact:
        operations = [row for row in sealed["operations"].values() if row["case_id"] == identity]
        artifact.update(phase_observations=[{"phase": row["phase"], "agent": row.get("agent")} for row in operations],
                        physical_requests_observed=len(operations))
    digest = write(root, f"case-{identity}.json", artifact)
    sealed["cases"][identity] = {"artifact_sha256": digest, "outcome": outcome}
    state["completed_cases"].append(identity)
    return artifact


def physical(batch, identity, *, phase="intent", agent=None, outcome="received", tokens=(30, 4)):
    root, _, state, sealed = batch
    state["attempts"] += 1
    number = state["attempts"]
    op_id = identity + f":turn-1:{phase}:{agent or 'none'}:{number}"
    operation = {"case_id": identity, "attempt": number, "phase": phase, "agent": agent,
                 "turn_id": "turn-1", "outcome": outcome,
                 "payload_sha256": "a" * 64, "request_bytes_sha256": "b" * 64}
    raw = {**operation, "operation_id": op_id, "latency_ms": 42.,
           "response": {"usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}}
    operation["output_sha256"] = write(root, f"physical-{number:03d}.json", raw)
    sealed["operations"][op_id] = operation
    return op_id


def review_for(batch, identity, *, flags=None, blockers=(), burden="none"):
    root, _, _, sealed = batch
    return {"frame_case_id": identity,
            "artifact_sha256": sealed["cases"][identity]["artifact_sha256"],
            "reviewer": "controlled fixture, not a quality claim", "method": "controlled-fixture",
            "dimensions": flags or dict.fromkeys(DIMENSIONS, True),
            "blocking_errors": list(blockers), "reading_burden": burden,
            "reason": "Controlled report boundaries only; not semantic model validation."}


def result(batch, **kwargs):
    root, protocol, state, sealed = batch
    return report.build_report(protocol, state, sealed, root=root, **kwargs)


def fake_native(monkeypatch, flags=None):
    values = dict.fromkeys(report.TOOL_CHECKS, True)
    values.update(execution_eligible=True, expected_status=True)
    values.update(flags or {})
    monkeypatch.setattr(report, "_native_checks", lambda *a, **k: {
        "checks": dict(values), "blockers": [k for k, v in values.items() if not v]})


def test_all136_positions_and168_turns_remain_when_nothing_ran(batch):
    r = result(batch)
    assert r["planned_positions"] == len(r["rows"]) == 136
    assert r["planned_user_turns"] == 168 and r["reserved_requests"] == 0
    assert r["position_statuses"] == {"not_executed": 136}
    assert r["groups"]["validation/tasks/D"]["correct_handling"] == {"numerator": 0, "denominator": 28, "value": 0}
    assert r["groups"]["validation/tasks/D"]["tool_task_correct"]["denominator"] == 27
    assert r["retrieval_groups"]["enhanced"]["validation/evidence"]["planned"] == 24
    assert r["retrieval_groups"]["baseline"]["validation/out_of_scope"]["statuses"] == {"not_run": 8}
    assert not r["execution_complete"] and not r["quality_pass_claim"]
    assert r["intent_classification"] == {"status": "not_measured", "historical_scores_changed": False}


def test_original_stop_is_history_and_old_timeout_remains_failed_not_success(batch):
    root, _, state, _ = batch
    write(root, "stopped.json", {"error_code": "timeout", "historical": True})
    identity = "D-CT-D04"
    physical(batch, identity, outcome="execution_error", tokens=(None, None))
    seal_case(batch, identity, {"schema_version": report.FAILURE_SCHEMA,
        "execution_status": "failed", "continuation_failure": {"error_code": "timeout"},
        "tool_checks": {"checks": {"fabricated_success": True}}}, outcome="failed")
    write(root, f"review-{identity}.json", review_for(batch, identity))
    r = result(batch)
    row = next(row for row in r["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed" and not row["execution_eligible_for_review"]
    assert row["execution_segment"] == "original-prefix"
    assert row["checks"]["blockers"] == ["execution_failure"]
    assert not r["paused"]  # Explicit current state, not the preserved old stop.
    usage = r["physical_resources"]["usage"]
    assert usage["input_tokens"] == {"known_sum": None, "known_count": 0, "unknown_count": 1}
    assert r["paired_groups"]["development"]["incomplete_pairs"] == 12


def test_pending_is_counted_exactly_once_and_unknown_not_zero(batch):
    _, _, state, _ = batch
    state.update(attempts=1, pending=["A-CT-D01:turn-1:intent:none:1"], active_case="A-CT-D01", paused=True)
    r = result(batch)
    assert r["pending_physical"] == r["physical_resources"]["recorded_total"] == 1
    assert r["physical_resources"]["usage"]["physical_requests"]["known_sum"] == 1
    assert r["physical_resources"]["usage"]["output_tokens"]["known_sum"] is None
    assert r["position_statuses"] == {"unknown": 1, "not_executed": 135}


def test_resource_count_and_physical_hash_cannot_be_fabricated(batch):
    root, _, state, _ = batch
    physical(batch, "A-CT-D01")
    state["attempts"] += 1
    with pytest.raises(ValueError, match="physical_count"):
        result(batch)
    state["attempts"] -= 1
    (root / "physical-001.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="sealed_file_changed"):
        result(batch)


def test_native_tools_and_manual_semantics_are_separate_not_auto_pass(batch, monkeypatch):
    fake_native(monkeypatch)
    identity = "A-CT-D01"
    physical(batch, identity)
    seal_case(batch, identity, {"execution_status": "success"})
    r = result(batch)
    row = next(row for row in r["rows"] if row["case_id"] == identity)
    assert row["status"] == "pending_review" and row["tool_task_correct"]
    group = r["groups"]["development/tasks/A"]
    assert group["correct_handling"]["numerator"] == 0
    assert group["tool_task_correct"]["numerator"] == 1
    flags = dict.fromkeys(DIMENSIONS, True); flags["conditions_preserved"] = False
    write(batch[0], f"review-{identity}.json", review_for(batch, identity, flags=flags,
        blockers=["Controlled amount unit error"], burden="some"))
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed" and row["tool_task_correct"]


def test_recorded_alltrue_checks_cannot_hide_failed_recomputed_identity(batch, monkeypatch):
    fake_native(monkeypatch, {"actual_trusted_identity": False})
    identity = "D-CT-D01"
    physical(batch, identity)
    seal_case(batch, identity, {"execution_status": "success", "tool_checks": {"checks": {"all": True}}})
    write(batch[0], f"review-{identity}.json", review_for(batch, identity))
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed" and not row["tool_task_correct"]
    assert row["checks"]["checks"]["actual_trusted_identity"] is False


def test_one_bad_group_dimension_cannot_pass_five_turn_conversation(batch, monkeypatch):
    fake_native(monkeypatch)
    identity = "M-CC-V01"
    physical(batch, identity)
    seal_case(batch, identity, {"execution_status": "success"})
    flags = dict.fromkeys(DIMENSIONS, True); flags["coverage"] = False
    write(batch[0], f"review-{identity}.json", review_for(batch, identity, flags=flags,
        blockers=["One of the five turns omits a legally requested field"], burden="heavy"))
    group = result(batch)["groups"]["validation/conversations/current"]
    assert group["correct_handling"]["numerator"] == 0 and group["correct_handling"]["denominator"] == 6
    assert group["dimensions"]["coverage"]["numerator"] == 0
    assert group["reading_burden"]["heavy"] == 1 and group["reading_burden"]["unreviewed"] == 5


def source_rows(*ids):
    corpus, _, _, _ = prep.load_validated()
    chunks = prep.source_chunks(corpus["documents"])
    return [{key: row[key] for key in ("document_id", "title", "content", "version", "chunk")} | {"score": .5}
            for row in chunks.values() if row["explicit_document_id"] in ids]


def test_rag_same_document_hit_is_not_complete_fact_coverage(batch):
    identity = "R-CR-D01"
    desired = batch[1]["cases"]["CR-D01"]["expected"]["relevant_document_ids"][0]
    rows = source_rows(desired)
    artifact = {"status": "success", "baseline": {"status": "success", "rows": rows},
                "enhanced": {"status": "success", "rows": rows}}
    physical(batch, identity, phase="rag_rewrite"); physical(batch, identity, phase="rag_rerank")
    seal_case(batch, identity, artifact)
    r = result(batch)
    observed = next(row for row in r["retrieval"]["enhanced"] if row["case_id"] == "CR-D01")
    assert observed["document_hit_at_1"] and observed["document_hit_at_3"]
    assert observed["all_required_facts_at_3"]
    group = r["retrieval_groups"]["enhanced"]["development/answerable"]
    assert group["hit_at_1"] == group["hit_at_3"] == {"numerator": 1, "denominator": 8, "value": .125}
    assert group["mrr_at_3"]["numerator"] == 1
    assert group["all_required_facts_at_3"]["numerator"] == 1
    assert group["micro_required_fact_recall_at_3"]["numerator"] == observed["required_fact_count"]
    assert group["measurement_complete"] is False


def test_rag_rank2_keeps_hit1_separate_from_hit3_and_mrr(batch):
    identity = "R-CR-D01"
    rows = [source_rows("CX-D02")[0], source_rows("CX-D01")[0]]
    physical(batch, identity, phase="rag_rewrite"); physical(batch, identity, phase="rag_rerank")
    seal_case(batch, identity, {"status": "success", "baseline": {"rows": rows, "status": "success"},
                             "enhanced": {"rows": rows, "status": "success"}})
    row = next(row for row in result(batch)["retrieval"]["baseline"] if row["case_id"] == "CR-D01")
    assert row["document_hit_at_3"] and not row["document_hit_at_1"]
    assert row["reciprocal_rank_at_3"] == .5


def test_long_document_hit_does_not_hide_the_missing_exception_chunk(batch):
    identity = "R-CR-V15"
    rows = source_rows("CX-V15")[:1]
    physical(batch, identity, phase="rag_rewrite"); physical(batch, identity, phase="rag_rerank")
    seal_case(batch, identity, {"status": "success", "baseline": {"status": "success", "rows": rows},
                             "enhanced": {"status": "success", "rows": rows}})
    row = next(row for row in result(batch)["retrieval"]["enhanced"] if row["case_id"] == "CR-V15")
    assert row["document_hit_at_3"] and not row["all_required_facts_at_3"]
    assert 0 < row["required_fact_recall_at_3"] < 1


def test_observed_sdk_phases_must_match_the_durable_physical_records(batch, monkeypatch):
    fake_native(monkeypatch)
    identity = "D-CT-D01"
    physical(batch, identity)
    artifact = seal_case(batch, identity, {"execution_status": "success"})
    artifact["phase_observations"].append({"phase": "agent", "agent": "billing"})
    artifact["physical_requests_observed"] += 1
    batch[3]["cases"][identity]["artifact_sha256"] = write(batch[0], f"case-{identity}.json", artifact)
    write(batch[0], f"review-{identity}.json", review_for(batch, identity))
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed" and not row["physical_structure_valid"]


def test_out_of_scope_empty_timeout_is_not_a_healthy_rejection(batch):
    identity = "R-CR-V25"
    physical(batch, identity, phase="rag_rewrite", outcome="execution_error", tokens=(None, None))
    seal_case(batch, identity, {"schema_version": report.FAILURE_SCHEMA, "execution_status": "failed",
                             "continuation_failure": {"error_code": "timeout"}}, outcome="failed")
    r = result(batch)
    group = r["retrieval_groups"]["enhanced"]["validation/out_of_scope"]
    assert group["hit_at_3"] is None and group["healthy_empty_rejections"]["numerator"] == 0
    assert group["statuses"] == {"failed": 1, "not_run": 7}
    assert group["out_of_scope_false_hits"]["numerator"] == 0


def test_healthy_empty_and_irrelevant_match_are_reported_separately(batch):
    for identity, rows in (("R-CR-V25", []), ("R-CR-V26", source_rows("CX-V01"))):
        physical(batch, identity, phase="rag_rewrite"); physical(batch, identity, phase="rag_rerank")
        seal_case(batch, identity, {"status": "success", "baseline": {"status": "success", "rows": rows},
                                 "enhanced": {"status": "success", "rows": rows}})
    group = result(batch)["retrieval_groups"]["enhanced"]["validation/out_of_scope"]
    assert group["healthy_empty_rejections"]["numerator"] == group["out_of_scope_false_hits"]["numerator"] == 1


def test_rag_missing_actual_rerank_never_counts_enhanced_success(batch):
    identity = "R-CR-D01"; rows = source_rows("CX-D01")
    physical(batch, identity, phase="rag_rewrite")
    seal_case(batch, identity, {"status": "success", "baseline": {"status": "success", "rows": rows},
                             "enhanced": {"status": "success", "rows": rows}})
    r = result(batch)
    baseline = next(row for row in r["retrieval"]["baseline"] if row["case_id"] == "CR-D01")
    enhanced = next(row for row in r["retrieval"]["enhanced"] if row["case_id"] == "CR-D01")
    assert baseline["document_hit_at_3"]
    assert enhanced["status"] == "failed" and not enhanced["document_hit_at_3"]


def test_reviews_cannot_change_artifact_identity_evidence_kind_or_structure(batch):
    identity = "D-CT-D01"
    artifact = seal_case(batch, identity, {"execution_status": "success"})
    value = review_for(batch, identity)
    assert report.validate_review(value, batch[1], root=batch[0]) == value
    wrong = deepcopy(value); wrong["method"] = "executor-rubric-review"
    with pytest.raises(ValueError, match="review_value"):
        report.validate_review(wrong, batch[1], root=batch[0])
    wrong = deepcopy(value); wrong["dimensions"]["coverage"] = 1
    with pytest.raises(ValueError, match="review_value"):
        report.validate_review(wrong, batch[1], root=batch[0])
    fake = deepcopy(artifact); fake["evidence_kind"] = "live-model"
    with pytest.raises(ValueError, match="review_artifact_changed"):
        report.validate_review(value, batch[1], root=batch[0], artifacts={identity: fake})
    wrong = deepcopy(value); wrong["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="review_binding"):
        report.validate_review(wrong, batch[1], root=batch[0])


def test_fixed_plan_unknown_operations_and_source_callback_fail_closed(batch):
    invalid = deepcopy(batch[1]); invalid["jobs"].reverse()
    with pytest.raises(ValueError, match="fixed_plan"):
        report.build_report(invalid, batch[2], batch[3], root=batch[0])
    seen = []
    def verify():
        seen.append(True)
        if len(seen) > 1:
            raise ValueError("source_changed_during_report")
    with pytest.raises(ValueError, match="source_changed_during_report"):
        result(batch, verify_source=verify)
    assert len(seen) == 2


def test_real_native_evidence_recomputed_instead_of_recorded_alltrue(batch):
    from test_coverage_runtime import execute
    artifact, replies, _ = asyncio.run(execute("CT-D01", "D"))
    identity = "D-CT-D01"
    for frame, _payload in replies.calls:
        physical(batch, identity, phase=frame.phase, agent=frame.agent)
    seal_case(batch, identity, artifact)
    r = result(batch)
    row = next(row for row in r["rows"] if row["case_id"] == identity)
    assert row["status"] == "pending_review" and row["tool_task_correct"]
    changed = deepcopy(artifact)
    changed["turns"][0]["tool_evidence"]["calls"][0]["trusted_customer_id"] = "demo-customer-b"
    changed["tool_checks"] = {"checks": {"all": True}, "blockers": []}
    digest = write(batch[0], f"case-{identity}.json", changed)
    batch[3]["cases"][identity]["artifact_sha256"] = digest
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed" and not row["tool_task_correct"]


def test_partial_real_conversation_cannot_become_eligible_from_saved_checks(batch):
    from test_coverage_conversation_runtime import execute
    async def run():
        artifact, replies, _, factory = await execute(0)
        try:
            return deepcopy(artifact), [(frame.phase, frame.agent) for frame, _payload in replies.calls]
        finally:
            await factory.redis.aclose()
    artifact, frames = asyncio.run(run())
    identity = "M-CC-D01"
    for phase, agent in frames:
        physical(batch, identity, phase=phase, agent=agent)
    seal_case(batch, identity, artifact)
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "pending_review"
    artifact["turns"].pop()
    artifact["tool_checks"] = {"checks": {"all": True}, "blockers": []}
    batch[3]["cases"][identity]["artifact_sha256"] = write(batch[0], f"case-{identity}.json", artifact)
    row = next(row for row in result(batch)["rows"] if row["case_id"] == identity)
    assert row["status"] == "failed"
