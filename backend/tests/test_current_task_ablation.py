"""Current A/D execution-contract checks using local files and MockTransport.

All authorizations and reviews below are controlled fixtures, never permission
or evidence for a real model trial. The actual SDK/native handlers are exercised
with sockets denied; the original task questions and labels remain unchanged.
"""
import asyncio
from copy import deepcopy
import json
import os
import socket
import stat
from types import SimpleNamespace

import httpx
import pytest

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, current_task_ablation as module
from evaluation import prospective_task_runtime as tasks
from evaluation import prospective_trial as original
from evaluation import task_supervisor
from evaluation.intent_transport import TrialTransportError
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import CURRENT
from test_prospective_trial import BASE, Data, response, send


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("current task ablation tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module.proof, "source_proof", lambda: {"controlled_source": "a" * 64})
    monkeypatch.setattr(module.proof, "artifacts", lambda: {"controlled_artifacts": "b" * 64})
    monkeypatch.setattr(module.proof, "prior_proof", lambda: {"old_57_16_16_records": "c" * 64})
    monkeypatch.setattr(module.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(module.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(module.task_data, "offline_data_proof", lambda: {"controlled_data": True})
    def config(*, secret=False):
        return {"base_url": BASE, "model": module.base.MODEL,
                **({"api_key": "controlled-only-not-a-real-key"} if secret else {})}
    monkeypatch.setattr(module.base, "_configuration", config)
    return tmp_path


@pytest.fixture
def prepared(configured):
    trial = module.CurrentTaskAblation(configured / "new-current-task-ablation")
    asyncio.run(trial.prepare())
    return trial


def approve(trial, limit=120):
    return trial.authorize("controlled offline fixture, no live-call allowance",
                           max_requests=limit, confirmed=True)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def assessment(trial, key, *, failed_dimension=None, error=None):
    dimensions = dict.fromkeys(original.design.DIMENSIONS, True)
    if failed_dimension:
        dimensions[failed_dimension] = False
    return {"logical_case_id": key,
        "artifact_sha256": bench.sha256(trial.root / f"case-{key}.json"),
        "reviewer": "controlled fixture reviewer", "method": "controlled-fixture",
        "dimensions": dimensions, "blocking_errors": [error] if error else [],
        "reading_burden": "some",
        "reason": "Controlled review plumbing only; not a semantic quality measurement."}


def execute_controlled(trial, monkeypatch, *, stop_before=None):
    """Real SDK plus real role/native/tool path, never actual model inference."""
    data, observed = Data(), []
    source = {case["id"]: case for case in module.cases()}
    async def handle(request):
        frame = CURRENT.get()
        payload = json.loads(request.content)
        observed.append((frame, deepcopy(payload)))
        cid = frame.case_id[2:]
        question = source[cid]["input"]["message"]
        if frame.phase == "intent":
            intent = "technical" if cid in {"p07", "p08"} else "order_status"
            return response(json.dumps({"intent": intent, "confidence": .99, "reasoning": "controlled"}))
        if frame.phase == "composer":
            return response("Controlled natural summary; semantics require a separate review.")
        assert frame.phase == "agent", "No Judge/background or separate retrieval evaluation request"
        required = payload.get("tool_choice", {}).get("name")
        content = payload["messages"][-1]["content"]
        has_results = isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in content)
        if not required and has_results:
            return response("Controlled natural summary; semantics require a separate review.")
        tool = required or ("lookup_error_code" if frame.agent == "technical" else "read_demo_order")
        if tool == "lookup_error_code":
            code = "500" if "500" in question else "401" if "401" in question else "403"
            args = {"error_code": code}
        else:
            assert tool == "read_demo_order"
            import re
            args = {"order_id": re.search(r"DEMO-[0-9]{4}", question).group(0)}
        return response(content=[{"type": "tool_use", "id": f"controlled_{len(observed)}",
                                  "name": tool, "input": args}])
    actual = module.execute_arm_case
    if stop_before is not None:
        async def bounded(arm, cid, *args, **kwargs):
            if f"{arm}-{cid}" == stop_before:
                raise ValueError("controlled stop at a predeclared test position")
            return await actual(arm, cid, *args, **kwargs)
        monkeypatch.setattr(module, "execute_arm_case", bounded)
    run = trial.execute(delegate_factory=lambda frame: httpx.MockTransport(handle), controlled_data=data)
    if stop_before is not None:
        with pytest.raises(ValueError, match="controlled stop"):
            asyncio.run(run)
        report = trial.report()
    else:
        report = asyncio.run(run)
    return report, data, observed


def test_new_identity_keeps_exact_twelve_tasks_and_twenty_four_paired_captures(prepared):
    p = prepared.verify()
    assert module.CurrentTaskAblation().root == module.ROOT
    assert module.ROOT != original.ROOT
    assert p["schema_version"] == prepared.protocol_schema != original.ProspectiveTrial.protocol_schema
    assert prepared.authorization_schema != original.ProspectiveTrial.authorization_schema
    assert p["cases"] == tasks.cases() == module.cases()
    assert len(p["plan"]) == 24 and all(row["turn_ids"] == ["turn-1"] for row in p["plan"])
    assert all(not row["case_id"].startswith("R-") for row in p["plan"])
    for index in range(12):
        pair = p["plan"][index * 2:index * 2 + 2]
        cid = f"p{index + 1:02d}"
        order = ("A", "D") if index % 2 == 0 else ("D", "A")
        assert [row["case_id"] for row in pair] == [f"{arm}-{cid}" for arm in order]
    assert len(p["initial_intent_requests"]) == 24
    assert "initial_rewrite_requests" not in p
    for left, right in zip(p["initial_intent_requests"][::2], p["initial_intent_requests"][1::2]):
        assert left["source_case_id"] == right["source_case_id"]
        assert left["payload"] == right["payload"]
        assert left["payload_sha256"] == right["payload_sha256"]
        assert left["request_bytes_sha256"] == right["request_bytes_sha256"]
        case = next(c for c in p["cases"] if c["id"] == left["source_case_id"])
        wire = json.dumps(left["payload"], ensure_ascii=False)
        assert case["input"]["message"] in wire
        for forbidden in ("required_facts", "missing_facts", "forbidden_claims", "expected_status", "related_prior_cases"):
            assert forbidden not in wire
        for fact in case["expected"]["required_facts"]:
            assert fact not in wire
    assert p["authorized_model_requests"] == p["judge_requests"] == p["external_business_requests"] == 0
    assert p["proposed_model_requests"] == p["limits"]["physical_requests"] == module.MAX_REQUESTS == 120
    assert p["limits"]["mathematical_phase_envelope"] == module.MATHEMATICAL_REQUESTS == 552
    assert "judge" not in p["phase_parameters"]
    assert prepared.ledger(p).send_limit == 120
    assert not (prepared.root / "authorization.json").exists()
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
               for path in prepared.root.rglob("*") if path.is_file())


def test_full_plan_denominators_are_not_replaced_by_the_completed_subset(prepared):
    report = prepared.report()
    assert report["planned_jobs"] == len(report["cases"]) == 24
    assert report["completed_jobs"] == report["reserved_requests"] == 0
    assert report["all_cases_previously_exposed"] is True
    for arm in ("A", "D"):
        assert {name: metric["denominator"] for name, metric in report["task_metrics"][arm].items()} == {
            "correct_handling": 12, "supported_completion": 10, "tool_tasks": 11, "boundary_handling": 2}
    assert report["comparison_ready"] is False and report["task_delta_percentage_points"] is None
    assert len(report["task_pairs"]) == 12
    assert report["task_pair_counts"]["unpaired_pending_unknown_or_unexecuted"] == 12


def test_without_new_authorization_no_secret_sdk_database_or_worker_is_opened(prepared, monkeypatch):
    import anthropic
    configuration = module.base._configuration
    def config(*, secret=False):
        assert secret is False
        return configuration(secret=False)
    def denied(*args, **kwargs):
        raise AssertionError("authorization must precede resource access")
    monkeypatch.setattr(module.base, "_configuration", config)
    monkeypatch.setattr(module.task_data, "build_live_data", denied)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", denied)
    monkeypatch.setattr(module.subprocess, "Popen", denied)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    with pytest.raises(FileNotFoundError): asyncio.run(prepared.execute())
    with pytest.raises(FileNotFoundError): prepared.run()
    assert prepared.report()["reserved_requests"] == 0


@pytest.mark.parametrize("limit", [0, -1, 121, 552, True, 1.5, None])
def test_invalid_allowance_cannot_create_or_enlarge_a_call_budget(prepared, limit):
    with pytest.raises(module.base.TrialError): approve(prepared, limit)
    assert not (prepared.root / "authorization.json").exists()


@pytest.mark.parametrize("limit", [0, 121, 552, True])
def test_direct_ledger_entry_cannot_bypass_trial_cap(prepared, limit):
    p = prepared.verify(); before = snapshot(prepared.root / "ledger")
    with pytest.raises((module.base.TrialError, TaskBudgetError)): prepared.ledger(p, send_limit=limit)
    assert snapshot(prepared.root / "ledger") == before


def test_smaller_allowance_is_spent_atomically_and_cannot_be_topped_up(prepared):
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1); observed = []
    async def handle(request): observed.append(request); return response()
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01", lease=lease)
        transport = module.TaskAblationTransport(BASE, prepared.root, budget, lease, prepared.verify,
            delegate_factory=lambda frame: httpx.MockTransport(handle))
        async def run():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await send(transport)
                with pytest.raises(TrialTransportError): await send(transport)
        asyncio.run(run())
    report = prepared.report()
    assert report["reserved_requests"] == len(observed) == 1 and report["paused"] is True
    assert prepared.ledger(p).status()["attempts"] == 1
    with pytest.raises(module.base.TrialError): approve(prepared, 120)


def test_unknown_attempt_survives_reopen_without_a_fresh_resource_or_send(prepared, monkeypatch):
    import anthropic
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01", lease=lease)
        budget.reserve("A-p01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    def denied(*args, **kwargs):
        raise AssertionError("unknown send must not open another SDK or database")
    monkeypatch.setattr(module.task_data, "build_live_data", denied)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", denied)
    reopened = module.CurrentTaskAblation(prepared.root)
    report = reopened.report()
    assert report["reserved_requests"] == report["pending_physical"] == 1
    resources = report["physical_resources"]
    assert report["pending_usage_unknown_count"] == 1
    assert "all reserved attempts" in report["resource_scope"]
    assert resources["recorded_total"] == 1
    assert resources["usage"]["physical_requests"] == {
        "known_sum": 1, "known_count": 1, "unknown_count": 0}
    for field in ("input_tokens", "output_tokens"):
        assert resources["usage"][field] == {
            "known_sum": None, "known_count": 0, "unknown_count": 1}
    assert resources["latency_ms"]["p50"] is resources["latency_ms"]["p95"] is None
    assert resources["latency_ms"]["known_count"] == 0
    assert resources["latency_ms"]["unknown_count"] == 1
    assert report["arm_resources"]["A"]["recorded_total"] == 1
    assert report["arm_resources"]["D"]["recorded_total"] == 0
    assert report["cases"][0]["status"] == "unknown"
    assert report["task_metrics"]["A"]["correct_handling"]["denominator"] == 12
    assert report["task_delta_percentage_points"] is None
    monkeypatch.setattr(module, "ROOT", reopened.root)
    with pytest.raises((module.base.TrialError, TaskBudgetError)): asyncio.run(reopened.execute())
    assert reopened.report()["reserved_requests"] == 1


@pytest.mark.parametrize("component", ["source", "artifacts", "prior", "dependencies", "data", "environment"])
def test_frozen_identity_changes_refuse_before_sending(prepared, monkeypatch, component):
    altered = lambda: {"changed_after_freeze": True}
    if component == "source": monkeypatch.setattr(module.proof, "source_proof", altered)
    elif component == "artifacts": monkeypatch.setattr(module.proof, "artifacts", altered)
    elif component == "prior": monkeypatch.setattr(module.proof, "prior_proof", altered)
    elif component == "dependencies": monkeypatch.setattr(module.engine, "_dependencies", altered)
    elif component == "data": monkeypatch.setattr(module.task_data, "offline_data_proof", altered)
    else: monkeypatch.setattr(module.history, "effective_environment", altered)
    with pytest.raises(module.base.TrialError): prepared.verify()


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(authorized_model_requests=1),
    lambda p: p["plan"].reverse(),
    lambda p: p["cases"][0]["expected"].update(required_tools=[]),
    lambda p: p["initial_intent_requests"][0]["payload"].update(gold="answer"),
])
def test_protocol_cannot_change_questions_gold_order_or_authorization(prepared, mutate):
    value = prepared.verify(); mutate(value)
    (prepared.root / "protocol.json").write_text(json.dumps(value, ensure_ascii=False))
    with pytest.raises((module.base.TrialError, bench.BenchmarkError, TaskBudgetError)): prepared.verify()


def test_old_schema_protocol_and_genesis_cannot_be_relabelled_as_new_authorization(prepared):
    approve(prepared, 1)
    path = prepared.root / "authorization.json"; valid = bench.read_json(path)
    for change in (lambda v: v.update(schema_version="serveflow-order-answer-recheck-authorization-v1"),
                   lambda v: v.update(protocol_sha256="d" * 64),
                   lambda v: v.update(initial_chain_hash="e" * 64)):
        damaged = deepcopy(valid); change(damaged)
        path.write_text(json.dumps(damaged))
        with pytest.raises(module.base.TrialError): prepared._approval(prepared.verify())
    path.write_text(json.dumps(valid))
    assert prepared.report()["reserved_requests"] == 0


def test_partial_execution_does_not_publish_a_complete_comparison_or_reuse_old_success(prepared, monkeypatch):
    approve(prepared)
    report, data, observed = execute_controlled(prepared, monkeypatch, stop_before="D-p01")
    assert data.closed and observed and (prepared.root / "case-A-p01.json").exists()
    assert not (prepared.root / "case-D-p01.json").exists()
    assert report["completed_jobs"] == 1
    assert report["comparison_ready"] is False and report["task_delta_percentage_points"] is None
    for arm in ("A", "D"):
        assert report["task_metrics"][arm]["correct_handling"]["denominator"] == 12
    assert report["task_pair_counts"]["unpaired_pending_unknown_or_unexecuted"] == 12
    assert report["semantic_reviews_imported"] == 0


def test_real_sdk_native_handlers_complete_only_plumbing_until_manual_seven_dimension_review(prepared, monkeypatch):
    approve(prepared)
    report, data, observed = execute_controlled(prepared, monkeypatch)
    assert data.closed and data.orders
    assert report["planned_jobs"] == report["completed_jobs"] == len(report["cases"]) == 24
    assert report["reserved_requests"] == len(observed) < 120
    assert report["pending_physical"] == 0 and report["paused"] is False
    assert report["semantic_reviews_imported"] == 0 and report["comparison_ready"] is False
    assert report["task_delta_percentage_points"] is None
    assert all(row["status"] == "pending_review" for row in report["cases"])
    assert all(frame.phase != "judge" and not frame.case_id.startswith("R-") for frame, _ in observed)
    assert any(frame.case_id.startswith("A-") and frame.phase == "agent"
               and "tool_choice" not in payload for frame, payload in observed)
    assert any(frame.case_id.startswith("D-") and frame.phase == "agent"
               and "tool_choice" in payload for frame, payload in observed)
    for row in report["cases"]:
        key = row["logical_case_id"]
        artifact = bench.read_json(prepared.root / f"case-{key}.json")
        assert artifact["trial_kind"] == "current-task-ablation-v1"
        assert artifact["judge"] is artifact["semantic_review"] is None
        assert artifact["shared_checks"]["checks"]["tool_task_correct"]
        assert artifact["evidence_kind"] == "controlled-transport"
    for _, payload in observed:
        encoded = json.dumps(payload, ensure_ascii=False)
        assert all(word not in encoded for word in ("required_facts", "forbidden_claims", "expected_status", "related_prior_cases"))
    for row in report["cases"]:
        prepared.review(assessment(prepared, row["logical_case_id"]))
    reviewed = prepared.report()
    assert reviewed["comparison_ready"] is True and reviewed["task_delta_percentage_points"] == 0
    assert reviewed["task_pair_counts"]["both_passed"] == 12
    before = snapshot(prepared.root)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("completed plan cannot restart a worker"))
    assert prepared.run() == {"exit_code": 0, "already_complete": True}
    assert snapshot(prepared.root) == before


def test_metadata_omissions_and_invented_facts_remain_different_review_findings(prepared, monkeypatch):
    approve(prepared)
    report, _, _ = execute_controlled(prepared, monkeypatch)
    prepared.review(assessment(prepared, "A-p01", failed_dimension="coverage", error="missing_business_record_time"))
    prepared.review(assessment(prepared, "A-p03", failed_dimension="conditions_preserved", error="unsupported_refund_amount"))
    result = prepared.report(); rows = {r["logical_case_id"]: r for r in result["cases"]}
    for key, expected in (("A-p01", "missing_business_record_time"), ("A-p03", "unsupported_refund_amount")):
        assert rows[key]["status"] == "failed"
        assert rows[key]["tool_status"] == "passed"
        assert rows[key]["review"]["blocking_errors"] == [expected]
    assert rows["A-p01"]["review"]["dimensions"]["conditions_preserved"] is True
    assert rows["A-p03"]["review"]["dimensions"]["coverage"] is True
    assert result["comparison_ready"] is False and result["task_delta_percentage_points"] is None
    assert result["task_metrics"]["A"]["correct_handling"]["denominator"] == 12


def test_old_case_and_wrong_review_identity_cannot_be_imported_even_with_resealed_file(prepared, monkeypatch):
    approve(prepared)
    report, _, _ = execute_controlled(prepared, monkeypatch, stop_before="D-p01")
    review = assessment(prepared, "A-p01")
    for change in (lambda v: v.update(logical_case_id="D-n02"),
                   lambda v: v.update(artifact_sha256="f" * 64),
                   lambda v: v["dimensions"].update(coverage=None)):
        forged = deepcopy(review); change(forged)
        with pytest.raises((module.base.TrialError, FileNotFoundError)): prepared.review(forged)
    assert not list(prepared.root.glob("review-*.json"))
    p = prepared.verify(); original_artifact = bench.read_json(prepared.root / "case-A-p01.json")
    # Independently create a new temporary ledger with an old-shaped artifact,
    # rather than weakening the integrity check on a real sealed case.
    other = module.CurrentTaskAblation(prepared.root.parent / "forged-old-case")
    asyncio.run(other.prepare())
    data = Data(); module.base._write(other.root, "data-proof.json", data.proof)
    data_hash = bench.sha256(other.root / "data-proof.json")
    module.base._write(other.root, "data-proof-seal.json", {
        "protocol_sha256": bench.sha256(other.root / "protocol.json"), "data_proof_sha256": data_hash})
    old = deepcopy(original_artifact); old.pop("trial_kind")
    old.update(protocol_sha256=bench.sha256(other.root / "protocol.json"), data_proof_sha256=data_hash)
    module.base._write(other.root, "case-A-p01.json", old)
    budget = other.ledger(other.verify())
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01", lease=lease)
        budget.finish_case("A-p01", bench.sha256(other.root / "case-A-p01.json"), outcome="failed", lease=lease)
    with pytest.raises(module.base.TrialError): other.report()


def test_closed_worker_has_only_new_module_fixed_root_and_three_thousand_six_hundred_seconds(prepared, monkeypatch):
    approve(prepared)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    observed = []
    def supervised(command, *, root, deadline_s):
        observed.append((command, root, deadline_s))
        return {"exit_code": 0, "timed_out": False, "cancelled": False}
    monkeypatch.setattr(task_supervisor, "supervise", supervised)
    assert prepared.run()["exit_code"] == 0
    assert observed == [([module.sys.executable, "-B", "-u", "-m", "evaluation.current_task_ablation", "_worker"],
                         prepared.root, 3600)]
    monkeypatch.chdir(prepared.root.parent)
    with pytest.raises(module.base.TrialError): prepared.run()


@pytest.mark.parametrize("result, expected", [
    ({"exit_code": 0, "timed_out": False, "cancelled": False}, 0),
    ({"exit_code": 0, "timed_out": True, "cancelled": False}, 2),
    ({"exit_code": 0, "timed_out": False, "cancelled": True}, 2),
])
def test_cli_success_requires_worker_success_without_timeout_or_cancellation(monkeypatch, result, expected):
    called = []
    def run():
        called.append("run")
        return deepcopy(result)
    monkeypatch.setattr(module, "CurrentTaskAblation", lambda: SimpleNamespace(run=run))
    monkeypatch.setattr(module.sys, "argv", ["current_task_ablation", "run"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == expected
    assert called == ["run"]

