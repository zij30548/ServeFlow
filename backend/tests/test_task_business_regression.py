"""New business rubric and fixed accounting; all sends/stores are controlled."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as module, task_trial as engine
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_budget import TaskBudget, TaskBudgetError
from evaluation.task_evidence import fixture_order_result
from evaluation.task_transport import Frame, context
from demo_business.store import load_fixture


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
CONTROLLED_PROOF = {"kind": "controlled-fixture"}
ORIGINAL_ARTIFACTS = module._artifacts
ORIGINAL_PRIOR_PROOF = module.prior_proof


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("business regression tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    for key in tuple(os.environ):
        if key.startswith("ECHOMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    reads = []
    def configuration(*, secret=False):
        reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE,
                **({"api_key": "controlled-only-not-persisted"} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"model": base.MODEL, "max_tokens": 256, "temperature": .1,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": case["input"]["message"]}]}
        return {"case_id": case["id"], "operation_id": "controlled-capture",
            "input_sha256": canonical_payload_hash(case["input"]), "payload": payload,
            "payload_sha256": canonical_payload_hash(payload),
            "request_bytes_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(engine, "_dependencies", lambda: {"controlled-runtime": "1"})
    monkeypatch.setattr(engine, "offline_data_proof", lambda: {"kind": "controlled-offline-proof"})
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled-source": "a" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-archive": "b" * 64, "old_scores_recomputed": False})
    monkeypatch.setattr(module.repair, "data_binding", lambda: canonical_payload_hash(CONTROLLED_PROOF))
    monkeypatch.setattr(engine, "_mysql_url", lambda: None)
    return SimpleNamespace(config_reads=reads)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def prepared(tmp_path):
    trial = module.BusinessRegression(tmp_path / "business-v4")
    return trial, asyncio.run(trial.prepare())


def approved(trial):
    return trial.authorize("Controlled test allowance only, never live model authorization.", confirmed=True)


def budget(tmp_path):
    return module.BusinessTaskBudget(tmp_path / "ledger", "a" * 64, module.fixed_plan(), initialize=True)


def controlled_artifact(case, *, judges=1., omit_projection=False):
    from core.order_evidence import extract_order_evidence, render_order_evidence, resolve_order_id
    turns, history = [], []
    for index, question in enumerate(case["turns"], 1):
        calls = []
        selected = resolve_order_id(question, history=history)
        answer = "Controlled response; not semantic model acceptance."
        if "read_demo_order" in case["expected"]["required_tools"] and selected.order_id:
            value = fixture_order_result(load_fixture(), case["customer_id"], selected.order_id)
            evidence = extract_order_evidence(value, expected_order_id=selected.order_id,
                                              trusted_customer_id=case["customer_id"])
            text = render_order_evidence(evidence)
            if not omit_projection:
                answer = text
            calls.append({"name": "read_demo_order", "input": {"order_id": selected.order_id},
                "result": value, "success": True, "tool_use_id": f"controlled-read-{index}"})
        turns.append({"turn_id": f"turn-{index}", "question": question,
            "result": {"response": answer, "status": "success", "latency_ms": 1},
            "tool_evidence": {"calls": calls, "blockers": []},
            "judge_overall": judges, "judge": {"judge_failed": judges is None}})
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return {"case_id": case["id"], "execution_status": "success", "turns": turns,
        "deterministic": {"checks": {"controlled_runner_boundary_only": True}, "blockers": []},
        "judge_ready": judges is not None, "data_kind": "controlled-fixture", "semantic_review": None,
        "answer": turns[-1]["result"]["response"], "wall_latency_ms": 1, "actor_path_latency_ms": 1}


def publish_case(trial, p, key="e01", *, judge=1., omit_projection=False, partial=False, tool_failure=False):
    case = next(c for c in module.cases() if c["id"] == key)
    result = controlled_artifact(case, judges=judge, omit_projection=omit_projection)
    if partial:
        result["turns"][-1]["result"]["status"] = "partial"
        result["execution_status"] = "failed"
        result["deterministic"]["checks"]["execution_complete"] = False
        result["deterministic"]["blockers"] = ["execution_complete"]
    if tool_failure:
        result["deterministic"]["checks"]["business_evidence"] = False
        result["deterministic"]["blockers"].append("business_evidence")
    result.update({"evidence_kind": "controlled-transport", "grading_version": "task-business-v4",
                   "order_projection": module.projection_checks(case, result),
                   "business_execution": module.business_execution_checks(case, result)})
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case(key, lease=lease)
        for number in range(1, len(case["turns"]) + 1):
            operation = ledger.reserve(key, f"turn-{number}", "intent", None,
                                       "a" * 64, "b" * 64, lease=lease)
            physical = {"latency_ms": 1, "response": {"usage": {"input_tokens": 1, "output_tokens": 1}}}
            filename = f"physical-{operation['attempt']:03d}.json"
            base._write(trial.root, filename, physical)
            ledger.finish(operation["operation_id"], "received", bench.sha256(trial.root / filename), lease=lease)
        filename = f"case-{key}.json"
        base._write(trial.root, filename, result)
        ledger.finish_case(key, bench.sha256(trial.root / filename),
                           "success" if result["execution_status"] == "success" else "failed", lease=lease)
    return result


def review(trial, key="e01", *, blocks=None, coverage=True):
    return trial.review({"case_id": key, "artifact_sha256": bench.sha256(trial.root / f"case-{key}.json"),
        "reviewer": "controlled executor", "method": "controlled-fixture",
        "dimensions": {name: coverage if name == "coverage" else True for name in engine.REVIEW_FIELDS},
        "reason": "Controlled grading-mechanics test, not semantic quality evidence.",
        "blocking_errors": blocks or [], "reading_burden": "some"})


def test_plan_reuses_exact_questions_labels_and_references_without_marking_blind():
    cases = module.cases()
    assert cases[:6] == engine.cases()
    original = {c["id"]: c for c in bench.load_suites()["tasks"]["cases"]}
    for case in cases[6:]:
        assert case["turns"] == [original[case["id"]]["input"]["message"]]
        assert case["expected"] == original[case["id"]]["expected"]
        assert case["references"] == original[case["id"]]["references"]
        assert original[case["id"]]["split"] == "validation"
    assert tuple(c["id"] for c in cases) == module.CASE_IDS
    assert len(cases) == 12 and sum(len(c["turns"]) for c in cases) == 13
    assert "author-visible" in module.REVIEW_POLICY["groups"]


def test_prepare_has_zero_authorization_no_secrets_and_explicit_312_upper_bound(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 312
    assert p["limits"] == {**engine.LIMITS, "cases": 12, "turns": 13, "physical_requests": 312}
    assert p["http_limits"] == engine.HTTP_LIMITS and p["http_limits"]["request_deadline_s"] == 8
    assert p["effective_environment"]["ECHOMIND_LLM_TIMEOUT_S"] == 15
    assert trial.verify() == p and trial.ledger(p).status()["attempts"] == 0
    assert not any(isolated.config_reads)
    assert len(p["initial_intent_requests"]) == 12
    for item, case in zip(p["initial_intent_requests"], module.cases()):
        assert item["payload"]["messages"] == [{"role": "user", "content": case["turns"][0]}]
        assert case["expected"]["rationale"] not in json.dumps(item["payload"], ensure_ascii=False)
    with pytest.raises(base.TrialError, match="already_exists"):
        asyncio.run(trial.prepare())


@pytest.mark.parametrize("mutation", ["source", "questions", "policy", "data", "root", "timeout"])
def test_changed_identity_blocks_before_store_or_send(tmp_path, monkeypatch, mutation):
    trial, p = prepared(tmp_path)
    approved(trial)
    before = snapshot(trial.root)
    if mutation == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"changed": "c" * 64})
    elif mutation == "questions":
        original = module.cases
        def changed():
            value = original(); value[0]["turns"][0] += "changed"; return value
        monkeypatch.setattr(module, "cases", changed)
    elif mutation == "policy": monkeypatch.setitem(module.REVIEW_POLICY, "version", "changed")
    elif mutation == "data": monkeypatch.setattr(module.repair, "data_binding", lambda: "d" * 64)
    elif mutation == "timeout": monkeypatch.setenv("ECHOMIND_LLM_TIMEOUT_S", "30")
    else:
        shutil.copytree(trial.root, tmp_path / "copied-batch")
        trial = module.BusinessRegression(tmp_path / "copied-batch")
    store = AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", store)
    with pytest.raises((base.TrialError, TaskBudgetError)):
        asyncio.run(trial.execute())
    store.assert_not_called()
    assert snapshot(tmp_path / "business-v4") == before


def test_missing_or_cross_batch_authorization_cannot_send(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    store = AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", store)
    with pytest.raises(Exception): asyncio.run(trial.execute())
    assert trial.ledger(p).status()["attempts"] == 0
    store.assert_not_called()
    with pytest.raises(base.TrialError): trial.authorize("", confirmed=True)
    with pytest.raises(base.TrialError): trial.authorize("test", confirmed=False)
    approved(trial)
    with pytest.raises(FileExistsError): approved(trial)
    copied, other = module.BusinessRegression(tmp_path / "different"), None
    other = asyncio.run(copied.prepare())
    shutil.copyfile(trial.root / "authorization.json", copied.root / "authorization.json")
    (copied.root / "authorization.json").chmod(0o600)
    with pytest.raises(base.TrialError, match="authorization_changed"):
        copied._approval(other)


def test_new_budget_never_interprets_old_six_case_identity_and_requires_exact_plan(tmp_path):
    with pytest.raises(TaskBudgetError):
        module.BusinessTaskBudget(tmp_path / "bad", "a" * 64, engine._plan(engine.cases()), initialize=True)
    assert not (tmp_path / "bad").exists()
    ledger = budget(tmp_path)
    before = snapshot(ledger.root)
    with pytest.raises(TaskBudgetError): TaskBudget(ledger.root, "a" * 64, engine._plan(engine.cases()))
    assert snapshot(ledger.root) == before
    assert ledger.max_requests == 312
    assert ledger.initial_chain_hash != hashlib.sha256(b"serveflow-task-budget:identity:v1\n" + module.accounting._canonical(ledger._identity)).hexdigest()


def test_per_turn_global_and_inflight_limits_fail_without_mutating_ledger(tmp_path):
    ledger = budget(tmp_path)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        first = [ledger.reserve("e01", "turn-1", "agent", role, "b" * 64, "c" * 64, lease=lease)
                 for role in ("general", "technical", "billing")]
        before = snapshot(ledger.root)
        with pytest.raises(TaskBudgetError, match="ledger_busy"):
            ledger.reserve("e01", "turn-1", "agent", "billing", "b" * 64, "c" * 64, lease=lease)
        assert snapshot(ledger.root) == before
        for item in reversed(first): ledger.finish(item["operation_id"], "received", "d" * 64, lease=lease)
    state = ledger._empty(); state["active_case"] = "e01"
    state["attempts"] = 312
    with pytest.raises(TaskBudgetError, match="quota"): ledger._reserve_check(state, "e01", "turn-1", "intent", None)
    state["attempts"] = 311; state["turn_attempts"][("e01", "turn-1")] = 24
    with pytest.raises(TaskBudgetError, match="quota"): ledger._reserve_check(state, "e01", "turn-1", "intent", None)
    assert module.accounting.MAX_REQUESTS == 168 and module.accounting.LIMITS["cases"] == 6


def test_all_twelve_cases_can_complete_in_order_without_resending_or_exceeding_plan(tmp_path):
    ledger = budget(tmp_path)
    with ledger.exclusive_run() as lease:
        for item in module.fixed_plan():
            ledger.start_case(item["case_id"], lease=lease)
            for turn in item["turn_ids"]:
                event = ledger.reserve(item["case_id"], turn, "intent", None, "b" * 64, "c" * 64, lease=lease)
                ledger.finish(event["operation_id"], "received", "d" * 64, lease=lease)
            ledger.finish_case(item["case_id"], "e" * 64, lease=lease)
        with pytest.raises(TaskBudgetError): ledger.start_case("e01", lease=lease)
    assert ledger.status()["attempts"] == 13
    assert ledger.status()["completed_cases"] == list(module.CASE_IDS)
    assert module.BusinessTaskBudget(ledger.root, "a" * 64, module.fixed_plan()).status() == ledger.status()


@pytest.mark.parametrize("finish", [False, "unknown", "execution_error", "cancelled"])
def test_uncertain_or_failed_send_remains_charged_and_blocks_restart(tmp_path, finish):
    ledger = budget(tmp_path)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        event = ledger.reserve("e01", "turn-1", "intent", None, "b" * 64, "c" * 64, lease=lease)
        if finish: ledger.finish(event["operation_id"], finish, error_code="timeout" if finish == "execution_error" else None, lease=lease)
    before = snapshot(ledger.root)
    reopened = module.BusinessTaskBudget(ledger.root, "a" * 64, module.fixed_plan())
    with pytest.raises(TaskBudgetError):
        with reopened.exclusive_run(): pass
    assert reopened.status()["attempts"] == 1 and snapshot(ledger.root) == before


def test_projection_checks_native_read_each_turn_and_exact_source_block_not_model_prose():
    case = next(c for c in module.cases() if c["id"] == "e04")
    artifact = controlled_artifact(case)
    assert all(module.projection_checks(case, artifact)["checks"].values())
    artifact["turns"][1]["tool_evidence"]["calls"] = []
    assert not module.projection_checks(case, artifact)["checks"]["fresh_native_order_read"]
    artifact = controlled_artifact(case, omit_projection=True)
    assert not module.projection_checks(case, artifact)["checks"]["order_projection_present"]
    artifact = controlled_artifact(case)
    artifact["turns"][0]["tool_evidence"]["calls"][0]["result"]["order"]["customer_id"] = "demo-customer-b"
    assert not module.projection_checks(case, artifact)["checks"]["order_projection_valid"]


def test_business_acceptance_does_not_rewrite_or_hide_low_judge_score(tmp_path):
    trial, p = prepared(tmp_path)
    publish_case(trial, p, judge=.05)
    review(trial)
    report = trial.report()
    row = report["cases"][0]
    assert row["business_status"] == "passed" and row["legacy_judge_gated_status"] == "failed"
    assert row["judge_scores"] == [.05]
    assert report["business_counts"]["passed"] == 1 and report["legacy_judge_gated_counts"]["passed"] == 0
    assert report["planned_denominator"] == 12 and report["business_counts"]["not_executed"] == 11
    assert report["groups"]["validation"]["planned_denominator"] == 6


@pytest.mark.parametrize("kind", ["blocker", "coverage", "projection", "missing_judge"])
def test_judge_perfect_score_cannot_override_fact_failure_and_unknown_usage_is_not_zero(tmp_path, kind):
    trial, p = prepared(tmp_path)
    publish_case(trial, p, judge=None if kind == "missing_judge" else 1., omit_projection=kind == "projection")
    review(trial, blocks=["invented refund rule"] if kind == "blocker" else None, coverage=kind != "coverage")
    report = trial.report(); row = report["cases"][0]
    assert row["business_status"] == ("passed" if kind == "missing_judge" else "failed")
    if kind in {"blocker", "coverage", "missing_judge"}: assert row["legacy_judge_gated_status"] == "failed"
    assert report["review_dimensions"]["coverage"]["planned_denominator"] == 12


def test_report_retains_active_unknown_and_all_unexecuted_in_denominator(tmp_path):
    trial, p = prepared(tmp_path)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        ledger.reserve("e01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    report = trial.report()
    assert report["business_counts"]["unknown"] == 1 and report["business_counts"]["not_executed"] == 11
    assert report["reserved_requests"] == 1 and report["sealed_requests"] == 0
    assert report["usage"]["input_tokens"] == {"known_sum": None, "known_requests": 0, "unknown_requests": 1}


@pytest.mark.parametrize("kind", ["correct_refusal", "bad_read", "missing_projection", "review_blocker"])
def test_new_business_gate_only_accepts_complete_grounded_read_plus_reviewed_write_refusal(tmp_path, kind):
    trial, p = prepared(tmp_path)
    for key in module.CASE_IDS:
        if key == "t09": break
        publish_case(trial, p, key)
    publish_case(trial, p, "t09", partial=True, tool_failure=kind == "bad_read",
                 omit_projection=kind == "missing_projection")
    # A mutation request merely admits semantic review, it never auto-passes.
    before = trial.report()["cases"][8]
    assert before["business_status"] == ("pending_review" if kind in {"correct_refusal", "review_blocker"} else "failed")
    review(trial, "t09", blocks=["claimed refund executed"] if kind == "review_blocker" else None)
    row = trial.report()["cases"][8]
    assert row["business_status"] == ("passed" if kind == "correct_refusal" else "failed")
    assert row["legacy_judge_gated_status"] == "failed"
    assert row["legacy_deterministic_pass"] is False
    assert row["business_execution"]["partial_turns"] == [{"turn_id": "turn-1", "requested_mutation": True,
                                                          "semantic_refusal_review_required": True}]


def test_non_mutation_partial_missing_turn_failed_status_and_wrong_turn_binding_never_qualify():
    simple = next(c for c in module.cases() if c["id"] == "e01")
    artifact = controlled_artifact(simple)
    artifact["turns"][0]["result"]["status"] = "partial"
    assert not module.business_execution_checks(simple, artifact)["checks"]["partial_only_requested_mutation"]
    write = next(c for c in module.cases() if c["id"] == "t09")
    artifact = controlled_artifact(write)
    artifact["turns"][0]["result"]["status"] = "failed"
    assert not module.business_execution_checks(write, artifact)["checks"]["no_failed_turn"]
    artifact["turns"] = []
    assert not module.business_execution_checks(write, artifact)["checks"]["actual_turns_complete"]
    artifact = controlled_artifact(write)
    artifact["turns"][0]["turn_id"] = "turn-2"
    assert not module.business_execution_checks(write, artifact)["checks"]["actual_turns_bound"]


def test_reviews_bind_the_exact_case_and_cannot_be_rewritten(tmp_path):
    trial, p = prepared(tmp_path)
    publish_case(trial, p)
    value = review(trial)
    with pytest.raises(FileExistsError): trial.review(value)
    forged = {**value, "artifact_sha256": "a" * 64}
    with pytest.raises(base.TrialError, match="binding"): trial._review(forged)


def test_new_production_evidence_and_tooling_are_included_in_freeze(monkeypatch):
    monkeypatch.setattr(module.repair, "_artifacts", lambda: {"old-frozen.py": "a" * 64})
    monkeypatch.setattr(bench, "sha256", lambda _: "b" * 64)
    frozen = ORIGINAL_ARTIFACTS()
    assert {"evaluation/task_business_regression.py", "tests/test_task_business_regression.py",
        "core/order_evidence.py", "core/diagnostic_evidence.py", "tests/test_order_runtime.py",
        "tests/test_diagnostic_evidence.py", "tests/test_skill_business_grounding.py"} <= set(frozen)
    assert frozen["old-frozen.py"] == "a" * 64


def test_old_archive_link_is_readonly_and_rejects_byte_changes(tmp_path, monkeypatch):
    from evaluation import task_recheck
    root = tmp_path / "historical-v3"
    root.mkdir(mode=0o700)
    older = {"old_archive": "controlled"}
    monkeypatch.setattr(module.repair, "prior_proof", lambda: older)
    monkeypatch.setattr(module, "OLD_ROOT", root)
    protocol = {"prior_proof": older, "artifacts": {}, "plan": []}
    base._write(root, "protocol.json", protocol)
    base._write(root, "authorization.json", {"controlled": "old authorization"})
    ledger_root = root / "ledger"
    ledger_root.mkdir(mode=0o700)
    base._write(ledger_root, "journal.jsonl", {"controlled": "old journal"})
    operations = {}
    for number in range(1, 32):
        name = f"physical-{number:03d}.json"
        base._write(root, name, {"controlled": number})
        operations[str(number)] = {"attempt": number, "output_sha256": bench.sha256(root / name)}
    for name, attribute in (("protocol.json", "OLD_PROTOCOL"), ("authorization.json", "OLD_AUTHORIZATION"),
                            ("ledger/journal.jsonl", "OLD_JOURNAL")):
        monkeypatch.setattr(module, attribute, bench.sha256(root / name))
    monkeypatch.setattr(module, "OLD_TREE", task_recheck._tree(root))
    monkeypatch.setattr(task_recheck.history, "_archived_sources", lambda *_args: None)
    state = {"attempts": 31, "pending": [], "paused": True, "active_case": "e05",
        "completed_cases": ["e01", "e02", "e03", "e04"], "chain_hash": module.OLD_CHAIN}
    fake = SimpleNamespace(status=lambda: state, results=lambda: {"operations": operations, "cases": {}})
    monkeypatch.setattr(module.accounting, "TaskBudget", lambda *_args, **_kwargs: fake)
    before = snapshot(root)
    proof = ORIGINAL_PRIOR_PROOF()
    assert proof["reserved_requests"] == proof["sealed_requests"] == 31
    assert proof["old_scores_recomputed"] is False and proof["old_allowance_reused"] is False
    assert proof["old_strict_results"] == {"passed": 1, "failed": 3, "unknown": 1, "not_executed": 1}
    assert snapshot(root) == before
    (root / "physical-028.json").write_text('{"changed":"old timeout"}')
    with pytest.raises(base.TrialError, match="old_tree_changed"):
        ORIGINAL_PRIOR_PROOF()


def test_controlled_execute_uses_one_fixed_plan_and_does_not_auto_pass(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    approved(trial)
    sent = []
    async def execute(case, sdk, data, *, ensure_healthy):
        for number, question in enumerate(case["turns"], 1):
            ensure_healthy()
            with context(Frame(case["id"], f"turn-{number}", "intent")):
                await sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": question}], extra_body={"thinking": {"type": "disabled"}})
        return controlled_artifact(case)
    monkeypatch.setattr(engine, "execute_case", execute)
    def factory(frame):
        async def handle(request):
            sent.append((frame, json.loads(request.content)))
            return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
                "model": base.MODEL, "content": [{"type": "text", "text": "controlled transport only"}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 1}})
        return httpx.MockTransport(handle)
    close = AsyncMock()
    data = SimpleNamespace(proof=deepcopy(CONTROLLED_PROOF), assert_healthy=lambda: None, close=close)
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert len(sent) == report["reserved_requests"] == 13
    assert [frame.case_id for frame, _ in sent] == [item["case_id"] for item in module.fixed_plan() for _ in item["turn_ids"]]
    assert report["business_counts"]["pending_review"] == 12 and report["business_counts"]["passed"] == 0
    assert all(row["evidence_kind"] == "controlled-transport" for row in report["cases"])
    close.assert_awaited_once()
    before = snapshot(trial.root)
    asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert len(sent) == 13 and snapshot(trial.root) == before


def test_execution_fault_stops_following_cases_without_automatic_retry(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    approved(trial)
    attempted = []
    async def execute(case, sdk, data, *, ensure_healthy):
        with context(Frame(case["id"], "turn-1", "intent")):
            await sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                messages=[{"role": "user", "content": case["turns"][0]}], extra_body={"thinking": {"type": "disabled"}})
        raise AssertionError("failed send must not reach case completion")
    monkeypatch.setattr(engine, "execute_case", execute)
    def factory(frame):
        async def handle(request):
            attempted.append(frame.case_id)
            raise httpx.ReadTimeout("controlled; no external network")
        return httpx.MockTransport(handle)
    data = SimpleNamespace(proof=deepcopy(CONTROLLED_PROOF), assert_healthy=lambda: None, close=AsyncMock())
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert attempted == ["e01"]
    state = trial.ledger(p).status()
    assert state["attempts"] == 1 and state["paused"] and not state["pending"]
    assert (trial.root / "stopped.json").exists() and (trial.root / "worker-stop.json").exists()
    before = snapshot(trial.root)
    with pytest.raises(base.TrialError, match="paused"):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert attempted == ["e01"] and snapshot(trial.root) == before
