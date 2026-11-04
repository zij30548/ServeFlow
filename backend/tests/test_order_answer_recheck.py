"""Fresh fixed-recheck identity and native-handler wiring, entirely offline.

Real SDK request serialization is exercised using memory-only HTTP transports.
All receipts in this file are controlled fixtures, not live-call authorization.
"""
import asyncio
from copy import deepcopy
import json
import signal
import socket
import stat
import sys

import httpx
import pytest

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, diagnostic_repair_trial as old
from evaluation import order_answer_recheck as module
from evaluation.intent_transport import TrialTransportError
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import CURRENT
from test_prospective_trial import BASE, Data, response, send


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("order recheck tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(old.os.environ):
        if key.startswith(("ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_")):
            monkeypatch.delenv(key)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module.proof, "source_proof", lambda: {"controlled_source": "a" * 64})
    monkeypatch.setattr(module.proof, "artifacts", lambda: {"controlled_artifacts": "b" * 64})
    monkeypatch.setattr(module.proof, "prior_proof", lambda: {"old_completed_ledgers": "c" * 64})
    monkeypatch.setattr(old.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(old.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(old.task_data, "offline_data_proof", lambda: {"controlled_data": True})
    def configuration(*, secret=False):
        return {"base_url": BASE, "model": old.base.MODEL,
                **({"api_key": "controlled-only-not-a-real-key"} if secret else {})}
    monkeypatch.setattr(old.base, "_configuration", configuration)
    return tmp_path


@pytest.fixture
def prepared(configured):
    trial = module.OrderAnswerRecheck(configured / "new-answer-recheck")
    asyncio.run(trial.prepare())
    return trial


def approve(trial, limit=24):
    return trial.authorize("controlled offline recheck receipt; no live allowance",
                           max_requests=limit, confirmed=True)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def assessment(trial, key, *, coverage=True):
    dimensions = dict.fromkeys(old.design.DIMENSIONS, True)
    dimensions["coverage"] = coverage
    return {"logical_case_id": key,
        "artifact_sha256": bench.sha256(trial.root / f"case-{key}.json"),
        "reviewer": "controlled source reviewer", "method": "controlled-fixture",
        "dimensions": dimensions, "blocking_errors": [] if coverage else ["controlled omission"],
        "reading_burden": "some", "reason": "Controlled reviewer plumbing only, not a real score."}


def test_recheck_owns_new_root_protocol_authorization_and_old_contract_is_unchanged(prepared):
    p = prepared.verify()
    assert prepared.protocol_schema != old.FocusedTrial.protocol_schema
    assert prepared.authorization_schema != old.FocusedTrial.authorization_schema
    assert module.ROOT != old.ROOT
    assert module.ROOT.name == "order-answer-recheck-v1"
    assert module.OrderAnswerRecheck().root == module.ROOT
    assert old.FocusedTrial().root == old.ROOT
    assert p["schema_version"] == prepared.protocol_schema
    assert p["cases"] == old.cases() and p["plan"] == old.fixed_plan()
    assert p["limits"]["physical_requests"] == module.MAX_REQUESTS == 24
    assert p["limits"]["mathematical_phase_envelope"] == module.MATHEMATICAL_REQUESTS == 138
    assert module.MAX_REQUESTS == p["proposed_model_requests"] == 24
    assert p["limits"]["per_turn"] == 23 and p["limits"]["judge_allowed"] is False
    assert p["phase_parameters"] == json.loads(json.dumps({k: v for k, v in old.PHASE_PARAMETERS.items() if k != "judge"}))
    assert p["agent_parameters"] == json.loads(json.dumps(old.AGENT_PARAMETERS))
    assert p["authorized_model_requests"] == p["judge_requests"] == p["external_business_requests"] == 0
    assert prepared.ledger(p).send_limit == 24
    report = prepared.report()
    assert report["all_cases_previously_exposed"] is True
    assert report["trial_request_limit"] == 24 and report["mathematical_ceiling"] == 138
    assert report["schema_version"] == "serveflow-order-answer-recheck-progress-v1"


def test_six_real_sdk_initial_payloads_keep_questions_order_and_no_gold(prepared):
    p = prepared.verify()
    assert [r["case_id"] for r in p["initial_intent_requests"]] == [
        "D-p07", "D-n01", "D-p01", "D-n02", "D-p09", "D-n03"]
    for case, captured in zip(p["cases"], p["initial_intent_requests"]):
        actor = bench.model_input(case)
        assert case["input"]["message"] == actor["message"]
        wire = json.dumps(captured["payload"], ensure_ascii=False)
        assert actor["message"] in wire
        assert captured["payload"]["model"] == old.base.MODEL
        for token in ("required_facts", "missing_facts", "forbidden_claims", "expected_status",
                      "base_case_id", "new_authored_rephrase", "seen_regression"):
            assert token not in wire
        for fact in case["expected"]["required_facts"]:
            assert fact not in wire
    assert not (prepared.root / "authorization.json").exists()
    assert prepared.report()["reserved_requests"] == 0
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
               for path in prepared.root.rglob("*") if path.is_file())


def test_new_initial_wire_is_identical_to_original_actor_request(prepared):
    p = prepared.verify()
    for case, captured in zip(p["cases"], p["initial_intent_requests"]):
        original = asyncio.run(old.base._capture(case, p["base_url"]))
        assert captured["payload"] == original["payload"]
        assert captured["payload_sha256"] == original["payload_sha256"]
        assert captured["request_bytes_sha256"] == original["request_bytes_sha256"]


def test_existing_roots_and_receipts_are_not_overwritten_or_reinitialized(prepared):
    before = snapshot(prepared.root)
    with pytest.raises(old.base.TrialError): asyncio.run(prepared.prepare())
    assert snapshot(prepared.root) == before
    with pytest.raises(old.base.TrialError): module.OrderAnswerRecheck(old.ROOT)
    approve(prepared, 1)
    receipt = (prepared.root / "authorization.json").read_bytes()
    with pytest.raises(FileExistsError): approve(prepared, 2)
    assert (prepared.root / "authorization.json").read_bytes() == receipt


def test_without_new_authorization_no_secret_sdk_store_or_worker_is_opened(prepared, monkeypatch):
    import anthropic
    configuration = old.base._configuration
    def config(*, secret=False):
        assert secret is False, "credentials may not be read before authorization"
        return configuration(secret=False)
    def denied(*args, **kwargs):
        raise AssertionError("approval must precede SDK, database and worker")
    monkeypatch.setattr(old.base, "_configuration", config)
    monkeypatch.setattr(old.task_data, "build_live_data", denied)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", denied)
    monkeypatch.setattr(old.subprocess, "Popen", denied)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    with pytest.raises(FileNotFoundError): asyncio.run(prepared.execute())
    with pytest.raises(FileNotFoundError): prepared.run()
    assert prepared.report()["reserved_requests"] == 0


def test_old_receipt_cannot_be_transplanted_even_when_schema_is_relabelled(configured, prepared, monkeypatch):
    monkeypatch.setattr(old.proof, "source_proof", lambda: {"controlled_old_source": "d" * 64})
    monkeypatch.setattr(old.proof, "artifacts", lambda: {"controlled_old_artifacts": "e" * 64})
    monkeypatch.setattr(old.proof, "prior_proof", lambda: {"controlled_old_history": "f" * 64})
    previous = old.FocusedTrial(configured / "old-controlled-repair")
    asyncio.run(previous.prepare())
    previous.authorize("controlled old allowance, never live", max_requests=24, confirmed=True)
    receipt = bench.read_json(previous.root / "authorization.json")
    before = snapshot(previous.root)
    for new_schema in (False, True):
        value = deepcopy(receipt)
        if new_schema: value["schema_version"] = prepared.authorization_schema
        old.base._write(prepared.root, "authorization.json", value)
        with pytest.raises(old.base.TrialError): prepared._approval(prepared.verify())
        (prepared.root / "authorization.json").unlink()  # Test-only forged receipt.
    assert snapshot(previous.root) == before
    assert prepared.report()["reserved_requests"] == 0


@pytest.mark.parametrize("limit", [0, -1, 25, 138, True, 1.5, None])
def test_invalid_new_authorization_is_not_a_send_allowance(prepared, limit):
    with pytest.raises(old.base.TrialError): approve(prepared, limit)
    assert not (prepared.root / "authorization.json").exists()


@pytest.mark.parametrize("limit", [25, 138, True, 0])
def test_ledger_entry_cannot_bypass_the_new_twenty_four_request_limit(prepared, limit):
    p = prepared.verify(); before = snapshot(prepared.root / "ledger")
    with pytest.raises(old.base.TrialError): prepared.ledger(p, send_limit=limit)
    assert snapshot(prepared.root / "ledger") == before
    assert prepared.ledger(p).status()["attempts"] == 0


@pytest.mark.parametrize("component", ["source", "artifacts", "prior", "dependencies", "data", "environment"])
def test_changed_proof_or_configuration_refuses_before_sending(prepared, monkeypatch, component):
    altered = lambda: {"changed_after_freeze": True}
    if component == "source": monkeypatch.setattr(module.proof, "source_proof", altered)
    elif component == "artifacts": monkeypatch.setattr(module.proof, "artifacts", altered)
    elif component == "prior": monkeypatch.setattr(module.proof, "prior_proof", altered)
    elif component == "dependencies": monkeypatch.setattr(old.engine, "_dependencies", altered)
    elif component == "data": monkeypatch.setattr(old.task_data, "offline_data_proof", altered)
    else: monkeypatch.setattr(old.history, "effective_environment", altered)
    with pytest.raises(old.base.TrialError): prepared.verify()


@pytest.mark.parametrize("change", [
    lambda p: p.update(authorized_model_requests=138),
    lambda p: p["plan"].reverse(),
    lambda p: p["cases"][0]["expected"].update(required_tools=[]),
    lambda p: p["initial_intent_requests"][0]["payload"].update(gold="answer"),
])
def test_protocol_edits_cannot_change_plan_gold_or_create_authorization(prepared, change):
    value = prepared.verify(); change(value)
    (prepared.root / "protocol.json").write_text(json.dumps(value, ensure_ascii=False))
    with pytest.raises((old.base.TrialError, bench.BenchmarkError, TaskBudgetError)): prepared.verify()


def test_lower_authorization_is_atomic_spent_and_cannot_be_topped_up_after_restart(prepared):
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1); calls = []
    async def handle(request): calls.append(request); return response()
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        transport = old.FocusedTransport(BASE, prepared.root, budget, lease, prepared.verify,
            delegate_factory=lambda frame: httpx.MockTransport(handle))
        async def execute():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await send(transport, job="D-p07")
                with pytest.raises(TrialTransportError): await send(transport, job="D-p07")
        asyncio.run(execute())
    reopened = module.OrderAnswerRecheck(prepared.root)
    assert reopened.report()["reserved_requests"] == len(calls) == 1
    assert reopened.report()["paused"] is True
    with pytest.raises(old.base.TrialError):
        asyncio.run(reopened.execute(delegate_factory=lambda frame: None, controlled_data=Data()))
    with pytest.raises(old.base.TrialError): approve(reopened, 24)
    assert reopened.report()["reserved_requests"] == 1


def test_unknown_send_survives_restart_without_a_second_resource_or_send(prepared, monkeypatch):
    import anthropic
    p = prepared.verify(); approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        budget.reserve("D-p07", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    def denied(*args, **kwargs):
        raise AssertionError("unknown request must not open a fresh client or database")
    monkeypatch.setattr(old.task_data, "build_live_data", denied)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", denied)
    reopened = module.OrderAnswerRecheck(prepared.root)
    report = reopened.report()
    assert report["reserved_requests"] == report["pending_physical"] == 1
    assert report["cases"][0]["status"] == "unknown"
    assert report["counts"]["planned"] == 6 and report["counts"]["not_executed"] == 5
    with pytest.raises(TaskBudgetError):
        with reopened.ledger(p, send_limit=1).exclusive_run():
            pytest.fail("an uncertain send cannot be reopened")
    monkeypatch.setattr(module, "ROOT", reopened.root)
    with pytest.raises(TaskBudgetError): asyncio.run(reopened.execute())
    with pytest.raises(old.base.TrialError): approve(reopened, 2)


def test_real_sdk_to_native_handlers_shows_new_answers_but_does_not_auto_grade(prepared, monkeypatch):
    approve(prepared)
    data, observed = Data(), []
    async def handle(request):
        frame = CURRENT.get(); payload = json.loads(request.content)
        observed.append((frame, deepcopy(payload)))
        if frame.phase == "intent":
            intent = "technical" if frame.case_id in {"D-p07", "D-n01"} else "order_status"
            return response(json.dumps({"intent": intent, "confidence": .99, "reasoning": "controlled"}))
        assert frame.phase == "agent", "Judge/background/extra composer requests are not permitted"
        tool = payload["tool_choice"]["name"]
        assert tool in {"read_demo_order", "lookup_error_code"}
        args = ({"error_code": "403"} if tool == "lookup_error_code" else
                {"order_id": "DEMO-1007" if frame.case_id in {"D-p01", "D-n02"} else "DEMO-1002"})
        return response(content=[{"type": "tool_use", "id": "controlled_" + str(len(observed)),
                                  "name": tool, "input": args}])
    report = asyncio.run(prepared.execute(delegate_factory=lambda frame: httpx.MockTransport(handle),
                                          controlled_data=data))
    assert data.closed
    assert report["planned_jobs"] == report["completed_jobs"] == 6
    assert report["reserved_requests"] == len(observed) == 16
    assert report["paused"] is False and report["pending_physical"] == 0
    assert report["counts"]["planned"] == report["counts"]["pending_review"] == 6
    assert report["counts"]["passed"] == report["counts"]["failed"] == 0
    assert report["semantic_reviews_imported"] == 0 and report["quality_pass_claim"] is False
    assert report["review_ready"] is True
    for case in old.cases():
        a = bench.read_json(prepared.root / f"case-D-{case['id']}.json")
        assert a["shared_checks"]["blockers"] == []
        assert a["semantic_review"] is a["judge"] is None
        assert a["evidence_kind"] == "controlled-transport" and a["arm"] == "D"
        assert a["trial_kind"] == module.TRIAL_KIND
        calls = a["turns"][0]["tool_evidence"]["calls"]
        assert set(case["expected"]["required_tools"]) <= {v["name"] for v in calls}
        assert all(v["tool_use_id"] and v["success"] for v in calls)
        if case["base_case_id"] == "p01":
            assert "无法确定这笔订单哪天出库或发货" in a["answer"]
            assert "159.00 元" in a["answer"] and "订单取消状态：未知" in a["answer"]
        if case["base_case_id"] == "p09":
            assert "付款记录不能单独证明订单或全部包裹已经发货" in a["answer"]
            assert "付款记录不能单独证明接口权限已经开通" in a["answer"]
            assert "899.00 元" in a["answer"] and "preorder" in a["answer"]
    for _, payload in observed:
        wire = json.dumps(payload, ensure_ascii=False)
        assert all(word not in wire for word in ("required_facts", "forbidden_claims", "base_case_id", "seen_regression", "new_authored_rephrase"))
    assert all(customer == "demo-customer-a" for customer, _ in data.orders)
    prepared.review(assessment(prepared, "D-p01", coverage=False))
    assert prepared.report()["counts"]["failed"] == 1
    assert prepared.report()["counts"]["pending_review"] == 5
    assert prepared.report()["counts"]["planned"] == 6
    assert prepared.report()["quality_pass_claim"] is False
    # All six have been executed, including a controlled negative review.
    # Running the entry again must not seek a more favorable sample or spawn.
    def denied(*args, **kwargs):
        raise AssertionError("completed fixed jobs must not reopen a worker")
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    monkeypatch.setattr(old.subprocess, "Popen", denied)
    before = snapshot(prepared.root)
    assert prepared.run() == {"exit_code": 0, "already_complete": True}
    assert snapshot(prepared.root) == before


def test_even_a_sealed_case_cannot_lose_the_new_trial_identity(prepared):
    p = prepared.verify(); data = Data()
    old.base._write(prepared.root, "data-proof.json", data.proof)
    data_hash = bench.sha256(prepared.root / "data-proof.json")
    old.base._write(prepared.root, "data-proof-seal.json", {
        "protocol_sha256": bench.sha256(prepared.root / "protocol.json"), "data_proof_sha256": data_hash})
    artifact = {"schema_version": "serveflow-diagnostic-repair-artifact-v1", "case_id": "D-p07",
        "logical_case_id": "D-p07", "source_case_id": "p07", "arm": "D", "turns": [],
        "data_kind": "controlled-fixture", "evidence_kind": "controlled-fixture", "execution_status": "failed",
        "protocol_sha256": bench.sha256(prepared.root / "protocol.json"), "data_proof_sha256": data_hash}
    artifact["shared_checks"] = old.checks(old.cases()[0], artifact)
    old.base._write(prepared.root, "case-D-p07.json", artifact)
    budget = prepared.ledger(p)
    with budget.exclusive_run() as lease:
        budget.start_case("D-p07", lease=lease)
        budget.finish_case("D-p07", bench.sha256(prepared.root / "case-D-p07.json"), outcome="failed", lease=lease)
    with pytest.raises(old.base.TrialError, match="artifact_identity_changed"): prepared.report()


def test_no_completed_rows_are_invented_for_unexecuted_cases(prepared):
    report = prepared.report()
    assert report["planned_jobs"] == len(report["cases"]) == 6
    assert report["counts"] == {"planned": 6, "passed": 0, "failed": 0,
                                "pending_review": 0, "unknown": 0, "not_executed": 6}
    assert report["quality_pass_claim"] is report["review_ready"] is False
    assert report["semantic_reviews_imported"] == 0


def test_wrong_root_or_working_directory_cannot_start_new_worker(prepared, monkeypatch):
    approve(prepared)
    def denied(*args, **kwargs):
        raise AssertionError("wrong context cannot start a worker")
    monkeypatch.setattr(old.subprocess, "Popen", denied)
    with pytest.raises(old.base.TrialError): prepared.run()
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(prepared.root.parent)
    with pytest.raises(old.base.TrialError): prepared.run()


def test_new_fixed_worker_uses_its_own_module_and_900_second_ceiling(prepared, monkeypatch):
    approve(prepared)
    monkeypatch.setattr(module, "ROOT", prepared.root)
    monkeypatch.chdir(bench.BACKEND_ROOT)
    called = []
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    class Process:
        def wait(self, timeout):
            assert timeout == 900
            return 0
    def spawn(command, **kwargs):
        called.append(command)
        assert command == [sys.executable, "-B", "-u", "-m", "evaluation.order_answer_recheck", "_worker"]
        assert kwargs["shell"] is False and kwargs["start_new_session"] is kwargs["close_fds"] is True
        return Process()
    monkeypatch.setattr(old.subprocess, "Popen", spawn)
    assert prepared.run() == {"exit_code": 0, "timed_out": False, "cancelled": False}
    assert len(called) == 1 and prepared.report()["reserved_requests"] == 0
    assert {s: signal.getsignal(s) for s in before} == before


@pytest.mark.parametrize("override", ["--root", "--model", "--url", "--base-url", "--timeout", "--retry"])
def test_cli_has_no_runtime_root_model_endpoint_or_retry_override(monkeypatch, override):
    monkeypatch.setattr(sys, "argv", ["order_answer_recheck", "verify", override, "controlled"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
