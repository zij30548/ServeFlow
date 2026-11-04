"""Offline repaired-protocol boundaries and honest grouped reporting.

All authorization receipts here explicitly describe controlled tests. SDK capture
uses memory transports, send tests use MockTransport, and sockets are denied.
"""
import asyncio
from copy import deepcopy
import json
import signal
import stat

import httpx
import pytest

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench
from evaluation import prospective_repaired_trial as module
from evaluation import prospective_trial as original
from evaluation.intent_transport import TrialTransportError
from evaluation.task_budget import TaskBudgetError
from evaluation import task_supervisor
from test_prospective_trial import BASE, Data, response, send


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError("repaired protocol tests cannot contact any service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module.proof, "repaired_design_proof", lambda: {"controlled_repair_identity": "same"})
    monkeypatch.setattr(module.proof, "artifacts", lambda: {"controlled_candidate": "a" * 64})
    monkeypatch.setattr(module.proof, "prior_ledgers", lambda: {"old_bytes": "b" * 64})
    monkeypatch.setattr(original, "design_proof", lambda: {"controlled_original_identity": "same"})
    monkeypatch.setattr(original, "artifacts", lambda: {"controlled_original_source": "c" * 64})
    monkeypatch.setattr(original, "prior_ledgers", lambda: {"old_bytes": "b" * 64})
    monkeypatch.setattr(original.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(original.task_data, "offline_data_proof", lambda: {"controlled_local_bytes": True})
    def configuration(*, secret=False):
        return {"base_url": BASE, "model": original.base.MODEL,
                **({"api_key": "controlled-only"} if secret else {})}
    monkeypatch.setattr(original.base, "_configuration", configuration)
    return tmp_path


@pytest.fixture
def prepared(configured):
    trial = module.RepairedProspectiveTrial(configured / "repaired-execution")
    asyncio.run(trial.prepare())
    return trial


def approve(trial, limit=20):
    return trial.authorize("controlled test receipt only; no live allowance",
                           max_requests=limit, confirmed=True)


def report_rows(status="not_executed"):
    return {"cases": [{"logical_case_id": row["case_id"], "status": status}
                      for row in original.fixed_plan()]}


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_real_sdk_capture_keeps_original_payloads_questions_and_zero_authorization(prepared):
    p = prepared.verify()
    assert p["schema_version"] == module.RepairedProspectiveTrial.protocol_schema
    assert p["authorized_model_requests"] == p["judge_requests"] == p["external_business_requests"] == 0
    assert p["proposed_model_requests"] == 576
    assert p["plan"] == original.fixed_plan() and len(p["plan"]) == 36
    assert len(p["initial_intent_requests"]) == 24 and len(p["initial_rewrite_requests"]) == 12
    assert p["evaluation_groups"] == module.groups()
    assert p["retrieval_group"]["case_ids"] == [f"q{i:02d}" for i in range(1, 13)]
    for left, right in zip(p["initial_intent_requests"][::2], p["initial_intent_requests"][1::2]):
        assert left["source_case_id"] == right["source_case_id"]
        assert left["payload"] == right["payload"]
        assert left["payload_sha256"] == right["payload_sha256"]
        assert left["request_bytes_sha256"] == right["request_bytes_sha256"]
    for item in p["initial_intent_requests"] + p["initial_rewrite_requests"]:
        payload = json.dumps(item["payload"], ensure_ascii=False)
        for banned in ("new_author_visible_tasks", "seen_scope_regression", "required_facts",
                       "forbidden_claims", "related_prior_cases", "expected_status"):
            assert banned not in payload
        source = next(c for c in p["cases"] if c["case_id"] == item["source_case_id"])
        assert source["conversation_input"]["message"] in payload
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((prepared.root / "protocol.json").stat().st_mode) == 0o600
    assert prepared.ledger(p).status()["attempts"] == 0
    assert not (prepared.root / "authorization.json").exists()


def test_group_membership_is_fixed_disjoint_complete_and_does_not_change_whole_plan(prepared):
    report = prepared.report()
    groups = report["task_groups"]
    assert groups["seen_scope_regression"]["case_ids"] == ["p12"]
    assert groups["new_author_visible_tasks"]["case_ids"] == [f"p{i:02d}" for i in range(1, 12)]
    first, second = (set(value["case_ids"]) for value in groups.values())
    assert not first & second
    assert first | second == {f"p{i:02d}" for i in range(1, 13)}
    assert report["planned_jobs"] == len(report["cases"]) == 36
    assert report["completed_jobs"] == report["reserved_requests"] == 0
    for arm in ("A", "D"):
        whole = report["task_metrics"][arm]
        assert {name: value["denominator"] for name, value in whole.items()} == {
            "correct_handling": 12, "supported_completion": 10, "tool_tasks": 11, "boundary_handling": 2}
        fresh = groups["new_author_visible_tasks"]["task_metrics"][arm]
        assert {name: fresh[name]["denominator"] for name in whole} == {
            "correct_handling": 11, "supported_completion": 10, "tool_tasks": 10, "boundary_handling": 1}
        seen = groups["seen_scope_regression"]["task_metrics"][arm]
        assert {name: seen[name]["denominator"] for name in whole} == {
            "correct_handling": 1, "supported_completion": 0, "tool_tasks": 1, "boundary_handling": 1}
        assert seen["supported_completion"]["applicable"] is False
        assert seen["supported_completion"]["numerator"] == 0
        assert sum(seen["supported_completion"]["counts"].values()) == 0
    assert report["comparison_ready"] is report["quality_pass_claim"] is False
    assert report["task_delta_percentage_points"] is None


@pytest.mark.parametrize("status", module.STATES)
def test_every_status_stays_in_both_groups_with_unchanged_denominators(status):
    source = report_rows(status)
    original_rows = deepcopy(source)
    groups = module.grouped_tasks(source)
    assert source == original_rows
    for name, count in (("new_author_visible_tasks", 11), ("seen_scope_regression", 1)):
        group = groups[name]
        for arm in ("A", "D"):
            metric = group["task_metrics"][arm]["correct_handling"]
            assert metric["denominator"] == count
            assert metric["counts"][status] == count and sum(metric["counts"].values()) == count
            assert metric["numerator"] == (count if status == "passed" else 0)
        if status in {"pending_review", "unknown", "not_executed"}:
            assert group["delta_percentage_points"] is None
            assert group["pair_counts"]["unpaired_pending_unknown_or_unexecuted"] == count
        else:
            assert group["delta_percentage_points"] == 0
            assert group["pair_counts"]["both_passed" if status == "passed" else "both_failed"] == count


def test_failed_seen_regression_cannot_be_hidden_by_favorable_new_task_subgroup():
    report = report_rows("passed")
    by_id = {r["logical_case_id"]: r for r in report["cases"]}
    by_id["A-p03"]["status"] = "failed"  # New group favors D by one.
    by_id["D-p12"]["status"] = "failed"  # Seen regression favors A by one.
    groups = module.grouped_tasks(report)
    fresh, seen = groups["new_author_visible_tasks"], groups["seen_scope_regression"]
    assert fresh["pair_counts"]["only_D_passed"] == 1
    assert fresh["delta_percentage_points"] == pytest.approx(100 / 11)
    assert seen["pair_counts"]["only_A_passed"] == 1
    assert seen["delta_percentage_points"] == -100
    for arm in ("A", "D"):
        total = sum(g["task_metrics"][arm]["correct_handling"]["numerator"] for g in groups.values())
        assert total == 11
        assert sum(g["task_metrics"][arm]["correct_handling"]["denominator"] for g in groups.values()) == 12


def test_group_tool_metric_and_dimensions_do_not_override_primary_failures():
    report = report_rows("failed")
    row = next(r for r in report["cases"] if r["logical_case_id"] == "D-p12")
    row["tool_status"] = "passed"
    row["review"] = {"dimensions": dict.fromkeys(original.design.DIMENSIONS, True)}
    seen = module.grouped_tasks(report)["seen_scope_regression"]["task_metrics"]["D"]
    assert seen["correct_handling"]["numerator"] == 0
    assert seen["tool_tasks"]["numerator"] == 1
    assert all(v == {"passed_reviewed": 1, "reviewed": 1, "planned": 1}
               for v in seen["dimensions"].values())


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign"])
def test_groups_reject_missing_duplicate_or_foreign_task_rows(damage):
    report = report_rows()
    if damage == "missing": report["cases"].pop(0)
    elif damage == "duplicate": report["cases"][0] = deepcopy(report["cases"][1])
    else: report["cases"][0]["logical_case_id"] = "D-p13"
    with pytest.raises(Exception, match="repaired_group_plan_changed"):
        module.grouped_tasks(report)


def test_rag_rows_never_become_extra_task_passes_or_regression_members():
    report = report_rows()
    for row in report["cases"]:
        if row["logical_case_id"].startswith("R-"):
            row["status"] = "passed"
    groups = module.grouped_tasks(report)
    assert sum(len(g["pairs"]) for g in groups.values()) == 12
    assert all(g["task_metrics"][arm]["correct_handling"]["numerator"] == 0
               for g in groups.values() for arm in ("A", "D"))


@pytest.mark.parametrize("mutation", [
    lambda p: p["evaluation_groups"]["seen_scope_regression"]["case_ids"].clear(),
    lambda p: p["evaluation_groups"]["new_author_visible_tasks"]["case_ids"].append("p12"),
    lambda p: p["retrieval_group"]["case_ids"].pop(),
    lambda p: p["plan"].pop(),
    lambda p: p["initial_intent_requests"][0]["payload"].update(expected="secret gold"),
])
def test_protocol_cannot_reclassify_exclude_or_add_gold_after_freeze(prepared, mutation):
    value = prepared.verify()
    mutation(value)
    (prepared.root / "protocol.json").write_text(json.dumps(value))
    with pytest.raises(Exception):
        prepared.verify()


def test_no_authorization_opens_no_worker_sdk_data_or_credentials(prepared, monkeypatch):
    monkeypatch.setattr(module.RepairedProspectiveTrial, "root_path", prepared.root)
    def forbidden(*args, **kwargs):
        raise AssertionError("no resources may be opened without this protocol's allowance")
    import anthropic
    configuration = original.base._configuration
    def no_secret(*, secret=False):
        assert secret is False, "credentials cannot be requested without authorization"
        return configuration(secret=False)
    monkeypatch.setattr(original.base, "_configuration", no_secret)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", forbidden)
    monkeypatch.setattr(original.task_data, "build_live_data", forbidden)
    monkeypatch.setattr(original.subprocess, "Popen", forbidden)
    with pytest.raises(FileNotFoundError):
        asyncio.run(prepared.execute())
    with pytest.raises(FileNotFoundError):
        original.supervise(prepared)
    assert prepared.report()["reserved_requests"] == 0


def test_old_authorization_cannot_be_transplanted_even_with_new_schema(configured, prepared):
    old = original.ProspectiveTrial(configured / "old-original-execution")
    asyncio.run(old.prepare())
    old.authorize("controlled original allowance", max_requests=2, confirmed=True)
    raw = (old.root / "authorization.json").read_bytes()
    old_snapshot = snapshot(old.root)
    for rewrite_schema in (False, True):
        receipt = json.loads(raw)
        if rewrite_schema: receipt["schema_version"] = prepared.authorization_schema
        path = prepared.root / "authorization.json"
        original.base._write(prepared.root, path.name, receipt)
        with pytest.raises(Exception, match="prospective_approval_changed"):
            prepared._approval(prepared.verify())
        path.unlink()  # This is a temporary test-only forged receipt, not a live ledger.
    assert snapshot(old.root) == old_snapshot
    assert prepared.report()["reserved_requests"] == 0


def test_existing_or_original_root_cannot_be_reinitialized_or_overwritten(configured, prepared):
    before = snapshot(prepared.root)
    with pytest.raises(Exception, match="execution_root_exists"):
        asyncio.run(prepared.prepare())
    assert snapshot(prepared.root) == before
    with pytest.raises(Exception, match="fixed_root_required"):
        module.RepairedProspectiveTrial(original.ROOT)
    assert module.RepairedProspectiveTrial().root == module.proof.ROOT
    assert original.ProspectiveTrial().root == original.ROOT
    assert module.proof.ROOT != original.ROOT


def test_smaller_allowance_is_shared_counted_and_does_not_reset_on_reopen(prepared):
    p = prepared.verify()
    approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    sent = []
    async def handler(request):
        sent.append(request)
        return response()
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01", lease=lease)
        transport = original.ProspectiveTransport(BASE, prepared.root, budget, lease, prepared.verify,
            delegate_factory=lambda frame: httpx.MockTransport(handler))
        async def run():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await send(transport)
                with pytest.raises(TrialTransportError): await send(transport)
        asyncio.run(run())
    assert len(sent) == 1
    reopened = module.RepairedProspectiveTrial(prepared.root)
    assert reopened.verify()["limits"]["physical_requests"] == 576
    assert reopened.report()["reserved_requests"] == 1
    assert reopened.report()["paused"] is True
    with pytest.raises(Exception, match="prospective_paused"):
        asyncio.run(reopened.execute(delegate_factory=lambda _: None, controlled_data=Data()))
    assert len(sent) == 1 and reopened.ledger(p).status()["attempts"] == 1


def test_unfinished_attempt_cannot_be_replayed_reauthorized_or_refunded(prepared):
    p = prepared.verify()
    approve(prepared, 1)
    budget = prepared.ledger(p, send_limit=1)
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01", lease=lease)
        budget.reserve("A-p01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    state = prepared.report()
    assert state["reserved_requests"] == state["pending_physical"] == 1
    assert state["cases"][0]["status"] == "unknown"
    assert state["comparison_ready"] is False
    reopened = module.RepairedProspectiveTrial(prepared.root)
    with pytest.raises(TaskBudgetError):
        with reopened.ledger(p, send_limit=1).exclusive_run():
            raise AssertionError("unfinished lease must not reopen")
    with pytest.raises(Exception, match="prospective_already_started"):
        approve(reopened, 2)
    assert reopened.report()["reserved_requests"] == reopened.report()["pending_physical"] == 1


def test_repaired_supervisor_spawns_only_its_fixed_module_and_restores_handlers(prepared, monkeypatch):
    approve(prepared)
    monkeypatch.setattr(module.RepairedProspectiveTrial, "root_path", prepared.root)
    seen = []
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    class Process:
        def wait(self, timeout):
            assert timeout == 3600
            return 0
    def spawn(command, **kwargs):
        seen.append(command)
        assert command == [original.sys.executable, "-B", "-u", "-m", "evaluation.prospective_repaired_trial", "_worker"]
        assert kwargs["cwd"] == bench.BACKEND_ROOT and kwargs["shell"] is False
        assert kwargs["start_new_session"] is kwargs["close_fds"] is True
        return Process()
    monkeypatch.setattr(original.subprocess, "Popen", spawn)
    assert original.supervise(prepared) == 0
    assert len(seen) == 1
    assert {number: signal.getsignal(number) for number in before} == before


def test_original_runner_keeps_schema_default_root_and_empty_extensions(configured):
    old = original.ProspectiveTrial(configured / "original-check")
    p = asyncio.run(old.prepare())
    assert p["schema_version"] == "serveflow-prospective-execution-protocol-v1"
    assert old.authorization_schema == "serveflow-prospective-authorization-v1"
    assert old.progress_schema == "serveflow-prospective-progress-v1"
    assert old.protocol_extension() == {}
    assert "evaluation_groups" not in p and "retrieval_group" not in p
    assert old.report()["schema_version"] == "serveflow-prospective-progress-v1"
    assert original.ProspectiveTrial().root == original.ROOT
    assert not (old.root / "authorization.json").exists()


def test_temporary_test_root_cannot_spawn_a_real_default_root_worker(prepared, monkeypatch):
    approve(prepared)
    def forbidden(*args, **kwargs):
        raise AssertionError("a temporary root cannot spawn a real worker")
    monkeypatch.setattr(original.subprocess, "Popen", forbidden)
    with pytest.raises(task_supervisor.SupervisorError, match="worker_root_invalid"):
        original.supervise(prepared)
    assert prepared.report()["reserved_requests"] == 0


def test_wrapped_report_keeps_unfavorable_regression_and_whole_plan_conclusion(monkeypatch):
    source = report_rows("passed")
    by_id = {r["logical_case_id"]: r for r in source["cases"]}
    by_id["A-p03"]["status"] = "failed"
    by_id["D-p12"]["status"] = "failed"
    source.update({"task_delta_percentage_points": 0.0,
        "task_metrics": {arm: {"correct_handling": {"numerator": 11, "denominator": 12}}
                         for arm in ("A", "D")},
        "planned_jobs": 36, "comparison_ready": False, "quality_pass_claim": False})
    monkeypatch.setattr(original.ProspectiveTrial, "report", lambda self: deepcopy(source))
    actual = module.RepairedProspectiveTrial().report()
    for field in source:
        assert actual[field] == source[field]
    assert actual["task_delta_percentage_points"] == 0
    assert actual["task_groups"]["new_author_visible_tasks"]["delta_percentage_points"] > 0
    assert actual["task_groups"]["seen_scope_regression"]["delta_percentage_points"] < 0
    assert actual["comparison_ready"] is actual["quality_pass_claim"] is False


def test_source_question_and_expected_bytes_stay_identical_during_new_prepare(configured):
    files = [bench.BACKEND_ROOT / f"evaluation/benchmark/prospective-{suite}-v1.json"
             for suite in ("tasks", "retrieval")]
    before = {path: path.read_bytes() for path in files}
    trial = module.RepairedProspectiveTrial(configured / "question-preservation")
    asyncio.run(trial.prepare())
    assert {path: path.read_bytes() for path in files} == before
    protocol = trial.verify()
    assert protocol["cases"] == original.design._plan(bench.BACKEND_ROOT)
    assert not (trial.root / "authorization.json").exists()
