"""Temporary-root integration of native SDK, durable counting and repair jobs.

All responses use memory transports and fixed synthetic records. These tests
never authorize or execute the real root, inspect secrets, or open a database.
Semantic review fixtures validate the report contract, not answer quality.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
import stat
import sys

import httpx
import pytest

from evaluation import benchmark as bench, repair_execution_trial as module
from evaluation import repair_execution_runtime as runtime
from evaluation import task_supervisor as supervisor
from evaluation.phase_deadline_transport import PHASE_DEADLINES
from evaluation.task_transport import CURRENT
from test_repair_execution_runtime import Data as RuntimeData, SDK, execute as runtime_execute


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repair trial tests cannot open services or credentials")
    for key in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, key, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    for key in tuple(os.environ):
        if key.startswith(("ANTHROPIC_", "ECHOMIND_", "SILICONFLOW_", "EVAL_JUDGE_", "SERVEFLOW_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(module.task_data, "build_live_data", denied)
    monkeypatch.setattr(module.engine, "_mysql_url", denied)


class Data(RuntimeData):
    async def close(self):
        self.closed = True


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Only proof metadata is stubbed. Questions, first SDK payloads, production
    # Orchestrator, native tool handlers and request admission remain actual.
    source = {"controlled-source": "a" * 64}
    dependencies, environment, data_proof = {"controlled": "1"}, {"controlled": True}, {"controlled": True}
    config = {"base_url": SDK.base_url, "model": module.base.MODEL}

    def configuration(*, secret=False):
        assert secret is False, "controlled run must never request credentials"
        return deepcopy(config)

    monkeypatch.setattr(module.base, "_configuration", configuration)
    monkeypatch.setattr(module, "source_proof", lambda: {"controlled": "b" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"old_records": "c" * 64, "old_allowance_reused": False})
    monkeypatch.setattr(module.ancestry, "_current_sources", lambda: deepcopy(source))
    monkeypatch.setattr(module.engine, "_dependencies", lambda: deepcopy(dependencies))
    monkeypatch.setattr(module.history, "effective_environment", lambda: deepcopy(environment))
    monkeypatch.setattr(module.task_data, "offline_data_proof", lambda: deepcopy(data_proof))
    captures = asyncio.run(module.preparation.capture_requests(config))
    preparation = {**config, "effective_environment": environment, "dependencies": dependencies,
        "offline_data_proof": data_proof, "rubric": deepcopy(module.preparation.RUBRIC),
        "sdk_first_captures": captures}
    monkeypatch.setattr(module, "frozen_preparation", lambda: deepcopy(preparation))
    trial = module.RepairExecutionTrial(tmp_path / "controlled-repair")
    assert trial.root != module.ROOT
    asyncio.run(trial.prepare())
    return trial


def approve(trial, limit=96):
    assert trial.root != module.ROOT
    return trial.authorize("Controlled test only; not a real model authorization",
                           max_requests=limit, confirmed=True)


def review(trial, identity, **changes):
    value = {"logical_case_id": identity,
        "artifact_sha256": bench.sha256(trial.root / f"case-{identity}.json"),
        "reviewer": "controlled schema reviewer", "method": "controlled-fixture",
        "dimensions": dict.fromkeys(module.preparation.RUBRIC["tasks"], True),
        "blocking_errors": [], "reading_burden": "none", "reason": "Schema test, not semantic quality evidence."}
    value.update(changes)
    return trial.review(value)


def controlled_run(trial, *, failure=None, refused_case=None, malformed_rerank_case=None):
    assert trial.root != module.ROOT
    data, observed = Data(), []
    oracles = {row["id"]: SDK(row, refuse=row["id"] == refused_case) for row in runtime.rows()}

    async def handle(request):
        frame = CURRENT.get()
        payload = json.loads(request.content)
        observed.append((frame, bytes(request.content), deepcopy(request.extensions)))
        if failure is not None:
            raise failure
        result = oracles[frame.case_id].response(**payload)
        if frame.case_id == malformed_rerank_case and frame.phase == "rag_rerank":
            result = {"content": [{"type": "text", "text": "controlled non-JSON support decision"}],
                      "stop_reason": "end_turn"}
        return httpx.Response(200, json={"id": "controlled-repair-native", "type": "message", "role": "assistant",
            "model": module.base.MODEL, **result, "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 2}})

    coro = trial.execute(delegate_factory=lambda frame: httpx.MockTransport(handle), controlled_data=data)
    return coro, data, observed


def test_prepare_is_zero_authorization_with_exact_sixteen_requests_and_private_files(prepared):
    protocol, report = prepared.verify(), prepared.report()
    assert protocol["authorized_model_requests"] == report["reserved_requests"] == report["authorized_requests"] == 0
    assert protocol["judge_requests"] == protocol["external_business_requests"] == 0
    assert len(protocol["cases"]) == len(protocol["jobs"]) == 16
    assert [row["id"] for row in protocol["cases"]] == list(runtime.TASK_IDS + runtime.RETRIEVAL_IDS)
    assert len(protocol["initial_intent_requests"]) == len(protocol["initial_rewrite_requests"]) == 8
    assert not (prepared.root / "authorization.json").exists()
    assert not (prepared.root / "data-proof.json").exists()
    assert report["quality_pass_claim"] is report["review_ready"] is report["real_comparison_ready"] is False
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((prepared.root / "protocol.json").stat().st_mode) == 0o600
    for row in protocol["initial_intent_requests"] + protocol["initial_rewrite_requests"]:
        encoded = json.dumps(row["payload"], ensure_ascii=False)
        assert all(term not in encoded for term in (row["case_id"], "required_facts", "relevant_titles", "demo-customer-a"))


def test_preparation_and_approval_cannot_overwrite_and_no_grant_opens_no_resources(prepared):
    protocol = (prepared.root / "protocol.json").read_bytes()
    with pytest.raises(Exception, match="root_exists"):
        asyncio.run(prepared.prepare())
    assert (prepared.root / "protocol.json").read_bytes() == protocol
    with pytest.raises(FileNotFoundError):
        asyncio.run(prepared.execute())
    with pytest.raises(Exception, match="explicit_authorization_required"):
        prepared.authorize("not confirmed", max_requests=96)
    approve(prepared, 1)
    raw = (prepared.root / "authorization.json").read_bytes()
    with pytest.raises(FileExistsError):
        approve(prepared, 96)
    assert (prepared.root / "authorization.json").read_bytes() == raw
    assert prepared.ledger(prepared.verify()).status()["attempts"] == 0


@pytest.mark.parametrize("pair", ["only_data", "only_delegate", "neither"])
def test_temp_root_requires_both_controlled_transport_and_controlled_data(prepared, pair):
    approve(prepared)
    kwargs = ({"controlled_data": Data()} if pair == "only_data" else
              {"delegate_factory": lambda frame: (_ for _ in ()).throw(AssertionError("no delegate"))}
              if pair == "only_delegate" else {})
    with pytest.raises(Exception, match="controlled_pair_required|live_controlled_root_mismatch"):
        asyncio.run(prepared.execute(**kwargs))
    assert prepared.ledger(prepared.verify()).status()["attempts"] == 0
    assert not (prepared.root / "data-proof.json").exists()


def test_all_sixteen_native_sdk_jobs_are_counted_bound_and_reported_without_quality_claim(prepared):
    approve(prepared)
    coro, data, calls = controlled_run(prepared)
    report = asyncio.run(coro)
    assert report == prepared.report()
    assert report["completed_jobs"] == 16
    assert report["reserved_requests"] == report["sealed_physical"] == len(calls) < 96
    assert report["pending_physical"] == 0 and not report["paused"]
    assert report["execution_complete"] and report["authorized_requests"] == 96
    assert report["artifact_evidence_kinds"] == {"controlled-transport": 16}
    assert report["semantic_task_reviews"] == 0
    assert report["review_ready"] is True and report["comparison_ready"] is report["real_comparison_ready"] is False
    assert report["quality_pass_claim"] is False and data.closed
    assert {frame.case_id for frame, _, _ in calls} == set(runtime.TASK_IDS + runtime.RETRIEVAL_IDS)
    assert not any(frame.phase in {"judge", "embedding", "memory"} for frame, _, _ in calls)
    assert all(row["status"] == "pending_review" for row in report["task_rows"])
    assert len(report["retrieval_rows"]) == 8
    assert all(row["status"] == "recorded" and row["physical_structure_valid"]
               and row["physical_outcomes"] == ["received", "received"] and row["rerank_applied"]
               for row in report["retrieval_rows"])
    assert all(group["correct_handling"] == {"numerator": 0, "denominator": 4}
               for group in report["task_groups"].values())
    assert report["physical_resources"]["usage"]["physical_requests"]["known_sum"] == len(calls)
    protocol = prepared.verify()
    for name in ("initial_intent_requests", "initial_rewrite_requests"):
        for frozen in protocol[name]:
            first = next(raw for frame, raw, _ in calls if frame.case_id == frozen["case_id"])
            assert hashlib.sha256(first).hexdigest() == frozen["request_bytes_sha256"]
    for frame, raw, extensions in calls:
        assert extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"),
                                                      float(PHASE_DEADLINES[frame.phase]))
        stored = bench.read_json(prepared.root / f"request-{hashlib.sha256(raw).hexdigest()}.json")
        assert stored["wire_text"].encode() == raw
        assert stored["payload_sha256"] == module.canonical_payload_hash(json.loads(raw))
        assert frame.case_id not in raw.decode() and "required_facts" not in raw.decode()
    for identity in runtime.TASK_IDS:
        artifact = bench.read_json(prepared.root / f"case-{identity}.json")
        assert artifact["shared_checks"] == runtime.checks(next(r for r in runtime.rows() if r["id"] == identity), artifact)
        assert artifact["semantic_review"] is None and artifact["judge"] is None
        for native in artifact["turns"][0]["tool_evidence"]["calls"]:
            assert native["tool_use_id"] and native["request_id"] == f"repair-task-{identity}-1"
            assert native["trusted_customer_id"] == "demo-customer-a"
    before = len(calls)
    again = asyncio.run(prepared.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no resend")),
                                        controlled_data=Data()))
    assert again == report and len(calls) == before
    for identity in runtime.TASK_IDS:
        review(prepared, identity)
    reviewed = prepared.report()
    assert reviewed["semantic_task_reviews"] == 8 and reviewed["comparison_ready"] is True
    assert reviewed["real_comparison_ready"] is reviewed["quality_pass_claim"] is False
    path = prepared.root / "review-RT01.json"
    changed = bench.read_json(path); changed["dimensions"] = {}
    path.write_text(json.dumps(changed))
    with pytest.raises(Exception, match="review_dimensions"):
        prepared.report()


def test_missing_native_execution_remains_failed_and_fixed_plan_continues(prepared):
    approve(prepared)
    coro, data, calls = controlled_run(prepared, refused_case="RT02")
    report = asyncio.run(coro)
    row = next(value for value in report["task_rows"] if value["id"] == "RT02")
    assert report["completed_jobs"] == 16 and not report["paused"] and data.closed
    assert row["status"] == row["tool_status"] == "failed"
    assert row["checks"]["checks"]["required_tools"] is False
    assert any(frame.case_id == "RR08" for frame, _, _ in calls)
    assert report["quality_pass_claim"] is False


def test_malformed_support_decision_retains_vector_ranking_but_is_not_review_ready(prepared):
    approve(prepared)
    coro, data, calls = controlled_run(prepared, malformed_rerank_case="RR01")
    report = asyncio.run(coro)
    assert report["execution_complete"] and report["completed_jobs"] == 16 and not report["paused"]
    row = next(value for value in report["retrieval_rows"] if value["id"] == "RR01")
    artifact = bench.read_json(prepared.root / "case-RR01.json")
    assert artifact["baseline"]["record"]["ranked_titles"]
    assert artifact["enhanced"]["record"]["status"] == "failed"
    assert row["physical_structure_valid"] and row["physical_outcomes"] == ["received", "received"]
    assert row["rerank_applied"] is False and row["status"] == "failed"
    assert report["review_ready"] is report["comparison_ready"] is report["quality_pass_claim"] is False
    assert report["reserved_requests"] == len(calls) and data.closed


@pytest.mark.parametrize("damage", ["identity", "source", "record", "ranking"])
def test_rag_dispatcher_independently_rejects_wrong_identity_source_or_score(damage):
    artifact, _, _ = asyncio.run(runtime_execute("RR01"))
    row = next(value for value in runtime.rows() if value["id"] == "RR01")
    original = module.artifact_checks(row, artifact)
    assert all(original["checks"].values()) and not original["blockers"]
    if damage == "identity":
        artifact["logical_case_id"] = "RR02"
    elif damage == "source":
        artifact["enhanced"]["sources"][0]["content"] += " unsupported source"
    elif damage == "record":
        artifact["enhanced"]["record"]["ranked_titles"] = []
    else:
        artifact["ranking_observations"]["enhanced"]["hit_at_3"] = "not a boolean score"
    observed = module.artifact_checks(row, artifact)
    assert not all(observed["checks"].values()) and observed["blockers"]
    assert observed["semantic_review_required"] is True and observed["quality_pass_claim"] is False


def test_unknown_send_keeps_its_claim_and_reopening_cannot_resume_or_refund(prepared):
    approve(prepared)
    protocol = prepared.verify()
    ledger = prepared.ledger(protocol)
    with ledger.exclusive_run() as lease:
        ledger.start_case("RT01", lease=lease)
        ledger.reserve("RT01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    prefix = (prepared.root / "ledger/journal.jsonl").read_bytes()
    reopened = module.RepairExecutionTrial(prepared.root)
    with pytest.raises(Exception, match="previous_run_requires_reconciliation"):
        asyncio.run(reopened.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no resend")),
                                    controlled_data=Data()))
    state = reopened.ledger(protocol).status()
    assert state["attempts"] == 1 and len(state["pending"]) == 1 and state["active_case"] == "RT01"
    assert (prepared.root / "ledger/journal.jsonl").read_bytes() == prefix
    with pytest.raises(Exception, match="already_started"):
        approve(prepared)


def test_transport_failure_is_counted_and_paused_without_automatic_retry(prepared):
    approve(prepared)
    coro, data, calls = controlled_run(prepared, failure=httpx.ReadTimeout("controlled timeout"))
    with pytest.raises(Exception):
        asyncio.run(coro)
    report = prepared.report()
    assert len(calls) == report["reserved_requests"] == 1
    assert report["paused"] and report["completed_jobs"] == 0 and data.closed
    assert report["quality_pass_claim"] is report["comparison_ready"] is False
    with pytest.raises(Exception, match="paused"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no retry")),
                                    controlled_data=Data()))


@pytest.mark.parametrize("damage", ["approval", "source", "data"])
def test_runtime_context_changes_stop_before_a_physical_request(prepared, monkeypatch, damage):
    approve(prepared)
    async def changed(row, sdk, data, *, ensure_healthy, frame_case_id):
        if damage == "approval":
            path = prepared.root / "authorization.json"
            value = bench.read_json(path); value["reference"] = "changed"
            path.write_text(json.dumps(value))
        elif damage == "source":
            monkeypatch.setattr(module.ancestry, "_current_sources", lambda: {"changed": "d" * 64})
        else:
            data.proof["changed"] = True
        ensure_healthy()
        raise AssertionError("mutated context cannot execute")
    monkeypatch.setattr(module, "execute_task", changed)
    with pytest.raises(Exception, match="runtime_context_changed"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                    controlled_data=Data()))
    assert prepared.ledger(bench.read_json(prepared.root / "protocol.json")).status()["attempts"] == 0
    assert (prepared.root / "worker-stop.json").exists()


def test_observed_transient_context_failure_stays_latched_after_restoration(prepared, monkeypatch):
    approve(prepared)
    previous = module.ancestry._current_sources
    async def transient(row, sdk, data, *, ensure_healthy, frame_case_id):
        monkeypatch.setattr(module.ancestry, "_current_sources", lambda: {"temporary": "d" * 64})
        with pytest.raises(Exception, match="runtime_context_changed"):
            ensure_healthy()
        monkeypatch.setattr(module.ancestry, "_current_sources", previous)
        ensure_healthy()
    monkeypatch.setattr(module, "execute_task", transient)
    with pytest.raises(Exception, match="context_previously_failed"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                    controlled_data=Data()))
    assert prepared.ledger(bench.read_json(prepared.root / "protocol.json")).status()["attempts"] == 0


def test_initial_payload_mutation_rejected_before_any_resource_or_allowance(prepared):
    approve(prepared)
    path = prepared.root / "protocol.json"
    value = bench.read_json(path)
    value["initial_rewrite_requests"][0]["payload"]["messages"][0]["content"] += " changed"
    path.write_text(json.dumps(value))
    with pytest.raises(Exception, match="freeze_changed"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame: (_ for _ in ()).throw(AssertionError("no send")),
                                    controlled_data=Data()))
    assert not (prepared.root / "data-proof.json").exists()


def worker_command():
    return [sys.executable, "-B", "-u", "-m", "evaluation.repair_execution_trial", "_worker"]


@pytest.mark.parametrize("deadline", [900, .1])
def test_supervisor_new_worker_accepts_only_its_exact_root_and_at_most_900_seconds(tmp_path, monkeypatch, deadline):
    # _open_root is deliberately replaced: the actual fixed root is never
    # opened or written, and Popen never executes a worker in this test.
    opened, launches, waits = [], [], []
    def open_controlled(root):
        opened.append(root)
        return os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    class Process:
        def wait(self, *, timeout):
            waits.append(timeout)
            return 0
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return Process()
    monkeypatch.setattr(supervisor, "_open_root", open_controlled)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    result = supervisor.supervise(worker_command(), root=module.ROOT, deadline_s=deadline)
    assert result == {"exit_code": 0, "timed_out": False, "cancelled": False}
    assert opened == [module.ROOT] and waits == [float(deadline)]
    assert launches == [(worker_command(), {"shell": False, "start_new_session": True,
        "stdout": supervisor.subprocess.DEVNULL, "stderr": supervisor.subprocess.DEVNULL, "close_fds": True})]


@pytest.mark.parametrize("deadline", [901, 3600, 0, -1, float("nan"), float("inf"), True])
def test_supervisor_repair_worker_cannot_extend_the_deadline_or_open_before_validation(monkeypatch, deadline):
    def denied(*args, **kwargs):
        raise AssertionError("invalid deadline must not open a root or create a process")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_deadline_invalid"):
        supervisor.supervise(worker_command(), root=module.ROOT, deadline_s=deadline)


def test_supervisor_repair_worker_rejects_other_root_and_altered_command(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("invalid worker must not open a root or create a process")
    monkeypatch.setattr(supervisor, "_open_root", denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", denied)
    with pytest.raises(supervisor.SupervisorError, match="repair_execution_worker_root_invalid"):
        supervisor.supervise(worker_command(), root=tmp_path, deadline_s=900)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_fixed_worker_required"):
        supervisor.supervise(worker_command() + ["--resume"], root=module.ROOT, deadline_s=900)


@pytest.mark.parametrize("evidence_kind,method,valid", [
    ("controlled-transport", "controlled-fixture", True),
    ("controlled-transport", "executor-rubric-review", False),
    ("live-model", "controlled-fixture", False),
    ("live-model", "executor-rubric-review", True),
])
def test_review_method_must_match_evidence_kind_without_sealing_a_live_fixture(tmp_path, evidence_kind, method, valid):
    # This is only the pure review contract: no protocol, authorization, ledger,
    # sender or sealed live artifact is created or claimed by this fixture.
    trial = module.RepairExecutionTrial(tmp_path / "review-contract-only")
    trial.root.mkdir(mode=0o700)
    module.base._write(trial.root, "case-RT01.json", {"evidence_kind": evidence_kind})
    value = {"logical_case_id": "RT01", "artifact_sha256": bench.sha256(trial.root / "case-RT01.json"),
        "reviewer": "controlled contract checker", "method": method,
        "dimensions": dict.fromkeys(module.preparation.RUBRIC["tasks"], True),
        "blocking_errors": [], "reading_burden": "none", "reason": "Controlled contract test only."}
    if valid:
        assert trial._validate_review(value, {"cases": runtime.rows()}) is None
    else:
        with pytest.raises(Exception, match="review_evidence_kind"):
            trial._validate_review(value, {"cases": runtime.rows()})
    assert not (trial.root / "protocol.json").exists() and not (trial.root / "ledger").exists()
