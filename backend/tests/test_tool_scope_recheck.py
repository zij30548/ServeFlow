"""Offline v7 accounting/freeze/report tests; sockets and real stores denied."""
import asyncio
from copy import deepcopy
import difflib
import hashlib
import json
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.model_deadline import await_model
from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as business, quality_trial as quality
from evaluation import tool_scope_recheck as module
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import Frame, context
from evaluation.task_supervisor import SupervisorError, supervise


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PROOF = {"kind": "controlled-fixture"}
ORIGINAL_ARTIFACTS = module._artifacts
ORIGINAL_CANDIDATE_PROOF = module.candidate_proof
ORIGINAL_ARCHIVED = module.archived_v2


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def deny(*args, **kwargs): raise AssertionError("tool-scope tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for key in tuple(os.environ):
        if key.startswith("ECHOMIND_"): monkeypatch.delenv(key)
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    secret_reads = []
    def config(*, secret=False):
        secret_reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": "controlled-not-a-key"} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"max_tokens": 256, "messages": [{"role": "user", "content": case["input"]["message"]}],
            "model": base.MODEL, "temperature": .1, "thinking": {"type": "disabled"}}
        return {"case_id": case["id"], "operation_id": "controlled", "input_sha256": module.canonical_payload_hash(case["input"]),
            "payload": payload, "payload_sha256": module.canonical_payload_hash(payload),
            "request_bytes_sha256": hashlib.sha256(httpx.Request("POST", BASE, json=payload).content).hexdigest()}
    monkeypatch.setattr(base, "_configuration", config)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(business.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(business.engine, "offline_data_proof", lambda: {"controlled": True})
    monkeypatch.setattr(business.repair, "data_binding", lambda: module.canonical_payload_hash(PROOF))
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled-current-source": "a" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-old-tree": "b" * 64, "old_scores_recomputed": False})
    old = tmp_path / "old-v2"; old.mkdir(mode=0o700)
    monkeypatch.setattr(module, "OLD_ROOT", old)
    return secret_reads


def write(path, value):
    path.write_text(json.dumps(value)); path.chmod(0o600)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def prepared(tmp_path):
    trial = module.ToolScopeRecheck(tmp_path / "new-v7")
    p = asyncio.run(trial.prepare())
    old = deepcopy(p)
    old["initial_intent_requests"] = [{**deepcopy(row), "case_id": arm + "-" + row["case_id"],
        "arm": arm, "source_case_id": row["case_id"]}
        for row in p["initial_intent_requests"] for arm in ("A", "B", "C", "D")]
    write(module.OLD_ROOT / "protocol.json", old)
    return trial, p


def approve(trial):
    return trial.authorize("Controlled offline test allowance only, no live model authorization", confirmed=True)


def checks(case, artifact):
    return {"checks": {"execution_complete": artifact["execution_status"] != "failed", "tool_task_correct": True},
        "blockers": [], "mechanisms": {"fresh_read_each_relevant_turn": False, "exact_projection": False,
                                       "required_agents": False}}


def artifact(case, status="success"):
    turns = [{"turn_id": f"turn-{i}", "question": question,
        "result": {"response": "Controlled response, not model quality evidence", "status": status},
        "tool_evidence": {"calls": [], "blockers": []}, "judge_overall": 1., "judge": {"judge_failed": False}}
        for i, question in enumerate(case["turns"], 1)]
    return {"case_id": case["id"], "data_kind": "controlled-fixture", "turns": turns,
        "execution_status": status, "judge_ready": True, "semantic_review": None,
        "answer": turns[-1]["result"]["response"], "wall_latency_ms": 1., "actor_path_latency_ms": 1.,
        "deterministic": {"checks": {"controlled_only": True}, "blockers": []}}


def fixture_execution(tmp_path, monkeypatch, failure=None):
    trial, p = prepared(tmp_path); approve(trial)
    seen = []
    async def execute(case, sdk, data, *, ensure_healthy):
        seen.append(case["id"]); ensure_healthy()
        for number, question in enumerate(case["turns"], 1):
            with context(Frame(case["id"], f"turn-{number}", "intent", None)):
                await await_model(sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": question}], extra_body={"thinking": {"type": "disabled"}}), timeout_s=8)
        return artifact(case, "failed" if len(seen) == 1 else "success")
    monkeypatch.setattr(business.engine, "execute_case", execute)
    monkeypatch.setattr(module, "shared_checks", checks)
    close = AsyncMock(); data = SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=close)
    def delegate(frame):
        def response(request):
            assert frame.case_id not in request.content.decode()
            assert request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 8.)
            assert trial.ledger(p).status()["pending"]
            if failure == "network": raise httpx.ReadTimeout("controlled")
            if failure == "auth": return httpx.Response(401)
            return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
                "model": base.MODEL, "content": [{"type": "text", "text": "controlled"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1}})
        return httpx.MockTransport(response)
    return trial, p, seen, close, data, delegate


def test_five_parent_hooks_preserve_old_default_execution_and_grading(tmp_path):
    old = business.BusinessRegression(tmp_path / "unused-old")
    assert old.sdk_timeout_s() == 8 and old.http_limits() == business.engine.HTTP_LIMITS
    assert old.http_limits()["request_deadline_s"] == 8 and old.deadline_policy() is None
    assert old.review_policy() == business.REVIEW_POLICY and old.grading_version() == "task-business-v4"
    assert old.execution_policy() == business.POLICY


def test_new_plan_is_original_twelve_d_cases_and_zero_sends(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 312
    assert p["plan"] == business.fixed_plan() and len(p["plan"]) == 12
    assert sum(len(v["turn_ids"]) for v in p["plan"]) == 13
    assert p["review_policy"] == quality.REVIEW_POLICY and trial.sdk_timeout_s() == 30
    assert p["http_limits"]["request_deadline_s"] == 30
    assert p["policy"]["stage_clock"]["network_deadlines_by_phase_s"] == {
        "intent": 8, "agent": 15, "rag_rewrite": 10, "rag_rerank": 30, "composer": 8, "judge": 8}
    assert trial.deadline_policy().overhead_limit_s == 30 and trial.deadline_policy().cleanup_grace_s == .25
    assert trial.verify() == p and not any(isolated)
    r = trial.report()
    assert r["correct_handling"]["counts"]["not_executed"] == 12
    assert r["supported_completion"]["denominator"] == 8
    assert r["tool_tasks"]["denominator"] == 10 and r["boundary_handling"]["denominator"] == 4
    assert r["actual_requests"] == r["authorized_requests"] == 0
    assert not r["model_quality_evidence"] and "network deadline remains eight" not in json.dumps(r)


@pytest.mark.parametrize("field", ["source", "history", "phase", "model", "question", "judge", "old_payload", "old_parameters", "old_arm", "old_source", "old_unknown_field"])
def test_freeze_damage_stops_before_secret_store_and_send(tmp_path, monkeypatch, isolated, field):
    trial, p = prepared(tmp_path); approve(trial)
    if field == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"changed": "f" * 64})
    elif field == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    elif field in {"old_payload", "old_parameters", "old_arm", "old_source", "old_unknown_field"}:
        old = bench.read_json(module.OLD_ROOT / "protocol.json")
        if field == "old_payload": old["initial_intent_requests"][3]["request_bytes_sha256"] = "f" * 64
        elif field == "old_arm": old["initial_intent_requests"][3]["arm"] = "A"
        elif field == "old_source": old["initial_intent_requests"][3]["source_case_id"] = "e02"
        elif field == "old_unknown_field": old["initial_intent_requests"][3]["extra"] = "unexpected"
        else: old["agent_parameters"] = {"different": True}
        write(module.OLD_ROOT / "protocol.json", old)
    else:
        changed = deepcopy(p)
        if field == "phase": changed["policy"]["stage_clock"]["network_deadlines_by_phase_s"]["agent"] = 60
        elif field == "model": changed["model"] = "other-model"
        elif field == "question": changed["initial_intent_requests"][0]["payload"]["messages"][0]["content"] = "easier"
        else: changed["review_policy"] = {"ignore_failures": True}
        write(trial.root / "protocol.json", changed)
    store = AsyncMock(); monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    store.assert_not_called(); assert not any(isolated)


@pytest.mark.parametrize("field", ["base_url", "messages_url", "effective_environment", "data_proof_sha256", "offline_data_proof"])
def test_old_fixed_context_cannot_drift_while_current_protocol_remains_valid(tmp_path, monkeypatch, isolated, field):
    trial, _ = prepared(tmp_path)
    approve(trial)
    old = bench.read_json(module.OLD_ROOT / "protocol.json")
    old[field] = {"changed": True} if isinstance(old[field], dict) else "different-offline-value"
    write(module.OLD_ROOT / "protocol.json", old)
    store = AsyncMock()
    monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError, match="tool_scope_business_parameters_changed"):
        asyncio.run(trial.execute())
    store.assert_not_called()
    assert not any(isolated)


def test_unknown_attempt_preserves_occupancy_and_rejects_restart_and_reinitialization(tmp_path):
    trial, p = prepared(tmp_path); approve(trial)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        ledger.reserve("e01", "turn-1", "agent", "general", "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    with pytest.raises((TaskBudgetError, base.TrialError)):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None),
            controlled_data=SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=AsyncMock())))
    assert tree(trial.root) == before and ledger.status()["attempts"] == 1
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())


@pytest.mark.parametrize("failure", ["network", "auth"])
def test_execution_error_pauses_after_one_claim_without_retry_or_next_case(tmp_path, monkeypatch, failure):
    trial, p, seen, close, data, delegate = fixture_execution(tmp_path, monkeypatch, failure)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    state = trial.ledger(p).status()
    assert state["attempts"] == 1 and state["paused"] and not state["pending"]
    assert seen == ["e01"] and close.await_count == 1
    before = tree(trial.root)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert tree(trial.root) == before


def test_controlled_full_run_keeps_order_shared_scoring_and_exact_wire_records(tmp_path, monkeypatch):
    trial, p, seen, close, data, delegate = fixture_execution(tmp_path, monkeypatch)
    r = asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert seen == list(business.CASE_IDS) and r["actual_requests"] == r["sealed_requests"] == 13
    assert close.await_count == 1 and not r["ledger"]["paused"] and not r["model_quality_evidence"]
    assert r["correct_handling"]["counts"]["failed"] == 1
    assert r["tool_tasks"]["counts"]["pending_review"] == 10
    first = bench.read_json(trial.root / "physical-001.json")
    assert first["sdk_limit_s"] == 30 and first["network_limit_s"] == first["active_stage_limit_s"] == 8
    assert len(list(trial.root.glob("request-*.json"))) == 13
    for row in r["cases"]:
        review = {"case_id": row["case_id"], "artifact_sha256": bench.sha256(trial.root / f"case-{row['case_id']}.json"),
            "reviewer": "controlled-test", "method": "controlled-fixture", "reason": "Controlled report mechanics only",
            "dimensions": dict.fromkeys(business.engine.REVIEW_FIELDS, True), "blocking_errors": [], "reading_burden": "some"}
        trial.review(review)
    final = trial.report()
    assert final["correct_handling"]["numerator"] == 11 and final["tool_tasks"]["numerator"] == 10
    assert any(row["business_status"] == "failed" for row in final["business_projection_gate_diagnostic_only"]["cases"])
    # An answer's failed execution need not make its independently correct
    # tool-call dimension false; freshness/exact text are never main gates.
    before = tree(trial.root)
    asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert tree(trial.root) == before and len(seen) == 12


def test_old_authorization_cannot_be_reused(tmp_path):
    trial, p = prepared(tmp_path); value = approve(trial)
    value["schema_version"] = business.BusinessRegression.authorization_schema
    write(trial.root / "authorization.json", value)
    with pytest.raises(base.TrialError, match="authorization_changed"): asyncio.run(trial.execute())
    assert trial.ledger(p).status()["attempts"] == 0


def test_new_worker_is_fixed_and_keeps_nine_hundred_second_outer_limit(tmp_path):
    command = [sys.executable, "-B", "-u", "-m", "evaluation.tool_scope_recheck", "_worker"]
    with pytest.raises(SupervisorError, match="deadline_invalid"):
        supervise(command, root=tmp_path, deadline_s=901)
    with pytest.raises(SupervisorError, match="fixed_worker"):
        supervise(command[:-2] + ["evaluation.arbitrary", "_worker"], root=tmp_path, deadline_s=1)


def test_current_artifacts_freeze_runner_hooks_supervisor_and_production_test(monkeypatch):
    monkeypatch.setattr(module.phase, "_artifacts", lambda: {"evaluation/task_supervisor.py": "a" * 64,
        "evaluation/task_business_regression.py": "b" * 64})
    monkeypatch.setattr(module.phase, "_current_semantic_files", lambda: {"agents/agent_orchestrator.py": "c" * 64})
    monkeypatch.setattr(bench, "sha256", lambda path: "d" * 64)
    proof = ORIGINAL_ARTIFACTS()
    assert set(module._EXTRA) <= proof.keys()
    assert {"evaluation/task_supervisor.py", "evaluation/task_business_regression.py", "agents/agent_orchestrator.py"} <= proof.keys()


@pytest.mark.parametrize("change", [None, "agent", "other_business", "added", "removed", "evaluation", "old_agent"])
def test_only_exact_approved_agent_patch_is_accepted(tmp_path, monkeypatch, change):
    root = tmp_path / "source"; (root / "agents").mkdir(parents=True)
    old, new = b"old line\n", b"new line\n"
    current = root / module.AGENT_PATH; current.write_bytes(new)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    oldfiles = {module.AGENT_PATH: digest(old), "core/logic.py": "a" * 64}
    nowfiles = {**oldfiles, module.AGENT_PATH: digest(new)}
    oldart = {**oldfiles, "evaluation/task_runtime.py": "b" * 64}
    nowart = {**nowfiles, "evaluation/task_runtime.py": "b" * 64}
    patch = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
        fromfile="before/" + module.AGENT_PATH, tofile="after/" + module.AGENT_PATH, n=3)).encode()
    monkeypatch.setattr(module, "APPROVED_PATCH_SHA256", digest(patch))
    monkeypatch.setattr(bench, "BACKEND_ROOT", root)
    monkeypatch.setattr(module, "_git_agent", lambda: old)
    monkeypatch.setattr(module.phase, "_git_semantic_files", lambda: oldfiles)
    monkeypatch.setattr(module.phase, "_current_semantic_files", lambda: nowfiles)
    monkeypatch.setattr(module.phase, "_artifacts", lambda: nowart)
    if change == "agent": current.write_bytes(b"another patch\n")
    elif change == "other_business": nowfiles["core/logic.py"] = "c" * 64
    elif change == "added": nowfiles["skills/new/SKILL.md"] = "c" * 64
    elif change == "removed": nowfiles.pop("core/logic.py")
    elif change == "evaluation": nowart["evaluation/task_runtime.py"] = "c" * 64
    elif change == "old_agent": oldart[module.AGENT_PATH] = "c" * 64
    if change:
        with pytest.raises(base.TrialError): ORIGINAL_CANDIDATE_PROOF({"artifacts": oldart})
    else:
        p = ORIGINAL_CANDIDATE_PROOF({"artifacts": oldart})
        assert p["only_changed_business_file"] == module.AGENT_PATH
        assert p["approved_unified_patch_sha256"] == digest(patch)


def test_historical_tree_damage_is_rejected_without_old_verify_or_send(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_tree", lambda root: "changed")
    with pytest.raises(base.TrialError, match="old_tree_changed"): ORIGINAL_ARCHIVED()
