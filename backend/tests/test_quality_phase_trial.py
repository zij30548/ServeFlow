"""Phase-bound trial preparation/accounting with controlled SDK responses."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.model_deadline import await_model
from evaluation import benchmark as bench, intent_trial as base
from evaluation import quality_phase_trial as module, quality_trial as previous, quality_scoring
from evaluation.quality_budget import QualityBudget, fixed_plan
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import Frame, context


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PROOF = {"kind": "controlled-fixture"}
ORIGINAL_ARTIFACTS = module._artifacts
ORIGINAL_COMPLETE_SOURCE_PROOF = module.complete_source_proof


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def deny(*args, **kwargs): raise AssertionError("phase trial tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for key in tuple(os.environ):
        if key.startswith("ECHOMIND_"): monkeypatch.delenv(key)
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    secrets = []
    def config(*, secret=False):
        secrets.append(secret)
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
    monkeypatch.setattr(previous.business.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(previous.business.engine, "offline_data_proof", lambda: {"controlled": True})
    monkeypatch.setattr(previous.business.repair, "data_binding", lambda: module.canonical_payload_hash(PROOF))
    monkeypatch.setattr(previous, "_design_proof", lambda: {"controlled-design": "a" * 64})
    monkeypatch.setattr(previous, "prior_proof", lambda: {"controlled-old-v6": "b" * 64})
    monkeypatch.setattr(previous, "_artifacts", lambda: {"core/unchanged.py": "c" * 64})
    monkeypatch.setattr(module, "_artifacts", lambda: {"core/unchanged.py": "c" * 64, "evaluation/phase.py": "d" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-old-v1": "e" * 64})
    monkeypatch.setattr(module, "complete_source_proof", lambda: {"controlled-complete-source": True})
    old = tmp_path / "old-v1"; old.mkdir(mode=0o700)
    monkeypatch.setattr(previous, "ROOT", old)
    return secrets


def write(path, value):
    path.write_text(json.dumps(value)); path.chmod(0o600)


def prepared(tmp_path):
    trial = module.QualityPhaseTrial(tmp_path / "new-v2")
    p = asyncio.run(trial.prepare())
    # Controlled old reference only, never the operator's private evidence.
    write(previous.ROOT / "protocol.json", p)
    return trial, p


def approve(trial):
    return trial.authorize("Controlled test allowance, no real model authorization", confirmed=True)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_new_contract_has_zero_sends_and_keeps_every_business_input(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-quality-phase-protocol-v2"
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 1248
    assert p["plan"] == previous.fixed_plan() and len(p["plan"]) == 48
    assert sum(len(row["turn_ids"]) for row in p["plan"]) == 52
    assert p["http_limits"]["request_deadline_s"] == 30 and trial.sdk_timeout_s() == 30
    assert p["policy"]["stage_clock"]["network_deadlines_by_phase_s"] == {
        "intent": 8, "agent": 15, "rag_rewrite": 10, "rag_rerank": 30, "composer": 8, "judge": 8}
    assert p["policy"]["stage_clock"]["model_stage_budgets"] == previous.clock.CLOCK_POLICY["model_stage_budgets"]
    assert p["review_policy"] == previous.REVIEW_POLICY
    assert p["phase_parameters"] == json.loads(json.dumps(previous.business.engine.PHASE_PARAMETERS))
    assert p["agent_parameters"] == json.loads(json.dumps(previous.business.engine.AGENT_PARAMETERS))
    assert trial.verify() == p and trial.ledger(p).status()["attempts"] == 0 and not any(isolated)
    r = trial.report()
    assert r["batch"] == module.BATCH and r["candidate"] == module.CANDIDATE
    assert r["stage_clock_policy"] == p["policy"]["stage_clock"]
    assert r["actual_requests"] == r["authorized_requests"] == 0
    assert all(row["correct_handling"]["counts"]["not_executed"] == 12 for row in r["arms"].values())


def test_parent_defaults_still_use_original_eight_second_contract(tmp_path):
    trial = previous.QualityTrial(tmp_path / "parent-default-no-write")
    assert trial.sdk_timeout_s() == 8
    assert trial.http_limits() == previous.business.engine.HTTP_LIMITS
    assert trial.execution_policy() == previous.POLICY
    assert trial.deadline_policy().overhead_limit_s == 30
    assert trial.deadline_policy().cleanup_grace_s == .25
    assert trial.http_limits()["request_deadline_s"] == 8


@pytest.mark.parametrize("field", ["source", "history", "phase_budget", "global_ceiling", "active_budget", "model", "question", "judge", "old_payload"])
def test_freeze_tampering_stops_before_credentials_or_store(tmp_path, monkeypatch, isolated, field):
    trial, p = prepared(tmp_path); approve(trial)
    if field == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"changed": "f" * 64})
    elif field == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    elif field == "old_payload":
        changed = deepcopy(p); changed["initial_intent_requests"][0]["request_bytes_sha256"] = "f" * 64
        write(previous.ROOT / "protocol.json", changed)
    else:
        changed = deepcopy(p)
        if field == "phase_budget": changed["policy"]["stage_clock"]["network_deadlines_by_phase_s"]["agent"] = 60
        elif field == "active_budget": changed["policy"]["stage_clock"]["model_stage_budgets"]["agent"] = 60
        elif field == "global_ceiling": changed["http_limits"]["request_deadline_s"] = 60
        elif field == "model": changed["model"] = "different-model"
        elif field == "question": changed["initial_intent_requests"][0]["payload"]["messages"][0]["content"] = "easier"
        elif field == "judge": changed["review_policy"]["judge"] = "ignore all scores"
        write(trial.root / "protocol.json", changed)
    store = AsyncMock(); monkeypatch.setattr(previous.business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    store.assert_not_called(); assert not any(isolated)


def test_old_authorization_and_unknown_attempt_do_not_restore_new_allowance(tmp_path):
    trial, p = prepared(tmp_path); a = approve(trial)
    a["schema_version"] = previous.QualityTrial.authorization_schema
    write(trial.root / "authorization.json", a)
    with pytest.raises(base.TrialError, match="authorization_changed"):
        asyncio.run(trial.execute())
    a["schema_version"] = trial.authorization_schema; write(trial.root / "authorization.json", a)
    b = trial.ledger(p)
    with b.exclusive_run() as lease:
        b.start_case("A-e01", lease=lease)
        b.reserve("A-e01", "turn-1", "agent", "general", "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    with pytest.raises((TaskBudgetError, base.TrialError)):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None),
            controlled_data=SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=AsyncMock())))
    assert tree(trial.root) == before and b.status()["attempts"] == 1
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())


def fake_checks(case, artifact):
    return {"checks": {"execution_complete": artifact["execution_status"] != "failed", "tool_task_correct": True},
        "blockers": [], "mechanisms": {"fresh_read_each_relevant_turn": None, "exact_projection": None}}


def fake_result(case, status="success"):
    return {"schema_version": "serveflow-task-case-result-v1", "case_id": case["id"], "data_kind": "controlled-fixture",
        "turns": [{"turn_id": f"turn-{i}", "request_id": f"task-eval-{case['id']}-{i}", "question": q,
                   "result": {"status": status}, "judge_overall": 1., "judge": {"judge_failed": False}}
                  for i, q in enumerate(case["turns"], 1)],
        "deterministic": {"checks": {"execution_complete": status == "success"}, "blockers": []},
        "wall_latency_ms": 1., "actor_path_latency_ms": 1., "execution_status": status,
        "judge_ready": True, "semantic_review": None, "answer": "Controlled reply, not semantic model evidence"}


def execution_fixture(tmp_path, monkeypatch, failure=None):
    from evaluation import quality_arm_runtime
    trial, p = prepared(tmp_path); approve(trial)
    seen = []
    async def run(arm, case, sdk, data, *, ensure_healthy, frame_case_id):
        seen.append(frame_case_id); ensure_healthy()
        for number, question in enumerate(case["turns"], 1):
            with context(Frame(frame_case_id, f"turn-{number}", "intent", None)):
                await await_model(sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": question}], extra_body={"thinking": {"type": "disabled"}}), timeout_s=8)
        return fake_result(case, "failed" if len(seen) == 1 else "success")
    monkeypatch.setattr(quality_arm_runtime, "execute_arm_case", run)
    monkeypatch.setattr(quality_scoring, "shared_checks", fake_checks)
    closed = AsyncMock(); data = SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=closed)
    def delegate(frame):
        def response(request):
            assert frame.case_id not in request.content.decode()
            assert request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 8.)
            assert trial.ledger(p).status()["pending"]
            if failure == "network": raise httpx.ReadTimeout("controlled failure")
            if failure == "auth": return httpx.Response(401)
            return httpx.Response(200, json={"id": "controlled-response", "type": "message", "role": "assistant",
                "model": base.MODEL, "content": [{"type": "text", "text": "controlled"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1}})
        return httpx.MockTransport(response)
    return trial, p, seen, closed, data, delegate


def test_all_48_arms_use_shared_count_and_phase_transport_not_old_sdk_eight(tmp_path, monkeypatch):
    trial, p, seen, closed, data, delegate = execution_fixture(tmp_path, monkeypatch)
    r = asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert seen == [v["case_id"] for v in p["plan"]] and len(seen) == 48
    assert r["actual_requests"] == r["sealed_requests"] == 52 and closed.await_count == 1
    assert r["arms"]["A"]["correct_handling"]["counts"]["failed"] == 1
    assert not r["ledger"]["paused"] and not r["model_quality_evidence"]
    physical = bench.read_json(trial.root / "physical-001.json")
    assert physical["sdk_limit_s"] == 30
    assert physical["network_limit_s"] == physical["active_stage_limit_s"] == 8
    assert len(list(trial.root.glob("request-*.json"))) == 13
    before = tree(trial.root)
    asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert tree(trial.root) == before and len(seen) == 48


@pytest.mark.parametrize("failure", ["network", "auth"])
def test_execution_failure_is_sticky_and_no_next_arm_or_retry(tmp_path, monkeypatch, failure):
    trial, p, seen, closed, data, delegate = execution_fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    s = trial.ledger(p).status()
    assert s["attempts"] == 1 and s["paused"] and not s["pending"]
    assert seen == ["A-e01"] and closed.await_count == 1
    before = tree(trial.root)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert tree(trial.root) == before


@pytest.mark.parametrize("mutation", ["hash", "added_file", "removed_file"])
def test_business_file_bytes_and_set_are_same_as_old_trial(monkeypatch, mutation):
    old = {"artifacts": {"core/unchanged.py": "c" * 64}, "model": base.MODEL,
        "phase_parameters": {}, "agent_parameters": {}, "review_policy": {}, "cases_sha256": "a" * 64}
    assert module.business_input_proof(old)["semantic_file_set_and_bytes_unchanged"]
    changed = dict(old["artifacts"])
    if mutation == "hash": changed["core/unchanged.py"] = "d" * 64
    elif mutation == "added_file": changed["skills/extra/SKILL.md"] = "d" * 64
    else: changed.clear()
    monkeypatch.setattr(previous, "_artifacts", lambda: changed)
    with pytest.raises(base.TrialError, match="business_inputs_changed"):
        module.business_input_proof(old)


def test_frozen_sources_include_phase_transport_and_its_tests(monkeypatch):
    monkeypatch.setattr(previous, "_artifacts", lambda: {})
    monkeypatch.setattr(bench, "sha256", lambda path: "a" * 64)
    assert set(module._EXTRA) <= set(ORIGINAL_ARTIFACTS())


def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "backend"; root.mkdir()
    for folder in module._SOURCE_DIRS + module._DOCUMENT_DIRS:
        (root / folder).mkdir()
    originals = {"core/logic.py": b"print('controlled')\n", "skills/team/notes.txt": b"controlled non-SKILL rules\n"}
    for name, raw in originals.items():
        p = root / name; p.parent.mkdir(exist_ok=True); p.write_bytes(raw)
    objects = [(name, bytes(str(number) * 40, "ascii"), raw)
               for number, (name, raw) in enumerate(originals.items(), 1)]
    seen = []
    def run(command, *, cwd, capture_output, check, input=None, text=False, timeout=None):
        seen.append(command)
        if command == ["git", "rev-parse", "HEAD"]:
            assert text
            return SimpleNamespace(stdout="controlled-source-head\n")
        assert cwd == root.parent and capture_output and check and timeout == 10
        if command[1] == "ls-tree":
            return SimpleNamespace(stdout=b"".join(b"100644 blob " + oid + b"\tbackend/" + name.encode() + b"\0"
                for name, oid, raw in objects))
        assert command == ["git", "cat-file", "--batch"]
        assert input == b"\n".join(oid for name, oid, raw in objects) + b"\n"
        return SimpleNamespace(stdout=b"".join(oid + b" blob " + str(len(raw)).encode() + b"\n" + raw + b"\n"
            for name, oid, raw in objects))
    monkeypatch.setattr(module.subprocess, "run", run)
    return root, seen, run


def test_complete_git_proof_uses_two_bounded_batch_calls_and_preserves_file_set(tmp_path, monkeypatch):
    root, seen, _ = source_tree(tmp_path, monkeypatch)
    v = ORIGINAL_COMPLETE_SOURCE_PROOF(root)
    assert v["complete_source_file_count"] == 2
    assert v["complete_source_file_set_and_bytes_unchanged"]
    assert [c[1] for c in seen] == ["ls-tree", "cat-file"]
    # Existing Python cache bytes are not input; never modify/remove them.
    cache = root / "core/__pycache__"; cache.mkdir(); p = cache / "logic.pyc"; p.write_bytes(b"old-cache")
    assert ORIGINAL_COMPLETE_SOURCE_PROOF(root) == v
    assert p.read_bytes() == b"old-cache"


@pytest.mark.parametrize("name", ["skills/team/extra.txt", "skills/team/extra.json", "skills/team/extra.md",
    "skills/team/SKILL.md", "skills/__pycache__/still-loadable.JSON", "knowledge/extra.json", "core/new.py"])
def test_new_loadable_rule_or_source_outside_old_manifest_is_rejected(tmp_path, monkeypatch, name):
    root, _, _ = source_tree(tmp_path, monkeypatch)
    p = root / name; p.parent.mkdir(exist_ok=True); p.write_text("controlled extra")
    with pytest.raises(base.TrialError, match="complete_business_inputs_changed"):
        ORIGINAL_COMPLETE_SOURCE_PROOF(root)


def test_new_non_skill_rule_blocks_preparation_and_send_before_secret_or_store(tmp_path, monkeypatch, isolated):
    root, _, _ = source_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "complete_source_proof", lambda: ORIGINAL_COMPLETE_SOURCE_PROOF(root))
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-old-history": True,
                                                       "complete_source": module.complete_source_proof()})
    trial, p = prepared(tmp_path)
    approve(trial)
    extra = root / "skills/team/inline-rules.json"; extra.write_text('{"controlled":true}')
    store = AsyncMock(); monkeypatch.setattr(previous.business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError, match="complete_business_inputs_changed"):
        asyncio.run(trial.execute())
    store.assert_not_called()
    assert not any(isolated) and trial.ledger(p).status()["attempts"] == 0
    other = module.QualityPhaseTrial(tmp_path / "blocked-before-prepare")
    with pytest.raises(base.TrialError, match="complete_business_inputs_changed"):
        asyncio.run(other.prepare())
    assert not other.root.exists() and not any(isolated)


@pytest.mark.parametrize("change", ["modified", "deleted", "file_symlink", "directory_symlink"])
def test_complete_source_proof_rejects_mutation_or_outside_links(tmp_path, monkeypatch, change):
    root, _, _ = source_tree(tmp_path, monkeypatch)
    p = root / "core/logic.py"
    if change == "modified": p.write_text("different")
    elif change == "deleted": p.unlink()
    else:
        outside = tmp_path / "outside"; outside.mkdir(); (outside / "rule.txt").write_text("outside")
        if change == "file_symlink": (root / "skills/outside.txt").symlink_to(outside / "rule.txt")
        else: (root / "skills/outside-dir").symlink_to(outside, target_is_directory=True)
    with pytest.raises(base.TrialError): ORIGINAL_COMPLETE_SOURCE_PROOF(root)


@pytest.mark.parametrize("damage", ["missing_commit", "old_symlink", "truncated_blob", "extra_tail"])
def test_corrupt_or_unavailable_old_git_source_fails_closed(tmp_path, monkeypatch, damage):
    root, _, run = source_tree(tmp_path, monkeypatch)
    def changed(command, **kwargs):
        if damage == "missing_commit": raise module.subprocess.CalledProcessError(1, command)
        result = run(command, **kwargs)
        if command[1] == "ls-tree" and damage == "old_symlink":
            result.stdout = result.stdout.replace(b"100644", b"120000", 1)
        elif command[1] == "cat-file":
            if damage == "truncated_blob": result.stdout = result.stdout[:-2]
            elif damage == "extra_tail": result.stdout += b"unexpected"
        return result
    monkeypatch.setattr(module.subprocess, "run", changed)
    with pytest.raises(base.TrialError, match="old_git_source_invalid"):
        ORIGINAL_COMPLETE_SOURCE_PROOF(root)


def test_phase_cli_is_the_only_new_worker(monkeypatch):
    seen = []
    monkeypatch.setattr(module.business, "_main", lambda cls, worker, argv: seen.append((cls, worker, argv)) or 7)
    assert module.main(["verify"]) == 7
    assert seen == [(module.QualityPhaseTrial, "evaluation.quality_phase_trial", ["verify"])]
