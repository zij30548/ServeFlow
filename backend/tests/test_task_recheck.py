"""Post-fix batch tests: temporary evidence and controlled model transport only."""
import asyncio
from copy import deepcopy
import hashlib
import json
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_recheck as module, task_trial as engine
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_budget import TaskBudget
from evaluation.trial_ledger import TrialLedger
from evaluation.task_transport import Frame, context

BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
ORIGINAL_ARCHIVE = module.archived_proof


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*args, **kwargs): raise AssertionError("no services allowed")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    for key in tuple(__import__("os").environ):
        if key.startswith("ECHOMIND_"): monkeypatch.delenv(key)
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    def configuration(*, secret=False):
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": "controlled-only"} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"model": base.MODEL, "max_tokens": 256, "temperature": .1,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": case["input"]["message"]}]}
        return {"case_id": case["id"], "operation_id": "controlled", "input_sha256": canonical_payload_hash(case["input"]),
            "payload": payload, "payload_sha256": canonical_payload_hash(payload),
            "request_bytes_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(engine, "_dependencies", lambda: {"controlled-runtime": "1"})
    monkeypatch.setattr(engine, "offline_data_proof", lambda: {"kind": "controlled-proof"})
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled-source": "a" * 64})
    monkeypatch.setattr(module, "archived_proof", lambda: {"prior": "b" * 64})
    monkeypatch.setattr(engine, "_mysql_url", lambda: None)


def prepared(tmp_path):
    trial = module.TaskRecheck(tmp_path / "new-batch")
    return trial, asyncio.run(trial.prepare())


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_prepare_has_zero_authorization_and_same_fixed_six_case_seven_turn_plan(tmp_path):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-task-protocol-v2" and p["batch"] == module.BATCH
    assert p["candidate"] == module.CANDIDATE and p["authorized_model_requests"] == 0
    assert [v["case_id"] for v in p["plan"]] == [f"e{i:02d}" for i in range(1, 7)]
    assert sum(len(v["turn_ids"]) for v in p["plan"]) == 7
    assert trial.verify() == p and not (trial.root / "authorization.json").exists()
    report = trial.report()
    assert report["authorized_requests"] == report["ledger"]["attempts"] == 0
    assert report["counts"]["not_executed"] == 6
    for case, operation in zip(engine.cases(), p["initial_intent_requests"]):
        body = json.dumps(operation["payload"], ensure_ascii=False)
        assert case["expected"]["rationale"] not in body and "required_facts" not in body
    before = tree(trial.root)
    with pytest.raises(base.TrialError, match="task_root_already_exists"): asyncio.run(trial.prepare())
    assert tree(trial.root) == before


def test_no_authorization_means_no_secret_database_or_model_access(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    data, model = AsyncMock(), AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", data); monkeypatch.setattr(engine, "execute_case", model)
    def no_secret(*, secret=False):
        assert not secret
        return {"model": base.MODEL, "base_url": BASE}
    monkeypatch.setattr(base, "_configuration", no_secret)
    with pytest.raises(FileNotFoundError): asyncio.run(trial.execute())
    data.assert_not_called(); model.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0
    assert asyncio.run(trial.preflight()) == {"ready": False, "reason": "dedicated_eval_mysql_not_configured", "model_requests": 0}


@pytest.mark.parametrize("damage", ["code", "archive", "candidate", "root_copy", "authorization"])
def test_changed_freeze_or_allowance_blocks_before_database(tmp_path, monkeypatch, damage):
    trial, _ = prepared(tmp_path)
    trial.authorize("Controlled test fixture only", confirmed=True)
    before = tree(trial.root / "ledger")
    data = AsyncMock(); monkeypatch.setattr(engine, "build_live_data", data)
    if damage == "code": monkeypatch.setattr(module, "_artifacts", lambda: {"source": "c" * 64})
    elif damage == "archive": monkeypatch.setattr(module, "archived_proof", lambda: {"prior": "c" * 64})
    elif damage == "root_copy":
        other = tmp_path / "copied"; shutil.copytree(trial.root, other); trial = module.TaskRecheck(other)
    else:
        name = "authorization.json" if damage == "authorization" else "protocol.json"
        value = bench.read_json(trial.root / name)
        value["max_requests" if damage == "authorization" else "candidate"] = 169 if damage == "authorization" else "changed"
        (trial.root / name).write_text(json.dumps(value)); (trial.root / name).chmod(0o600)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    data.assert_not_called()
    if damage != "root_copy": assert tree(trial.root / "ledger") == before


@pytest.mark.parametrize("key,value", [
    ("ECHOMIND_INTENT_TIMEOUT_S", "9"), ("ECHOMIND_LLM_TIMEOUT_S", "30"),
    ("ECHOMIND_COMPOSER_TIMEOUT_S", "9"), ("ECHOMIND_COMPOSER_MAX_TOKENS", "2000"),
    ("ECHOMIND_COMPOSER_TEMPERATURE", ".2"), ("ECHOMIND_MONITOR_FALLBACK_PENALTY", ".8"),
    ("ECHOMIND_MONITOR_COOLDOWN_SECONDS", "120"), ("ECHOMIND_TOOL_TRACE_MAX", "100"),
    ("ECHOMIND_GENERAL_MODEL", "different"), ("ECHOMIND_TECHNICAL_MODEL", base.MODEL),
    ("ORT_DISABLE_TELEMETRY", "0"), ("INTENT_EMBEDDING_PROVIDER", "auto"),
    ("ECHOMIND_LLM_TIMEOUT_S", "nan"), ("ECHOMIND_TOOL_TRACE_MAX", "bad")])
def test_effective_configuration_changes_fail_closed(tmp_path, monkeypatch, key, value):
    trial, p = prepared(tmp_path); before = tree(trial.root)
    monkeypatch.setenv(key, value)
    with pytest.raises(base.TrialError): trial.verify()
    assert tree(trial.root) == before


def test_historical_directories_are_never_adopted():
    for name in ("intent-development-v1", "intent-validation-v1", "intent-recheck-v2", "task-e2e-v1"):
        with pytest.raises(base.TrialError, match="fixed_task_recheck_root_required"):
            module.TaskRecheck(bench.BACKEND_ROOT / "data/benchmarks" / name)


def test_inherited_execute_uses_new_verify_and_shared_budget(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); trial.authorize("Controlled test fixture only", confirmed=True)
    sends = []
    async def execute(case, sdk, data, *, ensure_healthy):
        results = []
        for number in range(1, len(case["turns"]) + 1):
            ensure_healthy()
            with context(Frame(case["id"], f"turn-{number}", "intent")):
                await sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": "controlled"}], extra_body={"thinking": {"type": "disabled"}})
            results.append({"judge_overall": 1., "judge": {"judge_failed": False}})
        return {"case_id": case["id"], "execution_status": "success", "turns": results,
            "deterministic": {"checks": {"controlled_gate": True}, "blockers": []},
            "judge_ready": True, "data_kind": "controlled-fixture", "semantic_review": None}
    monkeypatch.setattr(engine, "execute_case", execute)
    async def close(): pass
    data = SimpleNamespace(proof={"kind": "controlled-fixture"}, assert_healthy=lambda: None, close=close)
    def factory(frame):
        async def handle(request):
            sends.append(frame)
            return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant", "model": base.MODEL,
                "content": [{"type": "text", "text": "controlled"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1}})
        return httpx.MockTransport(handle)
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    assert len(sends) == report["ledger"]["attempts"] == 7
    assert report["counts"]["pending_review"] == 6 and report["counts"]["passed"] == 0
    assert all(record["evidence_kind"] == "controlled-transport" for record in report["cases"])
    assert trial.ledger(p).status()["pending"] == []


def archive_fixture(tmp_path, monkeypatch):
    """Real temporary hash chains; fake historical documents, never model sends."""
    recheck, task = tmp_path / "old-intent", tmp_path / "old-task"
    recheck.mkdir(mode=0o700); task.mkdir(mode=0o700)
    legacy = {"old_24_and_36": "controlled-proof"}
    operations = [{"case_id": f"c{i:02d}", "operation_id": f"old-c{i:02d}"} for i in range(1, 25)]
    protocol = {"git_head": "a" * 40, "artifacts": {"old.py": "a" * 64}, "operations": operations,
        "archived_evidence_proof": legacy, "root_binding_sha256": base._root_hash(recheck)}
    base._write(recheck, "protocol.json", protocol); digest = bench.sha256(recheck / "protocol.json")
    ledger = TrialLedger(recheck / "ledger", digest, [op["operation_id"] for op in operations], 24)
    for op in operations:
        filename = op["case_id"] + "-observation.json"
        base._write(recheck, filename, {"kind": "controlled-archived-output"})
        ledger.reserve(op["operation_id"], "a" * 64)
        ledger.finish(op["operation_id"], "received", bench.sha256(recheck / filename))
    state = ledger.status()
    base._write(recheck, "authorization.json", {"protocol_sha256": digest, "max_requests": 24,
        "root_binding_sha256": base._root_hash(recheck), "initial_ledger_chain_hash": ledger.initial_chain_hash})
    report = "report-" + state["chain_hash"] + ".json"
    base._write(recheck, report, {"kind": "controlled-old-score-not-recomputed"})
    prep = {"git_head": "43720fd797e4ae39c6fe2061f799cf0f98664a4c", "artifacts": {"old-task.py": "b" * 64},
        "plan": engine._plan(engine.cases()), "authorized_model_requests": 0, "root_binding_sha256": base._root_hash(task)}
    base._write(task, "protocol.json", prep); prep_hash = bench.sha256(task / "protocol.json")
    TaskBudget(task / "ledger", prep_hash, prep["plan"], initialize=True)
    constants = {"OLD_RECHECK": recheck, "OLD_TASK": task, "OLD_RECHECK_PROTOCOL": digest,
        "OLD_RECHECK_AUTHORIZATION": bench.sha256(recheck / "authorization.json"),
        "OLD_RECHECK_JOURNAL": bench.sha256(recheck / "ledger/journal.jsonl"), "OLD_RECHECK_CHAIN": state["chain_hash"],
        "OLD_RECHECK_REPORT": bench.sha256(recheck / report), "OLD_TASK_PROTOCOL": prep_hash}
    for key, value in constants.items(): monkeypatch.setattr(module, key, value)
    calls = []
    monkeypatch.setattr(module.history, "archived_evidence_proof", lambda: deepcopy(legacy))
    monkeypatch.setattr(module.history, "_archived_sources", lambda record, field: calls.append((record["git_head"], field)))
    return recheck, task, calls


def test_archive_reads_old_source_and_chains_without_current_verify_or_writes(tmp_path, monkeypatch):
    recheck, task, calls = archive_fixture(tmp_path, monkeypatch); before = tree(tmp_path)
    monkeypatch.setattr(module.history.RecheckTrial, "verify", lambda *a: pytest.fail("old current-code verify forbidden"))
    monkeypatch.setattr(module.history.RecheckTrial, "report", lambda *a: pytest.fail("old rescore forbidden"))
    proof = ORIGINAL_ARCHIVE()
    assert not proof["old_allowance_reused"] and proof["task_preparation"]["actual_model_requests"] == 0
    assert len(proof["recheck"]["observation_sha256"]) == 24 and len(calls) == 2
    assert tree(tmp_path) == before


@pytest.mark.parametrize("damage", ["observation", "report", "preparation_authorization", "missing_journal", "git_source"])
def test_bad_archives_stop_without_repair(tmp_path, monkeypatch, damage):
    recheck, task, _ = archive_fixture(tmp_path, monkeypatch)
    if damage == "observation":
        path = recheck / "c01-observation.json"; path.write_text("{}"); path.chmod(0o600)
    elif damage == "report":
        path = next(recheck.glob("report-*.json")); path.write_text("{}"); path.chmod(0o600)
    elif damage == "preparation_authorization": base._write(task, "authorization.json", {"not_allowed": True})
    elif damage == "missing_journal": (recheck / "ledger/journal.jsonl").unlink()
    else:
        def bad(record, field): base._fail("archived_sources_changed")
        monkeypatch.setattr(module.history, "_archived_sources", bad)
    before = tree(tmp_path)
    with pytest.raises(base.TrialError): ORIGINAL_ARCHIVE()
    assert tree(tmp_path) == before
