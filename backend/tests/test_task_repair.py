"""Post-repair preparation/admission checks; controlled data and transport only.

No application stores, credentials, historical evidence or model services are
read. Temporary ledgers exercise the production runner's existing protections.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_repair as module, task_trial as engine
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_budget import TaskBudgetError
from evaluation.task_transport import Frame, context


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
ORIGINAL_DATA_BINDING = module.data_binding
ORIGINAL_ARTIFACTS = module._artifacts
CONTROLLED_PROOF = {"kind": "controlled-fixture"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repair preparation tests must not contact services")
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
                   "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": case["input"]["message"]}]}
        return {"case_id": case["id"], "operation_id": "controlled-capture",
                "input_sha256": canonical_payload_hash(case["input"]), "payload": payload,
                "payload_sha256": canonical_payload_hash(payload),
                "request_bytes_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(engine, "_dependencies", lambda: {"controlled-runtime": "1"})
    monkeypatch.setattr(engine, "offline_data_proof", lambda: {"kind": "controlled-offline-proof"})
    monkeypatch.setattr(module, "_artifacts", lambda: {"controlled-source": "a" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {
        "previous_batch": "controlled-paused-v2", "attempts": 28, "unknown": 1,
        "old_allowance_reused": False, "tree_sha256": "b" * 64})
    monkeypatch.setattr(module, "data_binding", lambda: canonical_payload_hash(CONTROLLED_PROOF))
    monkeypatch.setattr(engine, "_mysql_url", lambda: None)
    return SimpleNamespace(config_reads=reads)


def prepared(tmp_path):
    trial = module.TaskRepair(tmp_path / "new-repair-batch")
    return trial, asyncio.run(trial.prepare())


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def authorize(trial):
    return trial.authorize("Controlled fixture authorization, never a live-send allowance.", confirmed=True)


def test_prepare_is_independent_zero_send_freeze_with_unchanged_six_case_seven_turn_plan(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-task-protocol-v3"
    assert p["batch"] == "serveflow-task-e2e-v3-evidence-fix"
    assert p["candidate"] == "mysql-collections-capture-stop-policy-sources"
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 168
    assert p["prior_proof"]["attempts"] == 28 and p["prior_proof"]["unknown"] == 1
    assert not p["prior_proof"]["old_allowance_reused"]
    assert p["plan"] == engine._plan(engine.cases())
    assert len(p["plan"]) == 6 and sum(len(v["turn_ids"]) for v in p["plan"]) == 7
    assert trial.verify() == p and not (trial.root / "authorization.json").exists()
    report = trial.report()
    assert report["authorized_requests"] == report["ledger"]["attempts"] == 0
    assert report["counts"]["not_executed"] == 6 and report["counts"]["passed"] == 0
    assert report["review_policy"] == p["review_policy"] == module.REVIEW_POLICY
    assert p["review_policy"]["dimensions"] == list(engine.REVIEW_FIELDS)
    assert not any(isolated.config_reads)
    for case, operation in zip(engine.cases(), p["initial_intent_requests"]):
        body = json.dumps(operation["payload"], ensure_ascii=False)
        assert case["turns"][0] in body
        assert case["expected"]["rationale"] not in body and "required_facts" not in body
        assert p["review_policy"]["record_time"] not in body
    before = tree(trial.root)
    with pytest.raises(base.TrialError, match="task_root_already_exists"):
        asyncio.run(trial.prepare())
    assert tree(trial.root) == before


def test_missing_authorization_stops_before_secret_data_and_model(tmp_path, monkeypatch, isolated):
    trial, p = prepared(tmp_path)
    data, execute = AsyncMock(), AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", data)
    monkeypatch.setattr(engine, "execute_case", execute)
    isolated.config_reads.clear()
    with pytest.raises(FileNotFoundError):
        asyncio.run(trial.execute())
    assert not any(isolated.config_reads)
    data.assert_not_called()
    execute.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0
    assert asyncio.run(trial.preflight()) == {
        "ready": False, "reason": "dedicated_eval_mysql_not_configured", "model_requests": 0}


@pytest.mark.parametrize("damage", [
    "code", "archive", "candidate", "review_policy", "data_proof", "dependencies",
    "root_copy", "authorization", "initial_wire_hash",
])
def test_freeze_or_authorization_changes_prevent_data_access_and_spending(tmp_path, monkeypatch, damage):
    trial, p = prepared(tmp_path)
    authorize(trial)
    before = tree(trial.root / "ledger")
    data = AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", data)
    if damage == "code":
        monkeypatch.setattr(module, "_artifacts", lambda: {"new-source": "c" * 64})
    elif damage == "archive":
        monkeypatch.setattr(module, "prior_proof", lambda: {"previous_batch": "changed"})
    elif damage == "data_proof":
        monkeypatch.setattr(module, "data_binding", lambda: "d" * 64)
    elif damage == "dependencies":
        monkeypatch.setattr(engine, "_dependencies", lambda: {"controlled-runtime": "2"})
    elif damage == "root_copy":
        target = tmp_path / "copied-batch"
        shutil.copytree(trial.root, target)
        trial = module.TaskRepair(target)
    else:
        name = "authorization.json" if damage == "authorization" else "protocol.json"
        value = bench.read_json(trial.root / name)
        if damage == "authorization":
            value["max_requests"] = 169
        elif damage == "review_policy":
            value["review_policy"]["judge_role"] = "replace manual evidence review"
        elif damage == "initial_wire_hash":
            value["initial_intent_requests"][0]["request_bytes_sha256"] = "bad"
        else:
            value["candidate"] = "changed"
        (trial.root / name).write_text(json.dumps(value))
        (trial.root / name).chmod(0o600)
    with pytest.raises((base.TrialError, TaskBudgetError)):
        asyncio.run(trial.execute())
    data.assert_not_called()
    if damage != "root_copy":
        assert tree(trial.root / "ledger") == before


@pytest.mark.parametrize("key,value", [
    ("ECHOMIND_LLM_TIMEOUT_S", "30"), ("ECHOMIND_GENERAL_MODEL", "other-model"),
    ("INTENT_EMBEDDING_PROVIDER", "auto"), ("ORT_DISABLE_TELEMETRY", "0"),
])
def test_existing_effective_parameter_guard_remains_active(tmp_path, monkeypatch, key, value):
    trial, _ = prepared(tmp_path)
    before = tree(trial.root)
    monkeypatch.setenv(key, value)
    with pytest.raises(base.TrialError):
        trial.verify()
    assert tree(trial.root) == before


def test_old_roots_are_not_adopted_and_old_allowance_cannot_authorize_new_protocol(tmp_path):
    for name in ("intent-development-v1", "intent-validation-v1", "intent-recheck-v2",
                 "task-e2e-v1", "task-e2e-v2-routing-fix", "task-e2e-v3-data"):
        with pytest.raises(base.TrialError, match="fixed_task_repair_root_required"):
            module.TaskRepair(bench.BACKEND_ROOT / "data/benchmarks" / name)
    trial, p = prepared(tmp_path)
    for reference, confirmed in (("old unused allowance", False), ("", True)):
        with pytest.raises(base.TrialError, match="task_explicit_authorization_required"):
            trial.authorize(reference, confirmed=confirmed)
    assert not (trial.root / "authorization.json").exists()
    assert trial.ledger(p).status()["attempts"] == 0


def test_copied_old_authorization_is_rejected_before_secrets_or_data(tmp_path, monkeypatch, isolated):
    trial, p = prepared(tmp_path)
    old = tmp_path / "old-authorization"
    old.mkdir(mode=0o700)
    base._write(old, "authorization.json", {
        "schema_version": "serveflow-task-authorization-v1", "reference": "controlled old batch",
        "recorded_at": base._now(), "protocol_sha256": "b" * 64,
        "root_binding_sha256": base._root_hash(old), "initial_chain_hash": "c" * 64,
        "max_requests": 168, "plan": p["plan"],
    })
    before = tree(old)
    shutil.copyfile(old / "authorization.json", trial.root / "authorization.json")
    (trial.root / "authorization.json").chmod(0o600)
    data = AsyncMock()
    monkeypatch.setattr(engine, "build_live_data", data)
    isolated.config_reads.clear()
    with pytest.raises(base.TrialError, match="task_authorization_changed"):
        asyncio.run(trial.execute())
    assert not any(isolated.config_reads)
    data.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0 and tree(old) == before


def test_runtime_data_swap_is_rejected_before_model_send(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    authorize(trial)
    close = AsyncMock()
    data = SimpleNamespace(proof={"kind": "controlled-fixture", "different_identity": True},
                           assert_healthy=lambda: None, close=close)
    sends = []
    def factory(frame):
        async def handle(request):
            sends.append(request)
            raise AssertionError("changed data must not reach a model")
        return httpx.MockTransport(handle)
    with pytest.raises(base.TrialError, match="task_repair_data_identity_changed"):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    close.assert_awaited_once()
    assert sends == [] and trial.ledger(p).status()["attempts"] == 0
    assert (trial.root / "worker-stop.json").exists()


def test_data_binding_uses_only_new_fixed_preflight_path(tmp_path, monkeypatch):
    folder = tmp_path / "new-data"
    folder.mkdir(mode=0o700)
    path = folder / "preflight-v1.json"
    monkeypatch.setattr(module, "DATA_PREFLIGHT", path)
    value = {"model_requests": 0,
             "data_proof": {"schema_version": "serveflow-task-live-data-v1",
                            "mysql": {"literal_schema_grants": True}},
             "order_checks": {"own_order_amount": True, "cross_customer_denied": True}}
    base._write(folder, path.name, value)
    assert ORIGINAL_DATA_BINDING() == canonical_payload_hash(value["data_proof"])
    for mutation in ({"model_requests": 1}, {"data_proof": {"mysql": []}},
                     {"order_checks": {"own_order_amount": True, "cross_customer_denied": False}}):
        path.write_text(json.dumps({**value, **mutation}))
        path.chmod(0o600)
        with pytest.raises(base.TrialError, match="task_repair_data_preflight_required"):
            ORIGINAL_DATA_BINDING()


def test_new_freeze_includes_runner_archive_and_their_tests(monkeypatch):
    monkeypatch.setattr(engine, "_artifacts", lambda: {"existing-production.py": "a" * 64})
    paths = []
    def digest(path):
        paths.append(str(path.relative_to(bench.BACKEND_ROOT)))
        return "b" * 64
    monkeypatch.setattr(bench, "sha256", digest)
    actual = ORIGINAL_ARTIFACTS()
    expected = {"evaluation/task_repair.py", "tests/test_task_repair.py",
                "evaluation/task_repair_archive.py", "tests/test_task_repair_archive.py"}
    assert expected <= set(paths) and expected <= set(actual)
    assert actual["existing-production.py"] == "a" * 64


def test_inherited_execute_counts_shared_physical_requests_and_never_auto_passes(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    authorize(trial)
    sent = []
    async def execute(case, sdk, data, *, ensure_healthy):
        turns = []
        for number in range(1, len(case["turns"]) + 1):
            ensure_healthy()
            with context(Frame(case["id"], f"turn-{number}", "intent")):
                await sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": case["turns"][number - 1]}],
                    extra_body={"thinking": {"type": "disabled"}})
            turns.append({"judge_overall": 1., "judge": {"judge_failed": False}})
        return {"case_id": case["id"], "execution_status": "success", "turns": turns,
                "deterministic": {"checks": {"controlled_only": True}, "blockers": []},
                "judge_ready": True, "data_kind": "controlled-fixture", "semantic_review": None}
    monkeypatch.setattr(engine, "execute_case", execute)
    close = AsyncMock()
    data = SimpleNamespace(proof=deepcopy(CONTROLLED_PROOF), assert_healthy=lambda: None, close=close)
    def factory(frame):
        async def handle(request):
            sent.append((frame, json.loads(request.content)))
            return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
                "model": base.MODEL, "content": [{"type": "text", "text": "controlled transport only"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1}})
        return httpx.MockTransport(handle)
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))
    close.assert_awaited_once()
    assert len(sent) == report["ledger"]["attempts"] == 7
    assert report["counts"]["pending_review"] == 6 and report["counts"]["passed"] == 0
    assert report["ledger"]["pending"] == []
    assert all(row["evidence_kind"] == "controlled-transport" for row in report["cases"])
    assert all(module.REVIEW_POLICY["record_time"] not in json.dumps(payload) for _, payload in sent)
    before = tree(trial.root)
    assert asyncio.run(trial.execute(delegate_factory=factory, controlled_data=data))["ledger"]["attempts"] == 7
    assert len(sent) == 7 and tree(trial.root) == before


def test_pending_operation_never_restarts_or_refunds(tmp_path):
    trial, p = prepared(tmp_path)
    authorize(trial)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        ledger.reserve("e01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    data = SimpleNamespace(proof=deepcopy(CONTROLLED_PROOF), assert_healthy=lambda: None, close=AsyncMock())
    with pytest.raises(TaskBudgetError, match="pending"):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None), controlled_data=data))
    assert ledger.status()["attempts"] == 1 and len(ledger.status()["pending"]) == 1
    assert tree(trial.root) == before


@pytest.mark.parametrize("name", ["stopped.json", "supervisor-stop.json", "worker-stop.json"])
def test_stop_marker_blocks_restart_without_spending(tmp_path, monkeypatch, name):
    trial, p = prepared(tmp_path)
    authorize(trial)
    base._write(trial.root, name, {"kind": "controlled-stop"})
    before = tree(trial.root)
    execute = AsyncMock()
    monkeypatch.setattr(engine, "execute_case", execute)
    with pytest.raises(base.TrialError, match="task_paused"):
        asyncio.run(trial.execute())
    execute.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0
    assert tree(trial.root) == before
