"""v5 prepares/admits one fixed batch; no stores, network or model services."""
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
from evaluation import task_business_regression as business, task_observation_recheck as module
from evaluation import task_transport
from evaluation.intent_transport import canonical_payload_hash


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PROOF = {"kind": "controlled-fixture"}
ORIGINAL_PRIOR = module.prior_proof
ORIGINAL_ARTIFACTS = module._artifacts


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("observation preparation tests must not contact services")
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
    monkeypatch.setattr(business.engine, "_dependencies", lambda: {"controlled-runtime": "1"})
    monkeypatch.setattr(business.engine, "offline_data_proof", lambda: {"kind": "controlled-offline-proof"})
    monkeypatch.setattr(business, "_artifacts", lambda: {"core/controlled.py": "a" * 64})
    monkeypatch.setattr(module, "_artifacts", lambda: {"core/controlled.py": "a" * 64, "evaluation/v5.py": "b" * 64})
    monkeypatch.setattr(business, "prior_proof", lambda: {"controlled-v3": "c" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-v4": "d" * 64, "old_allowance_reused": False,
        "business_source_sha256": "e" * 64, "business_source_unchanged": True,
        "question_and_grading_unchanged": True, "model_and_parameters_unchanged": True})
    monkeypatch.setattr(business.repair, "data_binding", lambda: canonical_payload_hash(PROOF))
    return SimpleNamespace(config_reads=reads)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def prepared(tmp_path):
    trial = module.ObservationRecheck(tmp_path / "v5")
    return trial, asyncio.run(trial.prepare())


def approved(trial):
    return trial.authorize("Controlled test permission; not a real-send allowance.", confirmed=True)


def old_parameters():
    return {"artifacts": business._artifacts(), "model": base.MODEL, "base_url": BASE,
        "limits": deepcopy(business.LIMITS),
        "dependencies": business.engine._dependencies(), "http_limits": business.engine.HTTP_LIMITS,
        "phase_parameters": json.loads(json.dumps(business.engine.PHASE_PARAMETERS)),
        "agent_parameters": json.loads(json.dumps(business.engine.AGENT_PARAMETERS)),
        "effective_environment": business.effective_environment(), "policy": deepcopy(business.POLICY),
        "review_policy": deepcopy(business.REVIEW_POLICY),
        "cases_sha256": canonical_payload_hash({"cases": business.cases()})}


def test_new_version_freezes_same_business_questions_parameters_and_rubric_with_zero_sends(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-task-business-protocol-v5-observation"
    assert p["batch"] == module.BATCH and p["candidate"] == module.CANDIDATE
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 312
    assert p["limits"] == business.LIMITS and p["plan"] == business.fixed_plan()
    assert p["cases_sha256"] == canonical_payload_hash({"cases": business.cases()})
    assert p["http_limits"] == business.engine.HTTP_LIMITS
    assert p["effective_environment"]["ECHOMIND_LLM_TIMEOUT_S"] == 15
    assert p["review_policy"] == business.REVIEW_POLICY
    assert p["policy"] == {**business.POLICY, "observation": module.OBSERVATION_POLICY}
    assert len(p["initial_intent_requests"]) == 12 and sum(len(v["turn_ids"]) for v in p["plan"]) == 13
    assert trial.verify() == p and trial.ledger(p).status()["attempts"] == 0
    assert not any(isolated.config_reads)


def test_default_v4_is_still_inline_and_only_fixed_v5_selects_threaded_transport(tmp_path, monkeypatch):
    old, new = [], []
    monkeypatch.setattr(business.engine, "TaskHTTPTransport", lambda *a, **k: old.append((a, k)) or "old-inline")
    monkeypatch.setattr(task_transport, "ThreadedTaskHTTPTransport", lambda *a, **k: new.append((a, k)) or "new-threaded", raising=False)
    assert business.BusinessRegression(tmp_path / "v4").make_transport("controlled") == "old-inline"
    assert module.ObservationRecheck(tmp_path / "v5").make_transport("controlled") == "new-threaded"
    assert len(old) == len(new) == 1
    with pytest.raises(TypeError): module.ObservationRecheck(tmp_path / "bad", transport="arbitrary")
    with pytest.raises(base.TrialError): module.ObservationRecheck(business.ROOT)
    assert business.BusinessRegression.protocol_schema == "serveflow-task-business-protocol-v4"
    assert business.BusinessRegression.authorization_schema == "serveflow-task-business-authorization-v4"


def test_v4_authorization_never_admits_v5_or_reuses_old_allowance(tmp_path, monkeypatch):
    old = business.BusinessRegression(tmp_path / "v4")
    old_p = asyncio.run(old.prepare())
    old.authorize("controlled old permission", confirmed=True)
    new, p = prepared(tmp_path)
    assert old.ledger(old_p).initial_chain_hash != new.ledger(p).initial_chain_hash
    shutil.copyfile(old.root / "authorization.json", new.root / "authorization.json")
    (new.root / "authorization.json").chmod(0o600)
    store = AsyncMock()
    monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError, match="authorization_changed"):
        asyncio.run(new.execute())
    store.assert_not_called()
    assert new.ledger(p).status()["attempts"] == 0


@pytest.mark.parametrize("damage", ["policy", "source", "history", "data"])
def test_changed_freeze_stops_before_models_or_stores(tmp_path, monkeypatch, damage):
    trial, p = prepared(tmp_path)
    approved(trial)
    before = tree(trial.root)
    if damage == "policy": monkeypatch.setitem(module.OBSERVATION_POLICY, "version", "changed")
    elif damage == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"changed": "f" * 64})
    elif damage == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": "f" * 64})
    else: monkeypatch.setattr(business.repair, "data_binding", lambda: "f" * 64)
    store = AsyncMock()
    monkeypatch.setattr(business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    store.assert_not_called()
    assert trial.ledger(p).status()["attempts"] == 0 and tree(trial.root) == before


def test_fresh_v5_authorization_is_protocol_root_and_genesis_bound_and_cannot_be_rewritten(tmp_path):
    trial, p = prepared(tmp_path)
    value = approved(trial)
    assert value["schema_version"] == "serveflow-task-business-authorization-v5-observation"
    assert value["max_requests"] == 312 and value["plan"] == business.fixed_plan()
    assert value["initial_chain_hash"] == trial.ledger(p).initial_chain_hash
    assert value["protocol_sha256"] == bench.sha256(trial.root / "protocol.json")
    assert trial._approval(p) == value
    with pytest.raises(FileExistsError): approved(trial)


def test_pending_v5_attempt_is_not_refunded_reinitialized_or_replayed(tmp_path):
    trial, p = prepared(tmp_path)
    approved(trial)
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("e01", lease=lease)
        ledger.reserve("e01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    with pytest.raises(business.accounting.TaskBudgetError, match="pending"):
        asyncio.run(trial.execute(delegate_factory=lambda _: httpx.MockTransport(lambda _: None),
            controlled_data=SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=AsyncMock())))
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())
    assert ledger.status()["attempts"] == 1 and len(ledger.status()["pending"]) == 1
    assert tree(trial.root) == before


def test_report_keeps_twelve_case_denominator_and_v4_semantic_rubric(tmp_path):
    trial, _ = prepared(tmp_path)
    report = trial.report()
    assert report["schema_version"] == "serveflow-task-business-report-v5-observation"
    assert report["batch"] == module.BATCH and report["review_policy"] == business.REVIEW_POLICY
    assert report["grading_version"] == "task-business-v4" and report["observation_policy"] == module.OBSERVATION_POLICY
    assert report["unchanged_business_proof"]["business_source_unchanged"] is True
    assert report["planned_cases"] == 12 and report["planned_turns"] == 13
    assert report["business_counts"]["not_executed"] == report["legacy_judge_gated_counts"]["not_executed"] == 12
    assert report["authorized_requests"] == report["reserved_requests"] == 0


@pytest.mark.parametrize("damage", ["source", "new_production_file", "model", "endpoint", "budget", "network_timeout", "phase_parameters", "rubric", "questions"])
def test_unchanged_business_proof_rejects_changes_outside_observation_scope(monkeypatch, damage):
    p = old_parameters()
    assert module.business_source_proof(p)["business_source_unchanged"] is True
    if damage == "source": monkeypatch.setattr(business, "_artifacts", lambda: {"core/controlled.py": "f" * 64})
    elif damage == "new_production_file": monkeypatch.setattr(business, "_artifacts", lambda: {**p["artifacts"], "core/new.py": "f" * 64})
    elif damage == "model": p["model"] = "other-model"
    elif damage == "endpoint": p["base_url"] = "https://other-beijing-space.example.invalid/apps/anthropic"
    elif damage == "budget": p["limits"] = {**p["limits"], "physical_requests": 313}
    elif damage == "network_timeout": p["http_limits"] = {**p["http_limits"], "request_deadline_s": 20}
    elif damage == "phase_parameters": p["phase_parameters"] = {**p["phase_parameters"], "intent": [100, .1]}
    elif damage == "rubric": p["review_policy"]["version"] = "other-version"
    else: p["cases_sha256"] = "f" * 64
    with pytest.raises(base.TrialError, match="business_(source|parameters)_changed"):
        module.business_source_proof(p)


def test_new_module_and_tests_are_frozen_but_not_added_to_old_protocol(monkeypatch):
    monkeypatch.setattr(bench, "sha256", lambda _: "e" * 64)
    artifacts = ORIGINAL_ARTIFACTS()
    assert artifacts["core/controlled.py"] == "a" * 64
    assert {"evaluation/task_observation_recheck.py", "tests/test_task_observation_recheck.py",
            "tests/test_task_threaded_admission.py"} <= set(artifacts)


def test_v4_archive_link_checks_original_tree_git_sources_and_all_seals_without_writes(tmp_path, monkeypatch):
    root = tmp_path / "old-v4"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(module, "OLD_ROOT", root)
    older = business.prior_proof()
    p = {**old_parameters(), "schema_version": "serveflow-task-business-protocol-v4", "batch": business.BATCH,
        "git_head": module.OLD_EXECUTION_COMMIT, "root_binding_sha256": base._root_hash(root),
        "prior_proof": older, "authorized_model_requests": 0, "plan": business.fixed_plan()}
    base._write(root, "protocol.json", p)
    genesis = "d" * 64
    approval = {"schema_version": "serveflow-task-business-authorization-v4", "protocol_sha256": bench.sha256(root / "protocol.json"),
        "max_requests": 312, "root_binding_sha256": base._root_hash(root), "initial_chain_hash": genesis, "plan": p["plan"]}
    base._write(root, "authorization.json", approval)
    ledger = root / "ledger"; ledger.mkdir(mode=0o700)
    base._write(ledger, "journal.jsonl", {"controlled": "historical journal"})
    operations, sealed_cases = {}, {}
    for number in range(1, 13):
        name = f"physical-{number:03d}.json"
        base._write(root, name, {"controlled": number})
        operations[str(number)] = {"attempt": number, "output_sha256": bench.sha256(root / name)}
    for key in ("e01", "e02"):
        name = f"case-{key}.json"
        base._write(root, name, {"controlled": key})
        sealed_cases[key] = {"artifact_sha256": bench.sha256(root / name)}
    for name, attribute in (("protocol.json", "OLD_PROTOCOL"), ("authorization.json", "OLD_AUTHORIZATION"),
                            ("ledger/journal.jsonl", "OLD_JOURNAL")):
        monkeypatch.setattr(module, attribute, bench.sha256(root / name))
    monkeypatch.setattr(module, "OLD_TREE", module._tree(root))
    sources = []
    monkeypatch.setattr(module.history, "_archived_sources", lambda protocol, key: sources.append((protocol, key)))
    state = {"attempts": 12, "pending": [], "paused": True, "active_case": "e03",
             "completed_cases": ["e01", "e02"], "chain_hash": module.OLD_CHAIN}
    fake = SimpleNamespace(status=lambda: state, results=lambda: {"operations": operations, "cases": sealed_cases},
                           initial_chain_hash=genesis)
    def readonly_budget(*_args, **kwargs):
        assert not kwargs.get("initialize")
        return fake
    monkeypatch.setattr(business, "BusinessTaskBudget", readonly_budget)
    before = tree(root)
    proof = ORIGINAL_PRIOR()
    assert proof["reserved_requests"] == proof["sealed_requests"] == 12
    assert proof["business_counts"] == {"passed": 2, "failed": 0, "unknown": 1, "not_executed": 9}
    assert proof["old_scores_recomputed"] is False and proof["old_allowance_reused"] is False
    assert proof["new_authorized_requests"] == 0 and proof["business_source_unchanged"] is True
    assert sources == [(p, "artifacts")] and tree(root) == before
    (root / "physical-012.json").write_text('{"changed":"old cancellation"}')
    with pytest.raises(base.TrialError, match="old_tree_changed"): ORIGINAL_PRIOR()
