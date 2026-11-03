"""Controlled trial/accounting checks; no model or business service access."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.model_deadline import await_model
from evaluation import benchmark as bench, intent_trial as base
from evaluation import quality_trial as module, quality_scoring
from evaluation.task_transport import Frame, context
from evaluation.task_budget import TaskBudgetError


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
PROOF = {"kind": "controlled-fixture"}
ORIGINAL_PRIOR = module.prior_proof
ORIGINAL_ARTIFACTS = module._artifacts


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def deny(*args, **kwargs): raise AssertionError("quality trial test cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for key in tuple(os.environ):
        if key.startswith("ECHOMIND_"): monkeypatch.delenv(key)
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    secret_reads = []
    def configuration(*, secret=False):
        secret_reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": "controlled-not-a-key"} if secret else {})}
    async def capture(case, endpoint):
        assert endpoint == BASE
        payload = {"max_tokens": 256, "messages": [{"role": "user", "content": case["input"]["message"]}],
                   "model": base.MODEL, "temperature": .1, "thinking": {"type": "disabled"}}
        return {"case_id": case["id"], "operation_id": "controlled", "input_sha256": module.canonical_payload_hash(case["input"]),
                "payload": payload, "payload_sha256": module.canonical_payload_hash(payload),
                "request_bytes_sha256": hashlib.sha256(httpx.Request("POST", BASE, json=payload).content).hexdigest()}
    monkeypatch.setattr(base, "_configuration", configuration)
    monkeypatch.setattr(base, "_capture", capture)
    monkeypatch.setattr(module.business.engine, "_dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(module.business.engine, "offline_data_proof", lambda: {"controlled": True})
    monkeypatch.setattr(module, "_artifacts", lambda: {"core/controlled.py": "a" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"controlled-history": "b" * 64})
    monkeypatch.setattr(module, "_design_proof", lambda: {"controlled-design": "c" * 64})
    monkeypatch.setattr(module.business.repair, "data_binding", lambda: module.canonical_payload_hash(PROOF))
    return secret_reads


def prepared(tmp_path):
    trial = module.QualityTrial(tmp_path / "quality")
    return trial, asyncio.run(trial.prepare())


def approve(trial): return trial.authorize("Controlled test permission; grants no real model authorization", confirmed=True)


def tree(root): return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_prepare_and_zero_report_bind_48_rotation_and_same_initial_payloads(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["schema_version"] == "serveflow-quality-protocol-v1"
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 1248
    assert len(p["plan"]) == 48 and sum(len(c["turn_ids"]) for c in p["plan"]) == 52
    assert p["policy"]["outer_process_deadline_s"] == 3600
    assert p["http_limits"]["request_deadline_s"] == 8
    assert p["policy"]["stage_clock"]["model_stage_budgets"]["intent"] == 8
    assert p["plan"] == module.fixed_plan()
    assert p["initial_intent_requests"][0]["case_id"] == "A-e01"
    for cid in module.design.CASE_IDS:
        same = [r for r in p["initial_intent_requests"] if r["source_case_id"] == cid]
        assert len(same) == 4 and len({r["request_bytes_sha256"] for r in same}) == 1
        for r in same:
            raw = json.dumps(r["payload"])
            assert r["case_id"] not in raw
            assert "required_facts" not in raw and "expected" not in raw
    assert trial.verify() == p and not any(isolated)
    report = trial.report()
    assert report["actual_requests"] == report["authorized_requests"] == 0
    assert all(arm["correct_handling"]["counts"]["not_executed"] == 12 for arm in report["arms"].values())
    assert all(pair["full_plan_delta_percentage_points"] is None for pair in report["pairs"].values())


def test_authorization_is_new_protocol_root_and_genesis_bound_and_not_overwritable(tmp_path):
    trial, p = prepared(tmp_path)
    with pytest.raises(base.TrialError, match="explicit_authorization"): trial.authorize("not confirmed")
    a = approve(trial); before = tree(trial.root)
    assert a["max_requests"] == 1248 and a["plan"] == p["plan"]
    assert a["initial_chain_hash"] == trial.ledger(p).initial_chain_hash
    assert a["protocol_sha256"] == bench.sha256(trial.root / "protocol.json")
    assert a["root_binding_sha256"] == base._root_hash(trial.root)
    with pytest.raises(FileExistsError): approve(trial)
    assert tree(trial.root) == before


@pytest.mark.parametrize("change", ["source", "design", "history", "limits", "clock", "order", "model", "data", "initial_arm_input"])
def test_changed_freeze_stops_before_store_credentials_or_send(tmp_path, monkeypatch, isolated, change):
    trial, p = prepared(tmp_path); approve(trial)
    if change == "source": monkeypatch.setattr(module, "_artifacts", lambda: {"new.py": "e" * 64})
    elif change == "design": monkeypatch.setattr(module, "_design_proof", lambda: {"changed": True})
    elif change == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    elif change == "data": monkeypatch.setattr(module.business.repair, "data_binding", lambda: "f" * 64)
    else:
        changed = deepcopy(p)
        if change == "limits": changed["limits"]["physical_requests"] += 1
        elif change == "clock": changed["policy"]["stage_clock"]["network_deadline_s"] = 60
        elif change == "order": changed["plan"].reverse()
        elif change == "model": changed["model"] = "another-model"
        else:
            changed["initial_intent_requests"][1]["payload"]["messages"][0]["content"] = "easier question"
            changed["initial_intent_requests"][1]["payload_sha256"] = module.canonical_payload_hash(changed["initial_intent_requests"][1]["payload"])
        (trial.root / "protocol.json").write_text(json.dumps(changed))
    store = AsyncMock(); monkeypatch.setattr(module.business.engine, "build_live_data", store)
    with pytest.raises(base.TrialError): asyncio.run(trial.execute())
    store.assert_not_called(); assert not any(isolated)


def test_old_authorization_and_pending_attempt_cannot_restore_budget(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); a = approve(trial)
    a["schema_version"] = "serveflow-task-business-authorization-v6-stage-clock"
    (trial.root / "authorization.json").write_text(json.dumps(a))
    with pytest.raises(base.TrialError, match="authorization_changed"): asyncio.run(trial.execute())
    a["schema_version"] = trial.authorization_schema
    (trial.root / "authorization.json").write_text(json.dumps(a))
    ledger = trial.ledger(p)
    with ledger.exclusive_run() as lease:
        ledger.start_case("A-e01", lease=lease)
        ledger.reserve("A-e01", "turn-1", "intent", None, "a" * 64, "b" * 64, lease=lease)
    before = tree(trial.root)
    # An unfinished claim remains charged; the run lease rejects continuation.
    with pytest.raises((base.TrialError, FileNotFoundError, TaskBudgetError)):
        asyncio.run(trial.execute())
    assert tree(trial.root) == before and ledger.status()["attempts"] == 1
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())


def fake_checks(case, artifact):
    return {"checks": {"execution_complete": artifact["execution_status"] != "failed", "tool_task_correct": artifact["execution_status"] != "failed"},
            "blockers": [], "mechanisms": {"fresh_read_each_relevant_turn": None, "exact_projection": None}}


def fake_result(case, status="success"):
    return {"schema_version": "serveflow-task-case-result-v1", "case_id": case["id"], "data_kind": "controlled-fixture",
        "turns": [{"turn_id": f"turn-{i}", "request_id": f"task-eval-{case['id']}-{i}", "question": q,
                   "result": {"status": status}, "judge_overall": 1., "judge": {"judge_failed": False}}
                  for i, q in enumerate(case["turns"], 1)],
        "deterministic": {"checks": {"execution_complete": status == "success"}, "blockers": []},
        "wall_latency_ms": 1., "actor_path_latency_ms": 1., "execution_status": status,
        "judge_ready": True, "semantic_review": None, "answer": "Controlled answer, not model quality evidence"}


def response(request):
    return httpx.Response(200, json={"id": "msg_controlled", "type": "message", "role": "assistant", "model": base.MODEL,
        "content": [{"type": "text", "text": "controlled"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1}})


def execution_fixture(tmp_path, monkeypatch, *, fail_first_semantic=False, failure_status=None):
    from evaluation import quality_arm_runtime
    trial, p = prepared(tmp_path); approve(trial)
    seen = []
    async def run(arm, case, sdk, data, *, ensure_healthy, frame_case_id):
        assert frame_case_id == arm + "-" + case["id"] and case["id"] in module.design.CASE_IDS
        seen.append((arm, case["id"], frame_case_id)); ensure_healthy()
        for i, question in enumerate(case["turns"], 1):
            with context(Frame(frame_case_id, f"turn-{i}", "intent", None)):
                await await_model(sdk.messages.create(model=base.MODEL, max_tokens=256, temperature=.1,
                    messages=[{"role": "user", "content": question}], extra_body={"thinking": {"type": "disabled"}}), timeout_s=8)
        return fake_result(case, "failed" if fail_first_semantic and len(seen) == 1 else "success")
    monkeypatch.setattr(quality_arm_runtime, "execute_arm_case", run)
    monkeypatch.setattr(quality_scoring, "shared_checks", fake_checks)
    closed = AsyncMock()
    data = SimpleNamespace(proof=PROOF, assert_healthy=lambda: None, close=closed)
    def delegate(frame):
        def handle(request):
            raw = request.content.decode(); digest = hashlib.sha256(request.content).hexdigest()
            persisted = module._read_request(trial.root, digest)
            assert persisted["wire_text"] == raw and "authorization" not in persisted
            # Reservation is durable before the physical delegate is entered.
            assert len(trial.ledger(p).status()["pending"]) >= 1
            assert frame.case_id not in raw and "required_facts" not in raw
            return httpx.Response(failure_status) if failure_status else response(request)
        return httpx.MockTransport(handle)
    return trial, p, seen, closed, data, delegate


def review_all(trial, *, bad_fact_logical=None):
    for arm, case, logical in module.arm_cases():
        path = trial.root / f"case-{logical}.json"
        if not path.exists(): continue
        trial.review({"case_id": logical, "artifact_sha256": bench.sha256(path), "reviewer": "controlled test reviewer",
            "method": "controlled-fixture", "dimensions": {k: not (logical == bad_fact_logical and k == "conditions_preserved") for k in module.design.DIMENSIONS},
            "reason": "Controlled accounting only, no semantic model claim", "blocking_errors": ["controlled wrong amount"] if logical == bad_fact_logical else [], "reading_burden": "some"})


def test_entire_fixed_order_uses_one_ledger_preserves_failures_and_does_not_leak_arms(tmp_path, monkeypatch):
    trial, p, seen, closed, data, delegate = execution_fixture(tmp_path, monkeypatch, fail_first_semantic=True)
    report = asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert len(seen) == 48 and report["actual_requests"] == report["sealed_requests"] == 52
    assert [v[2] for v in seen] == [row["case_id"] for row in p["plan"]]
    assert closed.await_count == 1 and not report["ledger"]["paused"]
    assert report["arms"]["A"]["correct_handling"]["counts"]["failed"] == 1
    assert len(list(trial.root.glob("request-*.json"))) == 13  # same initial questions; no arm leaked
    review_all(trial, bad_fact_logical="B-e02")
    final = trial.report()
    assert final["arms"]["A"]["correct_handling"]["numerator"] == 11
    assert final["arms"]["D"]["correct_handling"]["numerator"] == 12
    assert final["arms"]["B"]["correct_handling"]["numerator"] == 11
    assert final["arms"]["B"]["tool_tasks"]["numerator"] == 10
    assert final["arms"]["A"]["tool_tasks"]["numerator"] == 9
    assert not final["model_quality_evidence"]
    assert final["pairs"]["A_vs_D"]["only_right_passed"] == 1
    assert final["pairs"]["A_vs_D"]["full_plan_delta_percentage_points"] == pytest.approx(100 / 12)
    before = tree(trial.root)
    repeated = asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert repeated["actual_requests"] == 52 and len(seen) == 48
    assert tree(trial.root) == before


def test_network_execution_fault_counts_once_and_blocks_remaining_arms(tmp_path, monkeypatch):
    trial, p, seen, closed, data, delegate = execution_fixture(tmp_path, monkeypatch, failure_status=401)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    state = trial.ledger(p).status()
    assert state["attempts"] == 1 and state["paused"] and seen == [("A", "e01", "A-e01")]
    assert closed.await_count == 1
    assert (trial.root / "worker-stop.json").exists() and (trial.root / "stopped.json").exists()
    before = tree(trial.root)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    assert tree(trial.root) == before and trial.ledger(p).status()["attempts"] == 1


def test_payload_or_review_tampering_stops_verification(tmp_path, monkeypatch):
    trial, p, _, _, data, delegate = execution_fixture(tmp_path, monkeypatch)
    asyncio.run(trial.execute(delegate_factory=delegate, controlled_data=data))
    path = next(trial.root.glob("request-*.json"))
    value = json.loads(path.read_text()); value["wire_text"] += " "
    path.write_text(json.dumps(value))
    with pytest.raises(base.TrialError, match="request_changed"): trial.verify()


def test_frozen_code_includes_runtime_and_common_scorer(monkeypatch):
    monkeypatch.setattr(module.business, "_artifacts", lambda: {})
    monkeypatch.setattr(module.clock, "_artifacts", lambda: {})
    monkeypatch.setattr(bench, "sha256", lambda path: "a" * 64)
    paths = ORIGINAL_ARTIFACTS()
    assert {"evaluation/quality_trial.py", "evaluation/quality_budget.py", "evaluation/quality_scoring.py",
            "evaluation/quality_arm_runtime.py", "tests/test_quality_trial.py", "tests/test_quality_budget.py"} <= set(paths)
