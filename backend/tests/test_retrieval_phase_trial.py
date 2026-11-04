"""Real SDK serialization, private counters and local controlled RAG only."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.model_deadline import await_model, eval_deadlines, EvaluationDeadlinePolicy
from evaluation import benchmark as bench, intent_trial as base
from evaluation import retrieval_phase_trial as module, retrieval_trial as original
from evaluation.intent_transport import TrialTransportError
from evaluation.task_supervisor import SupervisorError, supervise


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
KEY = "controlled-not-a-real-secret"
DATA = {key: "controlled-" + key for key in original.DATA_FIELDS}
ORIGINAL_ARTIFACTS = module.artifacts
ORIGINAL_PRIOR = module.prior_proof


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def deny(*args, **kwargs): raise AssertionError("retrieval-phase tests forbid network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    reads = []
    def config(*, secret=False):
        reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": KEY} if secret else {})}
    monkeypatch.setattr(base, "_configuration", config)
    monkeypatch.setattr(original, "artifacts", lambda: {"controlled-unchanged-source": "a" * 64})
    monkeypatch.setattr(module, "artifacts", lambda: {"controlled-current-source": "b" * 64})
    monkeypatch.setattr(original, "dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(original, "data_identity", lambda: deepcopy(DATA))
    monkeypatch.setattr(original, "baseline_binding", lambda: {"sealed-baseline": "c" * 64})
    monkeypatch.setattr(module, "prior_proof", lambda: {"old-zero-send": "d" * 64, "old_allowance_reused": False})
    old = tmp_path / "old-v1"; old.mkdir(mode=0o700)
    baseline = tmp_path / "sealed-baseline"; baseline.mkdir(mode=0o700)
    write(baseline / "report.json", {"by_split": {"development": {}, "validation": {}}})
    monkeypatch.setattr(module, "OLD_ROOT", old)
    monkeypatch.setattr(original, "BASELINE_ROOT", baseline)
    return reads


def write(path, value):
    path.write_text(json.dumps(value)); path.chmod(0o600)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def prepared(tmp_path):
    trial = module.RetrievalPhaseTrial(tmp_path / "new-phase")
    p = asyncio.run(trial.prepare())
    old = deepcopy(p); old.update(schema_version=original.RetrievalTrial.protocol_schema,
        batch=original.BATCH, limits=deepcopy(original.LIMITS), policy=deepcopy(original.POLICY))
    write(module.OLD_ROOT / "protocol.json", old)
    return trial, p


def approve(trial):
    return trial.authorize("Controlled fixture allowance only, no live authorization", confirmed=True)


def knowledge(*, count=5, close=None):
    rows = [{"document_id": hashlib.sha256(json.dumps(["title", d["title"]], ensure_ascii=True).encode()).hexdigest(),
        "title": d["title"], "content": d["content"], "version": hashlib.sha256(d["content"].encode()).hexdigest(),
        "chunk": 0, "score": .5} for d in original.task_data._policies()[:count]]
    async def search(params, context): return deepcopy(rows[:params.get("top_k", 5)])
    return SimpleNamespace(proof=deepcopy(DATA), kb=SimpleNamespace(search_handler=search), close=close or (lambda: None))


def response(text):
    return httpx.Response(200, json={"id": "controlled-response", "type": "message", "role": "assistant",
        "model": base.MODEL, "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
        "provider_debug": "not-a-whitelisted-field-do-not-persist"}, headers={"request-id": "controlled-id"})


def test_original_defaults_are_unchanged_and_new_preparation_authorizes_zero(tmp_path, isolated):
    old = original.RetrievalTrial(tmp_path / "unused-original")
    assert old.sdk_timeout_s() == 8 and old.limits() == original.LIMITS
    assert old.execution_policy() == original.POLICY and old.outer_deadline_s() == 900
    assert old.protocol_schema == "serveflow-retrieval-protocol-v1"
    trial, p = prepared(tmp_path)
    assert p["authorized_model_requests"] == 0 and p["proposed_model_requests"] == 48
    assert p["operations"] == list(original.OPERATIONS) and len(p["initial_requests"]) == 24
    assert p["limits"]["stage_network_deadlines_s"] == {"rewrite": 10, "rerank": 30}
    assert trial.sdk_timeout_s() == p["limits"]["request_deadline_s"] == 30
    assert trial.outer_deadline_s() == 3600 and trial.verify() == p
    assert trial.report()["model_requests"] == 0 and not any(isolated)
    for case, row in zip(original.cases(), p["initial_requests"]):
        text = json.dumps(row["payload"], ensure_ascii=False)
        assert case["input"]["message"] in text and case["expected"]["rationale"] not in text
        assert "relevant_titles" not in text


def test_all_48_operations_share_one_counter_and_preserve_full_actual_payloads(tmp_path):
    trial, p = prepared(tmp_path); approve(trial)
    sent, closed = [], []
    def factory(frame):
        def handle(request):
            sent.append((frame, json.loads(request.content)))
            assert trial.budget(p).status()["attempts"] == len(sent)
            wire_hash = hashlib.sha256(request.content).hexdigest()
            actual = module.read_request(trial.root, wire_hash)
            assert actual["wire_text"].encode() == request.content
            assert actual["payload"] == sent[-1][1]
            assert request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), float(module.PHASE_SECONDS[frame[1]]))
            return response('["受控改写"]' if frame[1] == "rewrite" else '[2,1,0,4,3]')
        return httpx.MockTransport(handle)
    r = asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge(close=lambda: closed.append(True))))
    assert len(sent) == r["model_requests"] == 48 and closed == [True]
    assert [f"{case}:{stage}:1" for (case, stage), _ in sent] == list(original.OPERATIONS)
    assert r["paired_comparison_ready"] and r["owned_collection_cleanup_complete"]
    assert r["evidence_kinds"] == ["controlled-transport"]
    assert bench.read_json(trial.root / "actual-data-proof.json")["proof"] == DATA
    for number in range(1, 49):
        row = bench.read_json(trial.root / f"physical-{number:03d}.json")
        assert row["sdk_limit_s"] == 30
        assert row["network_limit_s"] == row["active_stage_limit_s"] == (10 if number % 2 else 30)
        assert row["response_received"] and row["response"]["content"][0]["text"]
        assert "raw_response_text" not in row
    before = snapshot(trial.root)
    asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge()))
    assert snapshot(trial.root) == before and len(sent) == 48
    assert KEY not in "\n".join(p.read_text() for p in trial.root.rglob("*") if p.is_file())
    assert "not-a-whitelisted-field-do-not-persist" not in "\n".join(p.read_text() for p in trial.root.rglob("*") if p.is_file())


@pytest.mark.parametrize("damage", ["source", "history", "model", "clock", "old_input", "old_params", "old_data", "approval"])
def test_binding_damage_stops_before_secrets_native_builder_and_any_claim(tmp_path, monkeypatch, isolated, damage):
    trial, p = prepared(tmp_path); approve(trial)
    ledger = trial.budget(p)
    if damage == "source": monkeypatch.setattr(module, "artifacts", lambda: {"changed": "e" * 64})
    elif damage == "history": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    elif damage.startswith("old_"):
        value = bench.read_json(module.OLD_ROOT / "protocol.json")
        if damage == "old_input": value["initial_requests"][0]["request_bytes_sha256"] = "f" * 64
        elif damage == "old_params": value["parameters"]["rewrite"] = [500, .5]
        else: value["data_identity"] = {"changed": True}
        write(module.OLD_ROOT / "protocol.json", value)
    else:
        name = "authorization.json" if damage == "approval" else "protocol.json"
        value = bench.read_json(trial.root / name)
        if damage == "model": value["model"] = "other"
        elif damage == "clock": value["limits"]["stage_network_deadlines_s"]["rerank"] = 60
        else: value["schema_version"] = original.RetrievalTrial.authorization_schema
        write(trial.root / name, value)
    build = AsyncMock(); monkeypatch.setattr(original, "build_owned_knowledge", build)
    isolated.clear()
    with pytest.raises(Exception): asyncio.run(trial.execute())
    assert not any(isolated); build.assert_not_called()
    assert ledger.status()["attempts"] == 0


@pytest.mark.parametrize("failure", ["timeout", "auth", "cancel"])
def test_failed_or_cancelled_send_counts_and_stops_without_rerank_or_retry(tmp_path, failure):
    trial, p = prepared(tmp_path); approve(trial)
    sent, closed = [], []
    def factory(frame):
        async def handle(request):
            sent.append(frame)
            if failure == "timeout": raise httpx.ReadTimeout("controlled")
            if failure == "auth": return httpx.Response(401)
            raise asyncio.CancelledError
        return httpx.MockTransport(handle)
    with pytest.raises(BaseException):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge(close=lambda: closed.append(True))))
    state = trial.budget(p).status()
    assert sent == [("r01", "rewrite")] and state["attempts"] == 1 and state["paused"]
    row = bench.read_json(trial.root / "physical-001.json")
    assert row["delegate_entered"] and row["error_code"] == ("cancelled" if failure == "cancel" else failure)
    assert closed == [True] and trial.report()["model_requests"] == 1
    before = snapshot(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge()))
    assert snapshot(trial.root) == before and len(sent) == 1


def test_unknown_claim_blocks_restart_and_new_prepare(tmp_path):
    trial, p = prepared(tmp_path); approve(trial)
    trial.budget(p).reserve(original.OPERATIONS[0], "a" * 64)
    before = snapshot(trial.root)
    with pytest.raises(Exception):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: response("[]")),
                                  controlled_knowledge=knowledge()))
    assert trial.budget(p).status()["attempts"] == 1
    # The first rejected execute may add a stop receipt, never clear the claim.
    assert all((trial.root / name).read_bytes() == raw for name, raw in before.items())
    with pytest.raises(base.TrialError, match="already_exists"): asyncio.run(trial.prepare())


def test_full_payload_tampering_is_rejected_even_when_physical_output_unchanged(tmp_path):
    trial, p = prepared(tmp_path); approve(trial)
    def factory(frame):
        def handle(request):
            if frame[1] == "rerank": raise httpx.ReadTimeout("controlled stop")
            return response('["受控改写"]')
        return httpx.MockTransport(handle)
    with pytest.raises(Exception): asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge()))
    row = bench.read_json(trial.root / "physical-001.json")
    path = trial.root / ("request-" + row["request_bytes_sha256"] + ".json")
    value = bench.read_json(path); value["wire_text"] += " "
    write(path, value)
    with pytest.raises(base.TrialError, match="request_changed"): trial.verify()


def test_guard_refuses_old_sdk_eight_seconds_before_any_claim(tmp_path):
    root = tmp_path / "guard"; root.mkdir(mode=0o700)
    admitted, closed, delegated = [], [], []
    class ControlledDelegate(httpx.MockTransport):
        async def aclose(self): closed.append(True)
    async def run():
        payload = {"model": base.MODEL}
        request = httpx.Request("POST", BASE + "/v1/messages", json=payload,
            extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), 8.)})
        with original.stage("r01", "rewrite"):
            guard = module._PhaseRetrievalGuard(BASE, module.canonical_payload_hash(payload), lambda: admitted.append(True),
                request=request, root=root, transport=ControlledDelegate(lambda request: delegated.append(True)))
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await await_model(guard.handle_async_request(request), timeout_s=10)
    with pytest.raises(TrialTransportError): asyncio.run(run())
    assert not admitted and not list(root.iterdir()) and closed == [True] and not delegated


def test_cancel_during_admission_reaps_thread_and_never_sends_after_reservation(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); approve(trial)
    entered, release = threading.Event(), threading.Event()
    real_reserve = original.RetrievalBudget.reserve
    sent, closed = [], []
    def delayed_reserve(self, *args):
        value = real_reserve(self, *args)
        entered.set(); assert release.wait(2)
        return value
    monkeypatch.setattr(original.RetrievalBudget, "reserve", delayed_reserve)
    async def run():
        task = asyncio.create_task(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(
            lambda request: sent.append(True)), controlled_knowledge=knowledge(close=lambda: closed.append(True))))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel(); task.cancel()
            await asyncio.sleep(.015)
            assert not task.done() and not sent and not closed
        finally:
            release.set()
        with pytest.raises(BaseException): await task
    asyncio.run(run())
    state = trial.budget(p).status()
    assert state["attempts"] == 1 and state["paused"] and not state["pending"]
    row = bench.read_json(trial.root / "physical-001.json")
    assert not row["delegate_entered"] and row["error_code"] == "cancelled"
    assert closed == [True] and not sent


def test_admission_deadline_after_reservation_blocks_delegate_and_keeps_claim(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); approve(trial)
    real_reserve = original.RetrievalBudget.reserve
    def delayed_reserve(self, *args):
        value = real_reserve(self, *args); time.sleep(.04); return value
    monkeypatch.setattr(original.RetrievalBudget, "reserve", delayed_reserve)
    monkeypatch.setattr(trial, "execution_context", lambda: eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=.005)))
    sent = []
    with pytest.raises(Exception):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: sent.append(True)),
                                  controlled_knowledge=knowledge()))
    assert not sent and trial.budget(p).status()["attempts"] == 1
    row = bench.read_json(trial.root / "physical-001.json")
    assert not row["delegate_entered"] and row["error_code"] == "timeout"
    assert row["stage_deadline_reason"] == "overhead"


def test_cancel_before_success_seal_preserves_observed_response_and_known_usage(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); approve(trial)
    original_record = module.RetrievalPhaseTransport.record
    async def cancelled_record(self, *args):
        if args[3] != "execution_error": raise asyncio.CancelledError
        return await original_record(self, *args)
    monkeypatch.setattr(module.RetrievalPhaseTransport, "record", cancelled_record)
    sent = []
    def factory(frame):
        def handle(request): sent.append(frame); return response('["受控改写"]')
        return httpx.MockTransport(handle)
    with pytest.raises(BaseException): asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge()))
    row = bench.read_json(trial.root / "physical-001.json")
    assert sent == [("r01", "rewrite")] and row["delegate_entered"] and row["response_received"]
    assert row["error_code"] == "cancelled" and row["response"]["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert row["response"]["content"] == [{"type": "text", "text": '["受控改写"]'}]
    assert row["response_sha256"] and trial.report()["usage"]["input_tokens"]["known_sum"] == 5
    assert trial.budget(p).status()["paused"] and not trial.report()["paired_comparison_ready"]


def test_cancel_during_seal_waits_and_cannot_overwrite_already_sealed_received_output(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path); approve(trial)
    entered, release = threading.Event(), threading.Event()
    real_seal = module.RetrievalPhaseTransport.seal
    def delayed_seal(self, *args):
        real_seal(self, *args)
        entered.set(); assert release.wait(2)
    monkeypatch.setattr(module.RetrievalPhaseTransport, "seal", delayed_seal)
    sent = []
    def factory(frame):
        def handle(request): sent.append(frame); return response('["受控改写"]')
        return httpx.MockTransport(handle)
    async def run():
        task = asyncio.create_task(trial.execute(delegate_factory=factory, controlled_knowledge=knowledge()))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            sealed = (trial.root / "physical-001.json").read_bytes()
            task.cancel(); await asyncio.sleep(.015)
            assert not task.done()
        finally: release.set()
        with pytest.raises(BaseException): await task
        assert (trial.root / "physical-001.json").read_bytes() == sealed
    asyncio.run(run())
    row = bench.read_json(trial.root / "physical-001.json")
    assert row["outcome"] == "received" and row["response"]["usage"]["input_tokens"] == 5
    assert len(sent) == trial.budget(p).status()["attempts"] == 1
    assert (trial.root / "stopped.json").exists() and not trial.report()["paired_comparison_ready"]


@pytest.mark.parametrize("incomplete", ["last_stage", "cleanup"])
def test_ready_flag_requires_all_operations_and_successful_owned_cleanup(tmp_path, monkeypatch, incomplete):
    trial = module.RetrievalPhaseTrial(tmp_path / "ready"); trial.root.mkdir(mode=0o700)
    write(trial.root / "protocol.json", {})
    report = {"paired_comparison_ready": True, "stopped": False, "model_requests": 48,
              "ledger": {"paused": False, "pending": []}}
    results = {key: {"output_sha256": "a" * 64} for key in original.OPERATIONS}
    if incomplete == "last_stage": results.pop(original.OPERATIONS[-1]); report["model_requests"] = 47
    monkeypatch.setattr(original.RetrievalTrial, "report", lambda self: deepcopy(report))
    monkeypatch.setattr(trial, "verify", lambda: {})
    monkeypatch.setattr(trial, "budget", lambda p: SimpleNamespace(results=lambda: results))
    module.write_audit(trial.root, "owned-collection-cleanup.json", {"owned_collection_cleanup_complete": incomplete != "cleanup"})
    assert not trial.report()["paired_comparison_ready"]


@pytest.mark.parametrize("damage", ["body", "seal", "unsealed"])
def test_actual_data_and_cleanup_audits_reject_changes_and_partial_receipts(tmp_path, damage):
    root = tmp_path / "audit"; root.mkdir(mode=0o700)
    write(root / "protocol.json", {"controlled": True})
    value = {"controlled": "owned-data"}
    module.write_audit(root, "actual-data-proof.json", value)
    assert module.read_audit(root, "actual-data-proof.json") == value
    if damage == "body": write(root / "actual-data-proof.json", {"controlled": "changed"})
    elif damage == "seal": write(root / "actual-data-proof-seal.json", {})
    else: (root / "actual-data-proof-seal.json").unlink()
    with pytest.raises(Exception): module.write_audit(root, "actual-data-proof.json", value)


def test_invalid_request_identity_cannot_be_used_as_a_file_path(tmp_path):
    with pytest.raises(base.TrialError, match="hash_invalid"):
        module.read_request(tmp_path, "../not-a-hash")


def test_original_guard_cannot_accept_longer_deadline():
    with pytest.raises(TrialTransportError):
        module.GuardedIntentTransport(BASE, "a" * 64, lambda: None, timeout_s=30)


def test_old_tree_damage_fails_without_old_current_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_tree", lambda root: "changed")
    monkeypatch.setattr(original.RetrievalTrial, "verify", lambda self: pytest.fail("must not verify old current code"))
    with pytest.raises(base.TrialError, match="old_preparation_changed"): ORIGINAL_PRIOR()


def test_frozen_files_include_clocks_transport_new_runner_and_tests(monkeypatch):
    monkeypatch.setattr(original, "artifacts", lambda: {"evaluation/task_supervisor.py": "a" * 64})
    monkeypatch.setattr(bench, "sha256", lambda path: "b" * 64)
    assert set(module._EXTRA) <= ORIGINAL_ARTIFACTS().keys()


def test_only_fixed_new_worker_can_use_larger_parent_ceiling(tmp_path):
    command = [sys.executable, "-B", "-u", "-m", "evaluation.retrieval_phase_trial", "_worker"]
    with pytest.raises(SupervisorError, match="deadline_invalid"):
        supervise(command, root=tmp_path, deadline_s=3601)
    command[-2] = "evaluation.retrieval_trial"
    with pytest.raises(SupervisorError, match="deadline_invalid"):
        supervise(command, root=tmp_path, deadline_s=901)
