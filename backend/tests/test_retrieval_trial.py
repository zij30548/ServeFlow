"""Original pipeline with real SDK serialization and controlled responses only."""
import asyncio
from copy import deepcopy
import hashlib
import fcntl
import json
import socket
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation import retrieval_trial as module, intent_trial as base, benchmark as bench
from evaluation.intent_transport import canonical_payload_hash, TrialTransportError
from evaluation.retrieval_budget import OPERATIONS


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"
KEY = "controlled-only-secret-never-persisted"
ORIGINAL_CAPTURE = module.capture_rewrite
DATA = {key: "controlled-" + key for key in module.DATA_FIELDS}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("retrieval tests prohibit all network services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    reads = []
    def config(*, secret=False):
        reads.append(secret)
        return {"model": base.MODEL, "base_url": BASE, **({"api_key": KEY} if secret else {})}
    monkeypatch.setattr(base, "_configuration", config)
    monkeypatch.setattr(module, "artifacts", lambda: {"controlled.py": "a" * 64})
    monkeypatch.setattr(module, "dependencies", lambda: {"controlled": "1"})
    monkeypatch.setattr(module, "data_identity", lambda: deepcopy(DATA))
    monkeypatch.setattr(module, "prior_proof", lambda: {"old": "unchanged", "old_allowance_reused": False})
    monkeypatch.setattr(module, "baseline_binding", lambda: {"controlled_baseline": "b" * 64})
    # Real capture uses an in-memory SDK transport, not the runtime transport.
    return SimpleNamespace(reads=reads)


def prepared(tmp_path):
    trial = module.RetrievalTrial(tmp_path / "trial")
    return trial, asyncio.run(trial.prepare())


def grant(trial):
    trial.authorize("Controlled test fixture only, not a live-call authorization.", confirmed=True)


def fake_knowledge(*, count=5):
    rows = [{"document_id": hashlib.sha256(json.dumps(["title", doc['title']], ensure_ascii=True).encode()).hexdigest(),
             "title": doc['title'], "content": doc['content'],
             "version": hashlib.sha256(doc['content'].encode()).hexdigest(), "chunk": 0, "score": .5}
            for doc in module.task_data._policies()[:count]]
    async def search(params, context):
        return deepcopy(rows[:params.get("top_k", 5)])
    return SimpleNamespace(proof=deepcopy(DATA), kb=SimpleNamespace(search_handler=search), close=lambda: None)


def response(text):
    return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
        "model": base.MODEL, "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 2}})


def test_prepare_captures_real_request_and_does_not_leak_labels_or_authorize(tmp_path, isolated):
    trial, p = prepared(tmp_path)
    assert p["authorized_model_requests"] == trial.report()["model_requests"] == 0
    assert p["proposed_model_requests"] == 48 and len(p["initial_requests"]) == 24
    assert trial.report()["authorized_requests"] == 0 and not any(isolated.reads)
    assert not (trial.root / "authorization.json").exists()
    for case, row in zip(module.cases(), p["initial_requests"]):
        text = json.dumps(row["payload"], ensure_ascii=False)
        assert case["input"]["message"] in text
        assert "relevant_titles" not in text and case["expected"]["rationale"] not in text
        assert row["payload"]["max_tokens"] == 512 and row["payload"]["thinking"] == {"type": "disabled"}
    before = (trial.root / "protocol.json").read_bytes()
    with pytest.raises(base.TrialError, match="retrieval_root_already_exists"):
        asyncio.run(trial.prepare())
    assert (trial.root / "protocol.json").read_bytes() == before


def test_missing_authorization_never_loads_secret_or_builds_data(tmp_path, monkeypatch, isolated):
    trial, _ = prepared(tmp_path)
    build = AsyncMock()
    monkeypatch.setattr(module.task_data, "_build_knowledge", build)
    isolated.reads.clear()
    with pytest.raises(FileNotFoundError):
        asyncio.run(trial.execute())
    assert not any(isolated.reads)
    build.assert_not_called()


@pytest.mark.parametrize("damage", ["code", "data", "prior", "params", "approval"])
def test_changed_bindings_block_before_model_or_data(tmp_path, monkeypatch, isolated, damage):
    trial, p = prepared(tmp_path)
    grant(trial)
    ledger = trial.budget(p)
    build = AsyncMock()
    monkeypatch.setattr(module.task_data, "_build_knowledge", build)
    if damage == "code": monkeypatch.setattr(module, "artifacts", lambda: {"changed": "b" * 64})
    elif damage == "data": monkeypatch.setattr(module, "data_identity", lambda: {"changed": True})
    elif damage == "prior": monkeypatch.setattr(module, "prior_proof", lambda: {"changed": True})
    else:
        name = "authorization.json" if damage == "approval" else "protocol.json"
        data = bench.read_json(trial.root / name)
        if damage == "approval": data["max_requests"] = 49
        else: data["parameters"]["rewrite"] = [1024, .5]
        (trial.root / name).write_text(json.dumps(data))
    isolated.reads.clear()
    with pytest.raises(Exception):
        asyncio.run(trial.execute())
    assert not any(isolated.reads)
    build.assert_not_called()
    assert ledger.status()["attempts"] == 0


def test_original_pipeline_performs_exactly_two_controlled_physical_calls_per_query(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    def factory(frame):
        async def handle(request):
            sent.append((frame, json.loads(request.content)))
            assert trial.budget(p).status()["attempts"] == len(sent)
            assert (trial.root / f"admission-{frame[0]}-{frame[1]}.json").exists()
            body = json.loads(request.content)
            units = (json.loads(body["messages"][0]["content"].rsplit("\n", 1)[-1])["allowed_units"]
                     if frame[1] == "rewrite" else [])
            return response(json.dumps(units[:3], ensure_ascii=False) if frame[1] == "rewrite" else '[0,1,2,3,4]')
        return httpx.MockTransport(handle)
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))
    assert len(sent) == report["model_requests"] == 48
    assert report["evidence_kinds"] == ["controlled-transport"]
    assert report["splits"]["development"]["planned_total"] == 12
    assert report["splits"]["validation"]["planned_total"] == 12
    assert [f"{case}:{stage}:1" for (case, stage), _ in sent] == list(OPERATIONS)
    for row in trial.root.glob("physical-*.json"):
        assert bench.read_json(row)["request_model"] == base.MODEL
    assert KEY not in "\n".join(f.read_text() for f in trial.root.rglob("*") if f.is_file())
    assert asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))["model_requests"] == 48
    assert len(sent) == 48


def test_network_fault_cannot_be_swallowed_as_rewrite_fallback_or_send_rerank(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    def factory(frame):
        async def handle(request):
            sent.append(frame)
            raise httpx.ReadTimeout("controlled, no real network")
        return httpx.MockTransport(handle)
    with pytest.raises(TrialTransportError):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))
    assert sent == [("r01", "rewrite")]
    assert trial.budget(p).status()["attempts"] == 1 and trial.budget(p).status()["paused"]
    assert bench.read_json(trial.root / "stopped.json")["error_code"] == "timeout"
    reported = trial.report()["splits"]["development"]
    assert reported["status_counts"]["failed"] == 1 and reported["not_executed"] == 11
    assert reported["per_case"][0]["status"] == "failed"
    assert reported["per_case"][0]["ranked_titles"] == []
    with pytest.raises(base.TrialError, match="retrieval_stopped"):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))
    assert len(sent) == 1


def test_original_model_parse_fallback_is_preserved_not_repaired(tmp_path):
    trial, _ = prepared(tmp_path)
    grant(trial)
    def factory(frame):
        return httpx.MockTransport(lambda request: response("not a JSON list"))
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))
    sample = bench.read_json(trial.root / "sample-r01.json")
    assert report["model_requests"] == 48
    assert sample["method_observation"]["rewrite_queries"] == [module.cases()[0]["input"]["message"]]
    assert sample["method_observation"]["rerank_applied"] is False
    assert sample["record"]["status"] == "failed"
    assert sample["record"]["ranked_titles"] == []  # Invalid support is not evidence/no-materials success.


def test_empty_collection_never_creates_a_fake_second_call_or_continues_fixed_plan(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    def factory(frame):
        def handle(request):
            sent.append(frame)
            return response("[]")
        return httpx.MockTransport(handle)
    with pytest.raises(base.TrialError, match="retrieval_two_stage_plan_not_reached"):
        asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge(count=0)))
    assert sent == [("r01", "rewrite")] and trial.budget(p).status()["attempts"] == 1
    sample = bench.read_json(trial.root / "sample-r01.json")
    assert sample["record"]["usage"]["physical_requests"] == 1
    assert trial.report()["splits"]["development"]["not_executed"] == 11


def test_small_candidate_pool_still_uses_one_support_decision_not_a_fake_skip(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    def factory(frame):
        def handle(request):
            sent.append(frame)
            return response("[]" if frame[1] == "rewrite" else "[1,0]")
        return httpx.MockTransport(handle)
    report = asyncio.run(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge(count=2)))
    assert report["model_requests"] == len(sent) == 48
    assert [f"{case}:{stage}:1" for (case, stage) in sent] == list(OPERATIONS)
    sample = bench.read_json(trial.root / "sample-r01.json")
    assert sample["method_observation"]["rerank_applied"] is True
    assert sample["method_observation"]["rerank_input_count"] == 2
    assert sample["record"]["usage"]["physical_requests"] == 2


def test_actual_data_identity_change_has_zero_model_requests(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    data = fake_knowledge(); data.proof["corpus_sha256"] = "changed"
    sent = []
    with pytest.raises(base.TrialError, match="retrieval_actual_data_changed"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: sent.append(request)), controlled_knowledge=data))
    assert sent == [] and trial.budget(p).status()["attempts"] == 0


def test_unknown_attempt_or_missing_claim_blocks_without_refund_or_extra_send(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    trial.budget(p).reserve(OPERATIONS[0], "a" * 64)
    with pytest.raises(base.TrialError, match="retrieval_pending_or_paused"):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: None), controlled_knowledge=fake_knowledge()))
    assert trial.budget(p).status()["attempts"] == 1
    report = trial.report()
    assert report["splits"]["development"]["status_counts"]["unknown"] == 1
    assert report["splits"]["development"]["not_executed"] == 11
    assert report["usage"]["input_tokens"] == {"known_sum": None, "known_requests": 0, "unknown_requests": 1}


def test_busy_runner_does_not_stop_the_current_owner_or_send(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    with (trial.root / "run.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(base.TrialError, match="retrieval_run_busy"):
            asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: None), controlled_knowledge=fake_knowledge()))
    assert not (trial.root / "stopped.json").exists()
    assert trial.budget(p).status()["attempts"] == 0


def test_cancel_after_delegate_entry_keeps_one_occupied_attempt_and_prevents_rerank(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    async def run():
        entered = asyncio.Event()
        def factory(frame):
            async def handle(request):
                sent.append(frame)
                entered.set()
                await asyncio.Event().wait()
            return httpx.MockTransport(handle)
        job = asyncio.create_task(trial.execute(delegate_factory=factory, controlled_knowledge=fake_knowledge()))
        await entered.wait()
        job.cancel()
        with pytest.raises(TrialTransportError):
            await job
    asyncio.run(run())
    assert sent == [("r01", "rewrite")]
    assert trial.budget(p).status()["attempts"] == 1 and trial.budget(p).status()["paused"]
    assert bench.read_json(trial.root / "stopped.json")["error_code"] == "cancelled"
    report = trial.report()["splits"]["development"]
    assert report["status_counts"]["cancelled"] == 1 and report["not_executed"] == 11


def test_cancelled_native_build_is_drained_and_closed_before_run_lock_releases(tmp_path, monkeypatch):
    trial, p = prepared(tmp_path)
    grant(trial)
    native_started, native_release = threading.Event(), threading.Event()
    closed = []
    data = fake_knowledge()
    data.close = lambda: closed.append(True)
    def build():
        native_started.set()
        assert native_release.wait(5), 'controlled native builder timed out'
        return data
    monkeypatch.setattr(module.task_data, '_build_knowledge', build)
    async def run():
        job = asyncio.create_task(trial.execute())
        while not native_started.is_set():
            await asyncio.sleep(.001)
        try:
            job.cancel()
            await asyncio.sleep(.01)
            job.cancel()  # Repeated cancellation must not abandon the thread.
            await asyncio.sleep(.01)
            assert not job.done() and closed == []
            with pytest.raises(base.TrialError, match='retrieval_run_busy'):
                await trial.execute()
        finally:
            native_release.set()
        with pytest.raises(asyncio.CancelledError):
            await job
    asyncio.run(run())
    assert closed == [True] and trial.budget(p).status()['attempts'] == 0
    with (trial.root/'run.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_close_failure_still_releases_run_lock(tmp_path):
    trial, p = prepared(tmp_path)
    grant(trial)
    data = fake_knowledge()
    data.proof['corpus_sha256'] = 'changed'
    def close():
        raise RuntimeError('controlled close failure')
    data.close = close
    with pytest.raises(RuntimeError, match='controlled close failure'):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: None), controlled_knowledge=data))
    assert trial.budget(p).status()['attempts'] == 0
    with (trial.root/'run.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_counter_failure_blocks_before_network_and_keeps_safe_error(tmp_path, monkeypatch):
    from evaluation.trial_ledger import TrialLedgerError
    trial, p = prepared(tmp_path)
    grant(trial)
    sent = []
    def fail(*_args, **_kwargs):
        raise TrialLedgerError('storage_failure')
    monkeypatch.setattr(module.RetrievalBudget, 'reserve', fail)
    with pytest.raises(TrialTransportError):
        asyncio.run(trial.execute(delegate_factory=lambda frame: httpx.MockTransport(lambda request: sent.append(request)), controlled_knowledge=fake_knowledge()))
    assert sent == [] and trial.budget(p).status()['attempts'] == 0
    assert bench.read_json(trial.root/'stopped.json')['error_code'] == 'storage_failure'


@pytest.mark.parametrize('worker_code,timed_out,expected', [(0,False,0),(2,False,2),(None,True,2)])
def test_cli_run_propagates_worker_failure_without_retry(tmp_path, monkeypatch, capsys, worker_code, timed_out, expected):
    from evaluation import task_supervisor
    monkeypatch.chdir(tmp_path)
    trial = SimpleNamespace(verify=lambda: {'controlled': True}, approval=lambda p: None)
    monkeypatch.setattr(module, 'RetrievalTrial', lambda: trial)
    sent = []
    def supervise(command, *, root):
        sent.append((command, root))
        return {'exit_code': worker_code, 'timed_out': timed_out, 'cancelled': False}
    monkeypatch.setattr(task_supervisor, 'supervise', supervise)
    assert module.main(['run']) == expected
    assert len(sent) == 1 and sent[0][0][-2:] == ['evaluation.retrieval_trial','_worker']
    assert json.loads(capsys.readouterr().out)['exit_code'] == worker_code
