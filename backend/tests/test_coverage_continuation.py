"""Operator integration with a real temporary ledger and counted SDK mocks."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from core.model_deadline import await_model
from evaluation import coverage_continuation as module, coverage_continuation_transport as wire
from evaluation import intent_trial as base
from evaluation.coverage_budget import job_specs, fixed_plan
from evaluation.task_transport import Frame, context
from test_coverage_continuation_budget import original as old_ledger, manifest as write_manifest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError("ordinary continuation tests cannot contact services")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})


@pytest_asyncio.fixture
async def fixture(tmp_path, monkeypatch):
    ledger_root, prefix, chain = old_ledger(tmp_path)
    root = ledger_root.parent
    write_manifest(ledger_root, prefix, chain)
    p = json.loads((root / module.MANIFEST).read_text())
    captures = []
    for spec in job_specs()[7:9]:
        value = await base._capture({"id": spec["case_id"], "input": {"message": "核对当前订单", "history": []}},
                                   "https://dashscope.aliyuncs.com/apps/anthropic")
        value["case_id"] = spec["case_id"]
        captures.append(value)
    protocol = {"jobs": job_specs(), "plan": fixed_plan(), "initial_intent_requests": captures,
                "initial_rewrite_requests": [], "cases": {r["source_case_id"]: {} for r in job_specs()[7:9]}}
    base._write(root, "protocol.json", protocol)
    base._write(root, "data-proof.json", {"controlled_data": True})
    base._write(root, module.APPROVAL, {"explicit_controlled_approval": True})
    base._write(root, "stopped.json", {"old": "retained timeout"})
    base._write(root, "worker-stop.json", {"old": "retained worker stop"})
    old_hashes = {name: (root / name).read_bytes() for name in ("stopped.json", "worker-stop.json", "protocol.json", "data-proof.json")}
    trial = module.CoverageContinuation(root)
    runtime_manifest = {**p, "remaining_jobs": job_specs()[7:9], "redis_binding": {}}
    monkeypatch.setattr(trial, "verify", lambda: (protocol, runtime_manifest))
    monkeypatch.setattr(trial, "_approval", lambda *a: {"approved": True})
    monkeypatch.setattr(trial, "report", lambda: trial.ledger().status())
    monkeypatch.setattr(module.original, "configuration", lambda **k: {"api_key": "controlled-only", "base_url": "https://dashscope.aliyuncs.com/apps/anthropic"})
    data = SimpleNamespace(proof={"controlled_data": True}, assert_healthy=lambda: None, close=AsyncMock())
    from evaluation import coverage_runtime
    monkeypatch.setattr(coverage_runtime, "build_data", AsyncMock(return_value=data))
    redis = SimpleNamespace(aclose=AsyncMock())
    from redis.asyncio import Redis
    monkeypatch.setattr(Redis, "from_url", lambda *a, **k: redis)
    monkeypatch.setattr(module.original, "verify_redis", AsyncMock())
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_URL", "redis://127.0.0.1:16389/1")
    calls, errors = [], {}

    def response(request):
        frame = wire.CURRENT.get()
        calls.append(frame.case_id)
        status = errors.get(frame.case_id)
        if status == "timeout":
            raise httpx.ReadTimeout("controlled timeout", request=request)
        return httpx.Response(status or 200, json={"id": "controlled", "type": "message", "role": "assistant",
            "model": base.MODEL, "content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1}}, request=request)

    transport_type = wire.CoverageContinuationTransport
    monkeypatch.setattr(wire, "CoverageContinuationTransport", lambda *a, **k: transport_type(
        *a, **k, delegate_factory=lambda frame: httpx.MockTransport(response)))

    async def job(self, spec, case, sdk, unused_data, healthy):
        payload = next(row["payload"] for row in captures if row["case_id"] == spec["case_id"])
        arguments = deepcopy(payload)
        arguments["extra_body"] = {"thinking": arguments.pop("thinking")}
        with context(Frame(spec["case_id"], "turn-1", phase="intent", evidence_guard=healthy)):
            await await_model(sdk.messages.create(**arguments), timeout_s=8)
        healthy()
        return {"execution_status": "success", "tool_checks": {}, "answer": "controlled structural test only"}

    monkeypatch.setattr(module.original.CoverageExecutionTrial, "_job", job)
    return trial, root, prefix, old_hashes, calls, errors, data, redis


@pytest.mark.asyncio
async def test_sent_timeout_is_sealed_once_then_next_fixed_case_executes(fixture):
    trial, root, prefix, old, calls, errors, data, redis = fixture
    errors[job_specs()[7]["case_id"]] = "timeout"
    result = await trial.execute()
    assert calls == [r["case_id"] for r in job_specs()[7:9]]
    assert result["attempts"] == 22 and not result["paused"]
    assert result["completed_cases"] == [r["case_id"] for r in job_specs()[:9]]
    assert (root / "ledger/journal.jsonl").read_bytes().startswith(prefix)
    assert all((root / name).read_bytes() == value for name, value in old.items())
    failed = json.loads((root / f"case-{calls[0]}.json").read_text())
    assert failed["execution_status"] == "failed" and failed["continuation_failure"]["unknown_upstream_result"]
    assert failed["evidence_kind"] == "controlled-transport"
    assert data.close.await_count == redis.aclose.await_count == 1
    with pytest.raises(Exception):
        await trial.execute()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_third_consecutive_network_case_stops_without_extra_request(fixture):
    trial, root, prefix, old, calls, errors, data, redis = fixture
    for spec in job_specs()[7:9]:
        errors[spec["case_id"]] = "timeout"
    with pytest.raises(Exception):
        await trial.execute()
    s = trial.ledger().status()
    assert s["attempts"] == 22 and s["paused"] and s["consecutive_execution_failures"] == 3
    assert len(calls) == 2 and (root / module.STOP).exists()
    assert data.close.await_count == redis.aclose.await_count == 1
    assert (root / "ledger/journal.jsonl").read_bytes().startswith(prefix)


@pytest.mark.asyncio
async def test_auth_failure_does_not_continue_or_refund(fixture):
    trial, root, _, _, calls, errors, _, _ = fixture
    errors[job_specs()[7]["case_id"]] = 401
    with pytest.raises(BaseException):
        await trial.execute()
    assert len(calls) == 1 and trial.ledger().status()["attempts"] == 21
    assert (root / "continuation-v1-stop.json").exists()


def test_original_grant_uses_current_interpreter_after_resume(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    base._write(root, module.MANIFEST, {"present": True})
    calls = []
    class Legacy:
        def __init__(self, root): pass
        def _approval(self, p):
            calls.append(self.ledger(p).initial_chain_hash)
            return {"max_requests": 480}
    monkeypatch.setattr(module.original, "CoverageExecutionTrial", Legacy)
    monkeypatch.setattr(module.CoverageContinuation, "ledger", lambda *a, **k: SimpleNamespace(initial_chain_hash="unchanged"))
    p = {"plan": fixed_plan(), "jobs": job_specs(), "cases": {}, "configuration": {},
         "public_parameters": {"agent_parameters": {"role": [100, 0.1]}}, "effective_environment": {}, "dependencies": {},
         "preparation": {}, "prior_evidence": {}, "http_limits": module.PHASE_HTTP_LIMITS}
    monkeypatch.setattr(module, "_read", lambda *a: p)
    monkeypatch.setattr(module.bench, "sha256", lambda *a: module.ORIGINAL_PROTOCOL_SHA)
    monkeypatch.setattr(module.original, "source_cases", lambda: {})
    monkeypatch.setattr(module.original, "configuration", lambda: {})
    monkeypatch.setattr(module.preparation, "public_parameters", lambda: {"agent_parameters": {"role": (100, 0.1)}})
    monkeypatch.setattr(module.original.history, "effective_environment", lambda: {})
    monkeypatch.setattr(module.original.engine, "_dependencies", lambda: {})
    monkeypatch.setattr(module.original, "frozen_preparation", lambda: {})
    monkeypatch.setattr(module.original, "preserved_evidence", lambda: {})
    monkeypatch.setattr(module, "_original_sources", lambda ignored: {})
    assert module._original_valid(root) == p
    assert calls == ["unchanged"]


def test_supervisor_rejects_other_roots_and_excess_deadline_before_process(monkeypatch, tmp_path):
    from evaluation import task_supervisor as supervisor
    import sys
    starts = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: starts.append(a))
    command = [sys.executable, "-B", "-u", "-m", "evaluation.coverage_continuation", "_worker"]
    with pytest.raises(supervisor.SupervisorError, match="coverage_continuation_worker_root_invalid"):
        supervisor.supervise(command, root=tmp_path / "another-batch", deadline_s=3600)
    with pytest.raises(supervisor.SupervisorError, match="deadline_invalid"):
        supervisor.supervise(command, root=module.ROOT, deadline_s=3601)
    assert not starts


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "extra", "spent", "remaining", "maximum", "manifest"])
async def test_approval_cannot_reset_twenty_or_expand_original480(fixture, damage):
    trial, root, _, _, _, _, _, _ = fixture
    base._write(root, "authorization.json", {"controlled_original_grant": 480})
    m = json.loads((root / module.MANIFEST).read_text())
    value = {"schema_version": module.SCHEMA + "-authorization", "recorded_at": "2026-11-05T00:00:00Z",
             "reference": "explicit controlled continuation, no new allowance",
             "continuation_sha256": hashlib.sha256((root / module.MANIFEST).read_bytes()).hexdigest(),
             "original_authorization_sha256": hashlib.sha256((root / "authorization.json").read_bytes()).hexdigest(),
             "prefix_sha256": m["budget"]["prefix_sha256"], "original_genesis": trial.ledger().initial_chain_hash,
             "max_total_requests": 480, "already_spent": 20, "max_additional_requests": 460}
    if damage == "extra": value["ignore_version"] = True
    elif damage == "spent": value["already_spent"] = 0
    elif damage == "remaining": value["max_additional_requests"] = 480
    elif damage == "maximum": value["max_total_requests"] = 960
    elif damage == "manifest": value["continuation_sha256"] = "f" * 64
    # This is an owned temporary fixture file, never a production approval.
    path = root / module.APPROVAL
    path.write_text(json.dumps(value))
    if damage:
        with pytest.raises(Exception):
            module.CoverageContinuation._approval(trial, {}, m)
    else:
        assert module.CoverageContinuation._approval(trial, {}, m) == value
