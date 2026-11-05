"""Second operator flow: real temporary counter and SDK MockTransport only."""
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from core.model_deadline import await_model
from evaluation import coverage_continuation2 as module, coverage_continuation2_transport as wire
from evaluation import intent_trial as base
from evaluation.coverage_budget import job_specs, fixed_plan
from evaluation.task_transport import CURRENT, Frame, context
from test_coverage_continuation2_budget import snapshot201, frozen, H


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k): raise AssertionError("no external services in ordinary tests")
    for n in ("connect", "connect_ex"): monkeypatch.setattr(socket.socket, n, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})


@pytest_asyncio.fixture
async def fixture(frozen, monkeypatch):
    ledger_root, _, prefix, _, _ = frozen
    root = ledger_root.parent
    monkeypatch.setattr(module.prior, "ORIGINAL_PROTOCOL_SHA", H)
    m = json.loads((root / module.MANIFEST).read_text())
    captures = []
    for spec in job_specs()[70:72]:
        v = await base._capture({"id": spec["case_id"], "input": {"message": "请核对付款记录", "history": []}},
                               "https://dashscope.aliyuncs.com/apps/anthropic")
        v["case_id"] = spec["case_id"]
        captures.append(v)
    p = {"jobs": job_specs(), "plan": fixed_plan(), "initial_intent_requests": captures, "initial_rewrite_requests": [],
         "cases": {s["source_case_id"]: {} for s in job_specs()[70:72]}}
    base._write(root, "protocol.json", p)
    base._write(root, "data-proof.json", {"controlled": True})
    base._write(root, module.APPROVAL, {"controlled": True})
    trial = module.CoverageContinuation2(root)
    monkeypatch.setattr(trial, "verify", lambda: (p, {**m, "remaining_jobs": job_specs()[70:72], "redis_binding": {}}))
    monkeypatch.setattr(trial, "_approval", lambda *a: {})
    monkeypatch.setattr(trial, "report", lambda: trial.ledger().status())
    data = SimpleNamespace(proof={"controlled": True}, assert_healthy=lambda: None, close=AsyncMock())
    from evaluation import coverage_runtime
    monkeypatch.setattr(coverage_runtime, "build_data", AsyncMock(return_value=data))
    monkeypatch.setattr(module.initial, "configuration", lambda **k: {"api_key": "controlled-not-real", "base_url": "https://dashscope.aliyuncs.com/apps/anthropic"})
    redis = SimpleNamespace(aclose=AsyncMock())
    from redis.asyncio import Redis
    monkeypatch.setattr(Redis, "from_url", lambda *a, **k: redis)
    monkeypatch.setenv("SERVEFLOW_EVAL_REDIS_URL", "redis://127.0.0.1:16389/1")
    monkeypatch.setattr(module.initial, "verify_redis", AsyncMock())
    calls, errors = [], {}
    def respond(request):
        f = CURRENT.get()
        calls.append(f.case_id)
        mode = errors.get(f.case_id)
        if mode == "timeout": raise httpx.ReadTimeout("controlled", request=request)
        return httpx.Response(mode or 200, json={"id": "controlled", "type": "message", "role": "assistant",
            "model": base.MODEL, "content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1}}, request=request)
    real = wire.CoverageContinuation2Transport
    class ControlledTransport(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k, delegate_factory=lambda f: httpx.MockTransport(respond))
    monkeypatch.setattr(wire, "CoverageContinuation2Transport", ControlledTransport)
    async def job(self, spec, case, sdk, unused, healthy):
        args = deepcopy(next(x["payload"] for x in captures if x["case_id"] == spec["case_id"]))
        args['extra_body'] = {'thinking': args.pop('thinking')}
        with context(Frame(spec["case_id"], "turn-1", phase="intent", evidence_guard=healthy)):
            await await_model(sdk.messages.create(**args), timeout_s=8)
        healthy()
        return {"execution_status": "success", "answer": "controlled integration only"}
    monkeypatch.setattr(module.initial.CoverageExecutionTrial, "_job", job)
    return trial, root, prefix, calls, errors, data, redis


@pytest.mark.asyncio
async def test_old201_are_never_resent_and_isolated_fault_continues(fixture):
    trial, root, prefix, calls, errors, data, redis = fixture
    errors[job_specs()[70]["case_id"]] = "timeout"
    state = await trial.execute()
    assert calls == [s['case_id'] for s in job_specs()[70:72]]
    assert state['attempts'] == 203 and not state['paused']
    assert len(state['completed_cases']) == 72
    assert (root/'ledger/journal.jsonl').read_bytes().startswith(prefix)
    assert data.close.await_count == redis.aclose.await_count == 1
    assert json.loads((root/'case-A-CT-V14.json').read_text())['execution_status'] == 'failed'
    with pytest.raises(Exception): await trial.execute()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_auth_or_three_consecutive_faults_do_not_skip_failure(fixture):
    trial, root, prefix, calls, errors, _, _ = fixture
    for spec in job_specs()[70:72]: errors[spec['case_id']] = 'timeout'
    with pytest.raises(Exception): await trial.execute()
    assert trial.ledger().status()['attempts'] == 203 and trial.ledger().status()['paused']
    assert len(calls) == 2 and (root/module.STOP).exists()
    assert (root/'ledger/journal.jsonl').read_bytes().startswith(prefix)
