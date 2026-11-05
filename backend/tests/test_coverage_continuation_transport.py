"""Real SDK/MockTransport checks; no credentials, external socket or services."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

from anthropic import AsyncAnthropic
import httpx
import pytest
import pytest_asyncio

from core.model_deadline import EvaluationDeadlinePolicy, await_model, eval_deadlines
from evaluation import coverage_continuation_transport as module, intent_trial as base
from evaluation.intent_transport import TrialTransportError
from evaluation.task_transport import Frame, context


ENDPOINT = "https://dashscope.aliyuncs.com/apps/anthropic"
CASES = ("A-CT-D01", "D-CT-D01", "D-CT-D02")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("ordinary continuation tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    opened = Path.open
    def safe(path, *args, **kwargs):
        if path.name == ".env":
            denied()
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", safe)


class Budget:
    """Counting fixture only; persistent journal behavior has its own suite."""
    def __init__(self):
        self.attempts, self.active, self.paused = 20, None, False
        self.pending, self.finished = {}, []
        self.ordinals = {}
        self.lease = object()

    def status(self):
        return {"attempts": self.attempts, "active_case": self.active,
                "pending": deepcopy(self.pending), "paused": self.paused}

    def reserve(self, case_id, turn_id, phase, agent, payload, wire, *, lease):
        assert lease is self.lease and not self.paused and self.active == case_id
        self.attempts += 1
        actor = (case_id, turn_id, phase, agent)
        self.ordinals[actor] = self.ordinals.get(actor, 0) + 1
        operation = {"operation_id": f"{case_id}:{turn_id}:{phase}:{agent or 'none'}:{self.ordinals[actor]}",
                     "attempt": self.attempts, "case_id": case_id, "turn_id": turn_id,
                     "phase": phase, "agent": agent, "payload_sha256": payload,
                     "request_bytes_sha256": wire}
        assert operation["operation_id"] not in self.pending
        self.pending[operation["operation_id"]] = operation
        return operation

    def finish(self, identity, outcome, digest, code, *, lease):
        assert lease is self.lease
        self.finished.append({**self.pending.pop(identity), "outcome": outcome,
                              "output_sha256": digest, "error_code": code})
        self.paused |= outcome in {"execution_error", "cancelled"}

    def next_case(self, case_id):
        assert not self.pending
        self.active, self.paused = case_id, False


@pytest_asyncio.fixture
async def fixture(tmp_path):
    root = tmp_path / "trial"
    root.mkdir(mode=0o700)
    captures = []
    for identity in CASES:
        row = await base._capture({"id": identity, "input": {"message": "请核对订单", "history": []}}, ENDPOINT)
        row["case_id"] = identity
        captures.append(row)
    protocol = {"jobs": [{"case_id": identity, "suite": "tasks"} for identity in CASES],
                "plan": [{"case_id": identity, "turn_ids": ["turn-1"]} for identity in CASES],
                "initial_intent_requests": captures, "initial_rewrite_requests": []}
    base._write(root, "protocol.json", protocol)
    base._write(root, module.MANIFEST, {"schema_version": module.SCHEMA, "explicit_controlled": True})
    # This historical failure must remain byte-for-byte untouched.
    (root / "stopped.json").write_bytes(b'{"historical":"must remain"}\n')
    return root, protocol, hashlib.sha256((root / module.MANIFEST).read_bytes()).hexdigest(), Budget()


def response():
    return httpx.Response(200, json={"id": "controlled", "type": "message", "role": "assistant",
        "model": base.MODEL, "content": [{"type": "text", "text": "controlled reply"}],
        "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 2}})


def transport(fixture, factory, verifier=lambda: None):
    root, _, digest, budget = fixture
    return module.CoverageContinuationTransport(ENDPOINT, root, budget, budget.lease, verifier,
        continuation_sha256=digest, delegate_factory=lambda frame: httpx.MockTransport(factory(frame)))


async def sdk_send(client, protocol, identity, *, turn="turn-1"):
    payload = deepcopy(next(row["payload"] for row in protocol["initial_intent_requests"] if row["case_id"] == identity))
    thinking = payload.pop("thinking")
    with context(Frame(identity, turn, "intent", None)):
        return await await_model(client.messages.create(**payload, extra_body={"thinking": thinking}), timeout_s=8)


async def sdk(t):
    return AsyncAnthropic(api_key="controlled-not-a-real-key", base_url=ENDPOINT,
        max_retries=0, timeout=30, http_client=httpx.AsyncClient(transport=t, timeout=30,
            trust_env=False, follow_redirects=False, headers={"Accept-Encoding": "identity"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "network"])
async def test_sent_fault_spends_once_closes_case_then_independent_case_can_run(fixture, fault):
    root, protocol, _, budget = fixture
    calls = []
    def factory(frame):
        async def send(request):
            calls.append(frame.case_id)
            if frame.case_id == CASES[0]:
                raise (httpx.ReadTimeout("not logged") if fault == "timeout" else httpx.ConnectError("not logged"))
            return response()
        return send
    t, client = transport(fixture, factory), None
    client = await sdk(t)
    try:
        budget.next_case(CASES[0]); t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception):
                await sdk_send(client, protocol, CASES[0])
            with pytest.raises(module.CaseExecutionFailure):
                t.assert_case_healthy()
            # An additional SDK invocation is rejected, never sent or charged.
            with pytest.raises(Exception):
                await sdk_send(client, protocol, CASES[0])
        failure = await t.finish_case()
        assert failure["error_code"] == fault and t.fault is None
        assert calls == [CASES[0]] and budget.attempts == 21 and not budget.pending
        assert budget.finished[0]["error_code"] == fault
        assert not (root / module.STOP_FILE).exists()
        old = (root / "stopped.json").read_bytes()
        budget.next_case(CASES[1]); t.begin_case(CASES[1])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            await sdk_send(client, protocol, CASES[1])
        assert await t.finish_case() is None
        assert calls == list(CASES[:2]) and budget.attempts == 22
        assert (root / "stopped.json").read_bytes() == old
        physical = json.loads((root / "physical-021.json").read_text())
        assert physical["continuation_sha256"] == fixture[2]
        assert physical["response"]["usage"] == {"input_tokens": None, "output_tokens": None}
        assert physical["network_limit_s"] == 8 and physical["sdk_limit_s"] == 30
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(401, "auth"), (402, "quota"), (429, "rate_limit")])
async def test_auth_accounting_or_rate_failure_permanently_stops(fixture, status, code):
    root, protocol, _, budget = fixture
    calls = []
    def factory(frame):
        async def send(request):
            calls.append(frame.case_id)
            return httpx.Response(status, json={"error": {"message": "private upstream error"}})
        return send
    t = transport(fixture, factory); client = await sdk(t)
    try:
        budget.next_case(CASES[0]); t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception): await sdk_send(client, protocol, CASES[0])
        await t.finish_case()
        assert t.fault == code and t.case_failure is None and budget.attempts == 21
        record = json.loads((root / module.STOP_FILE).read_text())
        assert record["error_code"] == code and "private upstream" not in str(record)
        budget.next_case(CASES[1])
        with pytest.raises(TrialTransportError): t.begin_case(CASES[1])
        assert calls == [CASES[0]]
    finally: await client.close()


@pytest.mark.asyncio
async def test_manifest_mutation_before_delegate_is_rejected_without_a_send(fixture):
    root, protocol, _, budget = fixture
    calls = []
    def factory(frame):
        async def send(request): calls.append(frame); return response()
        return send
    t = transport(fixture, factory); client = await sdk(t)
    try:
        budget.next_case(CASES[0]); t.begin_case(CASES[0])
        (root / module.MANIFEST).write_text('{"schema_version":"wrong"}')
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception): await sdk_send(client, protocol, CASES[0])
        assert not calls and budget.attempts == 20 and t.fault == "protocol"
    finally: await client.close()


@pytest.mark.asyncio
async def test_late_old_frame_cannot_send_after_new_case_opened(fixture):
    _, protocol, _, budget = fixture
    calls = []
    def factory(frame):
        async def send(request): calls.append(frame.case_id); return response()
        return send
    t = transport(fixture, factory); client = await sdk(t)
    try:
        budget.next_case(CASES[0]); t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()): await sdk_send(client, protocol, CASES[0])
        await t.finish_case()
        budget.next_case(CASES[1]); t.begin_case(CASES[1])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception): await sdk_send(client, protocol, CASES[0])
            await sdk_send(client, protocol, CASES[1])
        await t.finish_case()
        assert calls == list(CASES[:2]) and budget.attempts == 22
    finally: await client.close()


@pytest.mark.asyncio
async def test_external_cancel_is_fatal_and_does_not_become_a_soft_timeout(fixture):
    root, protocol, _, budget = fixture
    sent = asyncio.Event()
    def factory(frame):
        async def send(request):
            sent.set(); await asyncio.sleep(30)
            return response()
        return send
    t = transport(fixture, factory); client = await sdk(t)
    budget.next_case(CASES[0]); t.begin_case(CASES[0])
    async def call():
        with eval_deadlines(EvaluationDeadlinePolicy()):
            await sdk_send(client, protocol, CASES[0])
    task = asyncio.create_task(call())
    try:
        await sent.wait(); task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await t.finish_case()
        assert t.fault == "cancelled" and t.case_failure is None
        assert budget.attempts == 21 and not budget.pending
        assert budget.finished[0]["outcome"] == "cancelled"
        assert json.loads((root / module.STOP_FILE).read_text())["error_code"] == "cancelled"
    finally: await client.close()


@pytest.mark.asyncio
async def test_one_fault_does_not_cancel_already_sent_sibling_and_finish_waits(fixture):
    root, protocol, _, budget = fixture
    both_sent, release, count = asyncio.Event(), asyncio.Event(), []
    def factory(frame):
        async def send(request):
            count.append(frame.agent)
            if len(count) == 2: both_sent.set()
            await both_sent.wait()
            if frame.agent == "general": raise httpx.ReadTimeout("controlled")
            await release.wait(); return response()
        return send
    t = transport(fixture, factory); client = await sdk(t)
    budget.next_case(CASES[0]); t.begin_case(CASES[0])
    async def call(agent):
        from evaluation.task_transport import AGENT_PARAMETERS
        tokens,temp = AGENT_PARAMETERS[agent]
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with context(Frame(CASES[0], "turn-1", "agent", agent)):
                return await await_model(client.messages.create(model=base.MODEL,max_tokens=tokens,temperature=temp,
                    messages=[{"role":"user","content":"controlled"}],extra_body={"thinking":{"type":"disabled"}}),timeout_s=15)
    first, second = asyncio.create_task(call("general")), asyncio.create_task(call("billing"))
    try:
        await both_sent.wait(); await asyncio.gather(first, return_exceptions=True)
        assert not second.done() and t.case_failure["error_code"] == "timeout"
        finishing = asyncio.create_task(t.finish_case())
        await asyncio.sleep(0)
        assert not finishing.done()
        release.set(); await second
        failure = await finishing
        assert failure["error_code"] == "timeout" and t.fault is None
        assert {x["outcome"] for x in budget.finished} == {"execution_error", "received"}
        assert budget.attempts == 22 and not budget.pending
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_unsealed_pending_claim_or_unclosed_prior_case_blocks_advance(fixture):
    root, _, _, budget = fixture
    t = transport(fixture, lambda frame: lambda request: response())
    budget.next_case(CASES[0]); t.begin_case(CASES[0])
    with pytest.raises(TrialTransportError): t.begin_case(CASES[1])
    budget.pending["uncertain"] = {"attempt":21}
    with pytest.raises(TrialTransportError): await t.finish_case()
    assert t.fault == "storage_failure"
    budget.next_case(CASES[1]) if not budget.pending else None
    assert (root / module.STOP_FILE).exists()


@pytest.mark.asyncio
async def test_native_tool_feedback_round_trip_preserves_sdk_objects_and_bytes(fixture):
    _, _, _, budget = fixture
    requests = []
    def factory(frame):
        async def send(request):
            body = json.loads(request.content); requests.append(body)
            if len(requests) == 1:
                return httpx.Response(200, json={"id":"controlled-first","type":"message","role":"assistant",
                    "model":base.MODEL,"content":[{"type":"tool_use","id":"controlled-tool-1",
                    "name":"read_demo_order","input":{"order_id":"DEMO-2101"}}],"stop_reason":"tool_use",
                    "stop_sequence":None,"usage":{"input_tokens":3,"output_tokens":2}})
            return response()
        return send
    t = transport(fixture, factory); client = await sdk(t)
    budget.next_case(CASES[0]); t.begin_case(CASES[0])
    messages = [{"role":"user","content":"查DEMO-2101"}]
    try:
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with context(Frame(CASES[0], "turn-1", "agent", "general")):
                args={"model":base.MODEL,"max_tokens":900,"temperature":.3,
                    "tools":[{"name":"read_demo_order","description":"controlled read",
                        "input_schema":{"type":"object","properties":{"order_id":{"type":"string"}},
                            "required":["order_id"],"additionalProperties":False}}],
                    "extra_body":{"thinking":{"type":"disabled"}}}
                first=await await_model(client.messages.create(messages=messages,**args),timeout_s=15)
                messages += [{"role":"assistant","content":first.content},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"controlled-tool-1",
                        "content":"controlled-source-not-a-real-transaction"}]}]
                await await_model(client.messages.create(messages=messages,**args),timeout_s=15)
        assert await t.finish_case() is None
        assert len(requests)==2 and budget.attempts==22
        assert requests[1]["messages"][1]["content"][0]["id"]=="controlled-tool-1"
        assert requests[1]["messages"][2]["content"][0]["tool_use_id"]=="controlled-tool-1"
        assert all(r["model"]==base.MODEL and r["max_tokens"]==900 and r["temperature"]==.3 for r in requests)
    finally: await client.close()


@pytest.mark.asyncio
async def test_http200_bad_json_is_sealed_failure_without_stopping_next_case(fixture):
    root, protocol, _, budget=fixture
    calls=[]
    def factory(frame):
        async def send(request):
            calls.append(frame.case_id)
            return httpx.Response(200,content=b'{',headers={'content-type':'application/json'}) if frame.case_id==CASES[0] else response()
        return send
    t=transport(fixture,factory);client=await sdk(t)
    try:
        budget.next_case(CASES[0]);t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(json.JSONDecodeError):await sdk_send(client,protocol,CASES[0])
        assert await t.finish_case() is None and t.fault is None
        assert budget.finished[0]["outcome"]=="structure_failed"
        assert json.loads((root/'physical-021.json').read_text())["http_status"]==200
        budget.next_case(CASES[1]);t.begin_case(CASES[1])
        with eval_deadlines(EvaluationDeadlinePolicy()):await sdk_send(client,protocol,CASES[1])
        await t.finish_case()
        assert calls==list(CASES[:2]) and budget.attempts==22
    finally:await client.close()


@pytest.mark.asyncio
async def test_manifest_changed_after_claim_never_enters_network_delegate(fixture):
    root, protocol, _, budget=fixture
    calls=[]
    def factory(frame):
        async def send(request):calls.append(frame);return response()
        return send
    old=budget.reserve
    def reserve(*args,**kwargs):
        result=old(*args,**kwargs)
        (root/module.MANIFEST).write_text('{"schema_version":"changed-after-claim"}')
        return result
    budget.reserve=reserve
    t=transport(fixture,factory);client=await sdk(t)
    try:
        budget.next_case(CASES[0]);t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
        await t.finish_case()
        assert not calls and budget.attempts==21 and t.fault=="protocol"
        record=json.loads((root/'physical-021.json').read_text())
        assert record["delegate_entered"] is False and record["error_code"]=="protocol"
    finally:await client.close()


@pytest.mark.asyncio
async def test_physical_seal_failure_is_permanent_not_a_recoverable_network_case(fixture,monkeypatch):
    root, protocol, _, budget=fixture
    def factory(frame):
        async def send(request):raise httpx.ReadTimeout('controlled')
        return send
    t=transport(fixture,factory);client=await sdk(t)
    old=module.base._write
    def write(path,name,value):
        if name.startswith('physical-'):raise OSError('private storage error')
        return old(path,name,value)
    monkeypatch.setattr(module.base,'_write',write)
    try:
        budget.next_case(CASES[0]);t.begin_case(CASES[0])
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with pytest.raises(Exception):await sdk_send(client,protocol,CASES[0])
        assert t.fault=="storage_failure" and budget.attempts==21 and len(budget.pending)==1
        with pytest.raises(TrialTransportError):await t.finish_case()
        assert not list(root.glob('physical-*'))
    finally:await client.close()


@pytest.mark.asyncio
async def test_failed_drain_cancels_then_permanently_blocks_advance(fixture,monkeypatch):
    root, _, _, budget=fixture
    t=transport(fixture,lambda frame:lambda request:response())
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    entered=asyncio.Event()
    async def stuck():
        entered.set()
        try:await asyncio.sleep(30)
        finally:t._tasks.discard(asyncio.current_task())
    task=asyncio.create_task(stuck());t._tasks.add(task)
    await entered.wait()
    monkeypatch.setattr(module,'MAX_DRAIN_S',.01)
    try:
        with pytest.raises(TrialTransportError,match='storage_failure'):await t.finish_case()
        assert task.done() and task.cancelled() and t.fault=="storage_failure"
        budget.next_case(CASES[1])
        with pytest.raises(TrialTransportError):t.begin_case(CASES[1])
        assert json.loads((root/module.STOP_FILE).read_text())["error_code"]=="storage_failure"
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_active_deadline_is_a_timeout_not_external_cancellation(fixture):
    root, protocol, _, budget=fixture
    def factory(frame):
        async def send(request):await asyncio.sleep(30);return response()
        return send
    t=transport(fixture,factory);client=await sdk(t)
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    payload=deepcopy(protocol['initial_intent_requests'][0]['payload']);thinking=payload.pop('thinking')
    try:
        with eval_deadlines(EvaluationDeadlinePolicy()):
            with context(Frame(CASES[0],'turn-1','intent',None)):
                # Short controlled clock exposes the same cancellation branch;
                # configured transport limits and production values stay 8/30.
                with pytest.raises(Exception):
                    await await_model(client.messages.create(**payload,extra_body={'thinking':thinking}),timeout_s=.03)
        failure=await t.finish_case()
        assert failure['error_code']=='timeout' and t.fault is None
        physical=json.loads((root/'physical-021.json').read_text())
        assert physical['error_code']=='timeout' and physical['stage_deadline_reason']=='active'
        assert physical['outcome']=='execution_error' and physical['network_limit_s']==8
    finally:await client.close()


@pytest.mark.asyncio
async def test_repeated_operator_cancel_waits_for_bounded_drain_cleanup(fixture):
    root, _, _, budget=fixture
    t=transport(fixture,lambda frame:lambda request:response())
    budget.next_case(CASES[0]);t.begin_case(CASES[0])
    cleaning,release=asyncio.Event(),asyncio.Event()
    async def inflight():
        try:await asyncio.sleep(30)
        finally:
            cleaning.set()
            await release.wait()
            t._tasks.discard(asyncio.current_task())
    task=asyncio.create_task(inflight());t._tasks.add(task)
    await asyncio.sleep(0)
    finishing=asyncio.create_task(t.finish_case())
    try:
        await asyncio.sleep(0);finishing.cancel()
        await cleaning.wait();finishing.cancel()
        await asyncio.sleep(0)
        assert not finishing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):await finishing
        assert task.done() and t.fault=='cancelled'
        assert json.loads((root/module.STOP_FILE).read_text())['error_code']=='cancelled'
    finally:
        release.set();task.cancel()
        await asyncio.gather(task,finishing,return_exceptions=True)
