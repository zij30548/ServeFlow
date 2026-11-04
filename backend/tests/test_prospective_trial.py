"""Offline SDK, real durable storage and controlled handlers; no model/DB calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import socket
import stat
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from core.model_deadline import EvaluationDeadlinePolicy, await_model, eval_deadlines
from evaluation import benchmark as bench, prospective_trial as module
from evaluation import prospective_task_runtime as task_runtime
from evaluation.intent_transport import TrialTransportError, canonical_payload_hash
from evaluation.task_transport import Frame, context, PHASE_PARAMETERS, AGENT_PARAMETERS
from evaluation import task_supervisor


BASE = "https://dashscope.aliyuncs.com/apps/anthropic"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("new execution tests must not contact a service")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "design_proof", lambda: {"original_design": "unchanged-controlled-proof"})
    monkeypatch.setattr(module, "artifacts", lambda: {"same_source": "a"*64})
    monkeypatch.setattr(module, "prior_ledgers", lambda: {"old_records": "b"*64})
    monkeypatch.setattr(module.history, "effective_environment", lambda: {"ORT_DISABLE_TELEMETRY": "1"})
    monkeypatch.setattr(module.task_data, "offline_data_proof", lambda: {"controlled": True})
    def config(*, secret=False):
        return {"base_url": BASE, "model": module.base.MODEL, **({"api_key": "controlled-only"} if secret else {})}
    monkeypatch.setattr(module.base, "_configuration", config)
    trial = module.ProspectiveTrial(tmp_path/"new-execution")
    asyncio.run(trial.prepare())
    return trial


def approved(trial, limit=20):
    trial.authorize("controlled test receipt, no live allowance", max_requests=limit, confirmed=True)
    return trial


def response(text="controlled answer", *, content=None):
    return httpx.Response(200, headers={"request-id": "controlled-request"}, json={
        "id": "msg_controlled", "type": "message", "role": "assistant", "model": module.base.MODEL,
        "content": content or [{"type": "text", "text": text}], "stop_reason": "tool_use" if content else "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 7, "output_tokens": 3}})


def wire(phase="agent", agent="general", payload=None):
    tokens, temperature = AGENT_PARAMETERS[agent] if phase == "agent" else PHASE_PARAMETERS[phase]
    value = payload or {"model": module.base.MODEL, "max_tokens": tokens, "temperature": temperature,
        "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "controlled"}]}
    # The production SDK serializes spaces; json= is only used for dynamic calls.
    return httpx.Request("POST", BASE+"/v1/messages", json=value,
        extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), 30)})


async def send(transport, phase="agent", agent="general", job="A-p01", request=None):
    with context(Frame(job, "turn-1", phase, agent if phase == "agent" else None)):
        return await await_model(transport.handle_async_request(request or wire(phase, agent)),
            timeout_s={"agent":15,"intent":8,"rag_rewrite":10,"rag_rerank":30,"composer":8,"judge":8}[phase])


def test_prepare_is_zero_send_exact_new_plan_with_real_offline_sdk_capture(prepared):
    p = prepared.verify()
    assert p["authorized_model_requests"] == p["judge_requests"] == p["external_business_requests"] == 0
    assert p["proposed_model_requests"] == 576
    assert len(p["plan"]) == 36 and len(p["initial_intent_requests"]) == 24 and len(p["initial_rewrite_requests"]) == 12
    assert len({r["source_case_id"] for r in p["initial_intent_requests"]}) == 12
    for pair in zip(p["initial_intent_requests"][::2], p["initial_intent_requests"][1::2]):
        assert pair[0]["payload"] == pair[1]["payload"]
        assert pair[0]["request_bytes_sha256"] == pair[1]["request_bytes_sha256"]
    for row in p["initial_intent_requests"]+p["initial_rewrite_requests"]:
        assert set(row["payload"]) <= {"model", "messages", "max_tokens", "temperature", "thinking", "stream"}
        assert "expected" not in row["payload"] and "references" not in row["payload"]
    assert stat.S_IMODE(prepared.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((prepared.root/"protocol.json").stat().st_mode) == 0o600
    report = prepared.report()
    assert report["reserved_requests"] == report["completed_jobs"] == 0
    assert report["comparison_ready"] is report["quality_pass_claim"] is False
    assert report["task_delta_percentage_points"] is None
    assert report["task_metrics"]["A"]["tool_tasks"]["denominator"] == 11


def test_prepare_and_authorize_never_overwrite_and_no_approval_opens_nothing(prepared, monkeypatch):
    raw = (prepared.root/"protocol.json").read_bytes()
    with pytest.raises(Exception): asyncio.run(prepared.prepare())
    assert (prepared.root/"protocol.json").read_bytes() == raw
    def forbidden(*args, **kwargs): raise AssertionError("must not open data or a delegate")
    monkeypatch.setattr(module.task_data, "build_live_data", forbidden)
    with pytest.raises(FileNotFoundError): asyncio.run(prepared.execute())
    assert prepared.report()["reserved_requests"] == 0
    approved(prepared, 1)
    approval = (prepared.root/"authorization.json").read_bytes()
    with pytest.raises(Exception): approved(prepared, 2)
    assert (prepared.root/"authorization.json").read_bytes() == approval


@pytest.mark.parametrize("limit", [0, -1, 577, True, 1.5])
def test_invalid_authorization_is_not_a_call_allowance(prepared, limit):
    with pytest.raises(Exception): approved(prepared, limit)
    assert not (prepared.root/"authorization.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda p:p.update(authorized_model_requests=1),
    lambda p:p["initial_intent_requests"][0].update(source_case_id="p02"),
    lambda p:p["initial_rewrite_requests"][0].update(request_bytes_sha256="z"*64),
    lambda p:p["policy"].update(clock="no deadline"),
])
def test_protocol_drift_refuses_before_sends(prepared, mutation):
    p = prepared.verify(); mutation(p)
    (prepared.root/"protocol.json").write_text(json.dumps(p))
    with pytest.raises(Exception): prepared.verify()


def test_atomic_smaller_allowance_and_known_usage_survive_reopen(prepared):
    p=prepared.verify(); approved(prepared,1)
    budget=prepared.ledger(p,send_limit=1); calls=[]
    async def handle(request): calls.append(request); return response()
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01",lease=lease)
        transport=module.ProspectiveTransport(BASE,prepared.root,budget,lease,prepared.verify,
            delegate_factory=lambda frame:httpx.MockTransport(handle))
        async def run():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await send(transport)
                with pytest.raises(TrialTransportError): await send(transport)
        asyncio.run(run())
    assert len(calls)==1 and prepared.ledger(p).status()["attempts"]==1
    value=bench.read_json(prepared.root/"physical-001.json")
    assert value["response"]["usage"]=={"input_tokens":7,"output_tokens":3}
    assert value["delegate_entered"] is True
    assert prepared.report()["paused"] is True


def test_multiple_roles_share_pending_counter_without_spurious_missing_output(prepared):
    p=prepared.verify(); seen=[]
    async def handle(request): await asyncio.sleep(.01); seen.append(1); return response()
    budget=prepared.ledger(p)
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01",lease=lease)
        transport=module.ProspectiveTransport(BASE,prepared.root,budget,lease,prepared.verify,
            delegate_factory=lambda frame:httpx.MockTransport(handle))
        async def run():
            with eval_deadlines(EvaluationDeadlinePolicy()):
                await asyncio.gather(send(transport,agent="general"),send(transport,agent="billing"))
        asyncio.run(run())
    assert len(seen)==2 and prepared.report()["sealed_physical"]==2
    assert not prepared.ledger(p).status()["pending"]


def test_unknown_closed_error_remains_spent_and_readable_without_fabricated_body(prepared):
    p=prepared.verify(); budget=prepared.ledger(p)
    with budget.exclusive_run() as lease:
        budget.start_case("A-p01",lease=lease)
        op=budget.reserve("A-p01","turn-1","intent",None,"a"*64,"b"*64,lease=lease)
        budget.finish(op["operation_id"],"execution_error",None,"network",lease=lease)
    report=prepared.report()
    assert report["reserved_requests"]==report["closed_operations"]==1
    assert report["sealed_physical"]==0 and report["paused"] is True
    assert not (prepared.root/"physical-001.json").exists()


def test_source_proof_is_closed_addition_and_preserves_original_git_blobs(tmp_path, monkeypatch):
    old={"source_commit":module.DESIGN_COMMIT,"root_binding_sha256":module.design._hash(str(tmp_path)),
        "schema_version":"serveflow-prospective-design-freeze-v1","design":{},"plan":[],"suites":[],"dependencies":{},
        "authorized_model_requests":0,"executed_model_requests":0,"external_business_requests":0,
        "live_trial_ready":False,"live_payloads_frozen":False,"sources":{"core/original.py":hashlib.sha256(b"old").hexdigest()}}
    path=tmp_path/"protocol.json";path.write_text(json.dumps(old))
    monkeypatch.setattr(module.design,"ROOT",tmp_path)
    monkeypatch.setattr(module,"DESIGN_HASH",bench.sha256(path))
    monkeypatch.setattr(module.design,"design",lambda:{})
    monkeypatch.setattr(module.design,"_plan",lambda root:[])
    monkeypatch.setattr(module.design,"validate",lambda:[])
    monkeypatch.setattr(module.design,"_dependencies",lambda:{})
    monkeypatch.setattr(module.bench,"BACKEND_ROOT",tmp_path)
    (tmp_path/"core").mkdir();(tmp_path/"core/original.py").write_bytes(b"old")
    current={**old["sources"],**{name:"c"*64 for name in module.EXECUTION_FILES if name.startswith("evaluation/")}}
    monkeypatch.setattr(module.design,"_sources",lambda root:dict(current))
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:SimpleNamespace(stdout=b"a"*40+b" blob 3\nold\n"))
    assert module.design_proof()["original_files"]==1
    current["evaluation/unapproved.py"]="d"*64
    with pytest.raises(Exception,match="unapproved_source_set"):module.design_proof()
    current.pop("evaluation/unapproved.py")
    (tmp_path/"core/original.py").write_bytes(b"edited")
    with pytest.raises(Exception,match="original_source_changed"):module.design_proof()


@pytest.mark.parametrize("failure", ["timeout","signal","spawn-signal","wait-error","system-exit"])
def test_closed_supervisor_reaps_on_every_exit_and_restores_handlers(prepared,monkeypatch,failure):
    approved(prepared); before={number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
    terminated=[];recorded=[]
    class Process:
        pid=987654
        def wait(self,timeout):
            assert timeout==3600
            if failure=="timeout":raise subprocess.TimeoutExpired("closed worker",timeout)
            if failure=="signal":signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            if failure=="system-exit":raise SystemExit(1)
            if failure=="wait-error":raise OSError("controlled wait failure")
            raise AssertionError("spawn signal should skip wait")
    def spawn(command,**kwargs):
        assert command==[module.sys.executable,"-B","-u","-m","evaluation.prospective_trial","_worker"]
        assert kwargs["shell"] is False and kwargs["start_new_session"] is True
        if failure=="spawn-signal":signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        return Process()
    monkeypatch.setattr(module.subprocess,"Popen",spawn)
    monkeypatch.setattr(task_supervisor,"_terminate_group",lambda process:(terminated.append(process.pid) or -15,True,True))
    monkeypatch.setattr(task_supervisor,"_write_stop",lambda *a,**k:recorded.append(k))
    assert module.supervise(prepared)==2
    assert terminated==[987654] and len(recorded)==1
    assert {number:signal.getsignal(number) for number in before}==before


def test_supervisor_no_approval_never_spawns(prepared,monkeypatch):
    monkeypatch.setattr(module.subprocess,"Popen",lambda *a,**k:pytest.fail("no approval must not spawn"))
    with pytest.raises(FileNotFoundError):module.supervise(prepared)


class Data:
    def __init__(self):
        self.proof={"kind":"controlled-fixture"};self.closed=False;self.orders=[]
        self.fixture=bench.read_json(bench.BACKEND_ROOT/"demo_business/fixtures.json")
    def assert_healthy(self):
        if self.closed:raise ValueError("controlled data closed")
    async def order_reader(self,customer_id,order_id):
        self.orders.append((customer_id,order_id))
        return module.task_data._fixture_projection(self.fixture,customer_id,order_id)
    async def knowledge_handler(self,params,context):raise AssertionError("this order-only controlled path should not use RAG")
    async def close(self):self.closed=True


def test_real_sdk_native_handlers_are_counted_and_semantics_stay_pending(prepared,monkeypatch):
    approved(prepared,20);data=Data();counts={};captured=[]
    async def handle(request):
        from evaluation.task_transport import CURRENT
        frame=CURRENT.get();captured.append((frame,json.loads(request.content)))
        if frame.phase=="intent":return response('{"intent":"order_status","confidence":0.99,"reasoning":"controlled"}')
        if frame.phase=="composer":return response("合成订单159元；未付款；支付、发货记录缺失，出库时间未知。")
        assert frame.phase=="agent"
        counts[frame.agent]=counts.get(frame.agent,0)+1
        if counts[frame.agent]==1:
            return response(content=[{"type":"tool_use","id":"controlled_"+frame.agent,
                "name":"read_demo_order","input":{"order_id":"DEMO-1007"}}])
        return response("合成订单159元；未付款；支付、发货记录缺失，出库时间未知。")
    real=task_runtime.execute_arm_case
    async def first_only(arm,case_id,*args,**kwargs):
        if arm!="A" or case_id!="p01":raise ValueError("controlled stop after first full case")
        return await real(arm,case_id,*args,**kwargs)
    monkeypatch.setattr(task_runtime,"execute_arm_case",first_only)
    with pytest.raises(ValueError,match="controlled stop"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame:httpx.MockTransport(handle),controlled_data=data))
    assert data.closed and data.orders
    assert all(customer=="demo-customer-a" and order=="DEMO-1007" for customer,order in data.orders)
    report=prepared.report();row=report["cases"][0]
    assert report["reserved_requests"]==len(captured)>0
    assert report["semantic_reviews_imported"]==0 and report["quality_pass_claim"] is False
    assert row["evidence_kind"]=="controlled-transport"
    assert row["status"] in {"pending_review","failed"}
    assert not any(frame.phase=="judge" for frame,_ in captured)
    assert (prepared.root/"cleanup-complete.json").exists()
    item=bench.read_json(prepared.root/"case-A-p01.json")
    assert item["shared_checks"]["checks"]["tool_task_correct"]
    review={"logical_case_id":"A-p01","artifact_sha256":bench.sha256(prepared.root/"case-A-p01.json"),
        "reviewer":"controlled executor","method":"controlled-fixture",
        "dimensions":dict.fromkeys(module.design.DIMENSIONS,True),"blocking_errors":[],"reading_burden":"some",
        "reason":"Controlled harness only, not a real answer-quality result."}
    prepared.review(review)
    assert prepared.report()["semantic_reviews_imported"]==1
    with pytest.raises(FileExistsError):prepared.review(review)
    assert prepared.report()["comparison_ready"] is False
    assert prepared.report()["physical_resources"]["usage"]["input_tokens"]["known_sum"]==len(captured)*7
    seal=prepared.root/"data-proof-seal.json";original=seal.read_bytes()
    seal.write_text('{}')
    with pytest.raises(Exception,match="data_proof_seal_changed"):prepared.verify()
    seal.write_bytes(original)
    (prepared.root/"cleanup-complete.json").write_text('{}')
    with pytest.raises(Exception,match="cleanup_record_changed"):prepared.verify()


def seed_failed_task_jobs(trial,data):
    """Controlled ledger fixtures only; never count these as model executions."""
    p=trial.verify();proof_hash=canonical_payload_hash(data.proof)
    module.base._write(trial.root,"data-proof.json",data.proof)
    file_hash=bench.sha256(trial.root/"data-proof.json")
    module.base._write(trial.root,"data-proof-seal.json",{"protocol_sha256":bench.sha256(trial.root/"protocol.json"),
        "data_proof_sha256":file_hash})
    budget=trial.ledger(p)
    with budget.exclusive_run() as lease:
        for planned in p["plan"][:24]:
            key=planned["case_id"]
            artifact={"logical_case_id":key,"case_id":key,"source_case_id":key[2:],"arm":key[0],
                "data_kind":"controlled-fixture","evidence_kind":"controlled-transport",
                "protocol_sha256":bench.sha256(trial.root/"protocol.json"),"data_proof_sha256":file_hash,
                "turns":[],"execution_status":"failed"}
            artifact["shared_checks"]=task_runtime.checks(key[2:],artifact)
            module.base._write(trial.root,f"case-{key}.json",artifact)
            budget.start_case(key,lease=lease)
            budget.finish_case(key,bench.sha256(trial.root/f"case-{key}.json"),outcome="failed",lease=lease)


def test_actual_sdk_rag_path_freezes_first_request_and_fills_b_usage(prepared,monkeypatch):
    from evaluation import prospective_retrieval_runtime as rag
    approved(prepared,10);data=Data();seed_failed_task_jobs(prepared,data)
    policies=bench.read_json(bench.BACKEND_ROOT/"knowledge/serveflow-demo-v2.json")
    def source(doc):
        return {**doc,"score":.5,"chunk":0,
            "document_id":hashlib.sha256(json.dumps(["title",doc["title"]],ensure_ascii=True).encode()).hexdigest(),
            "version":hashlib.sha256(doc["content"].encode()).hexdigest()}
    async def knowledge(params,context):return [source(d) for d in policies[:params["top_k"]]]
    data.knowledge_handler=knowledge
    sent=[]
    async def handle(request):
        from evaluation.task_transport import CURRENT
        frame=CURRENT.get();sent.append(frame)
        return response('["第一种新表述","第二种新表述","第三种新表述"]' if frame.phase=="rag_rewrite" else '[0,1,2,3,4]')
    real=rag.execute_retrieval_case
    async def one(case,**kwargs):
        if case["id"]!="q01":raise ValueError("controlled stop after one RAG pair")
        return await real(case,**kwargs)
    monkeypatch.setattr(rag,"execute_retrieval_case",one)
    with pytest.raises(ValueError,match="controlled stop"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame:httpx.MockTransport(handle),controlled_data=data))
    sample=bench.read_json(prepared.root/"case-R-q01.json")
    assert sample["evidence_kind"]=="controlled-transport"
    assert [f.phase for f in sent]==["rag_rewrite","rag_rerank"]
    assert sample["arms"]["A"]["record"]["usage"]=={"input_tokens":0,"output_tokens":0,"physical_requests":0}
    assert sample["arms"]["B"]["record"]["usage"]=={"input_tokens":14,"output_tokens":6,"physical_requests":2}
    assert sample["arm_order"]==["A","B"] and data.closed
    assert prepared.report()["reserved_requests"]==2
    assert prepared.report()["quality_pass_claim"] is False


def test_finished_batch_does_not_reopen_credentials_or_resources(prepared,monkeypatch):
    approved(prepared)
    state={"completed_cases":[row["case_id"] for row in module.fixed_plan()]}
    monkeypatch.setattr(prepared,"ledger",lambda p,**kwargs:SimpleNamespace(status=lambda:state,
        initial_chain_hash="fixed"))
    monkeypatch.setattr(prepared,"verify",lambda:{"plan":module.fixed_plan()})
    monkeypatch.setattr(prepared,"_approval",lambda p:{"max_requests":20})
    monkeypatch.setattr(prepared,"report",lambda:{"already_complete":True})
    def deny(*args,**kwargs):raise AssertionError("finished batch must not reopen")
    monkeypatch.setattr(module.base,"_configuration",deny)
    assert asyncio.run(prepared.execute(delegate_factory=lambda _:None,controlled_data=Data()))=={"already_complete":True}


def test_partial_rag_events_are_shown_separately_not_as_complete_results(prepared):
    p=prepared.verify();budget=prepared.ledger(p)
    module.base._write(prepared.root,"observation-R-q01-001.json",{"event":"arm_completed","case_id":"q01","arm":"A"})
    report=prepared.report()
    assert report["partial_retrieval_observations"][0]["completed_pair"] is False
    assert report["comparison_ready"] is False
