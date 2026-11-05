"""Shared execution wiring; controlled authorization is not a live allowance."""
import asyncio
from copy import deepcopy
import json
import os
import socket
import sys

import httpx
import pytest

from evaluation import benchmark as bench, expanded_trial as module
from evaluation import expanded_task_runtime as tasks, task_supervisor
from evaluation.task_transport import BoundClient, CURRENT, Frame, context
from test_prospective_trial import BASE, Data, response
from test_expanded_observation_runtime import Data as ObservationData


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k): raise AssertionError("expanded tests cannot contact services")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(socket.socket,"connect_ex",denied)
    monkeypatch.setattr(socket,"create_connection",denied)
    monkeypatch.setattr(socket,"getaddrinfo",denied)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source={s:bench.read_json(module.questions._path(bench.BACKEND_ROOT,s))["cases"] for s in module.questions.SUITES}
    monkeypatch.setattr(module,"cases",lambda:deepcopy(source))
    monkeypatch.setattr(module,"source_proof",lambda:{"controlled_source":"a"*64})
    monkeypatch.setattr(module.ancestry,"_current_sources",lambda:{"controlled":"b"*64})
    monkeypatch.setattr(module,"prior_proof",lambda:{"old_records":"c"*64})
    monkeypatch.setattr(module.engine,"_dependencies",lambda:{"controlled":"1"})
    monkeypatch.setattr(module.history,"effective_environment",lambda:{"controlled":True})
    monkeypatch.setattr(module.task_data,"offline_data_proof",lambda:{"controlled":True})
    def configuration(*,secret=False):
        return {"base_url":BASE,"model":module.base.MODEL,**({"api_key":"controlled-only"} if secret else {})}
    monkeypatch.setattr(module.base,"_configuration",configuration)
    trial=module.ExpandedTrial(tmp_path/"controlled-expanded")
    asyncio.run(trial.prepare())
    return trial


def approve(trial,limit=480):
    return trial.authorize("Controlled fixture; not real model authorization",max_requests=limit,confirmed=True)


def controlled_run(trial,monkeypatch,*,stop_at=None):
    """Actual SDK/transport plus real intent/RAG adapters; task wrapper stub.

    Task-specific native execution has its own real-runtime tests. This stub
    intentionally yields failed semantic/tool checks, never a success fixture.
    """
    calls=[];data=ObservationData()
    async def handle(request):
        frame=CURRENT.get();calls.append(frame)
        if frame.phase=="intent":return response('{"intent":"other","confidence":0.5,"reasoning":"controlled"}')
        if frame.phase=="rag_rewrite":return response("[]")
        assert frame.phase=="rag_rerank"
        return response("[0,1,2,3,4,5]")
    async def task_stub(arm,case,sdk,data,*,ensure_healthy,frame_case_id):
        if stop_at==frame_case_id:raise ValueError("predeclared controlled stop")
        ensure_healthy()
        with context(Frame(frame_case_id,"turn-1")):
            await module.base._recognizer(BoundClient(sdk,phase="intent"))._llm_recognize(**bench.model_input(case))
        ensure_healthy()
        artifact={"schema_version":tasks.SCHEMA,"case_id":frame_case_id,"source_case_id":case["id"],"arm":arm,
                  "turns":[],"answer":"Controlled failed task wrapper, not semantic evidence.","execution_status":"failed"}
        artifact["shared_checks"]=tasks.checks(case,artifact)
        return artifact
    monkeypatch.setattr(module,"execute_task",task_stub)
    coro=trial.execute(delegate_factory=lambda frame:httpx.MockTransport(handle),controlled_data=data)
    if stop_at:
        with pytest.raises(ValueError,match="predeclared controlled stop"):asyncio.run(coro)
    else:asyncio.run(coro)
    return calls,data


def test_zero_authorization_prepare_and_initial_payload_plan(prepared):
    p=prepared.verify();r=prepared.report()
    assert len(p["jobs"])==132 and sum(len(c) for c in p["cases"].values())==108
    assert len(p["initial_intent_requests"])==108 and len(p["initial_rewrite_requests"])==24
    assert p["authorized_model_requests"]==r["reserved_requests"]==0
    assert r["comparison_ready"] is r["real_comparison_ready"] is False
    assert all(v["task_delta_percentage_points"] is None for v in r["results_by_split"].values())
    assert r["results_by_split"]["development"]["tasks"]["A"]["correct_handling"]["denominator"]==8
    assert r["results_by_split"]["validation"]["tasks"]["A"]["correct_handling"]["denominator"]==16
    for row in p["initial_intent_requests"]+p["initial_rewrite_requests"]:
        assert set(row["payload"])<={"model","messages","max_tokens","temperature","thinking","stream"}


def test_prepare_authorization_never_overwrite_and_approval_before_resources(prepared,monkeypatch):
    raw=(prepared.root/"protocol.json").read_bytes()
    with pytest.raises(Exception):asyncio.run(prepared.prepare())
    assert (prepared.root/"protocol.json").read_bytes()==raw
    def no_resources(*a,**k):raise AssertionError("must not open resources without grant")
    monkeypatch.setattr(module.task_data,"build_live_data",no_resources)
    with pytest.raises(FileNotFoundError):asyncio.run(prepared.execute())
    approve(prepared,1)
    with pytest.raises(FileExistsError):approve(prepared,480)
    assert prepared._approval(prepared.verify())["max_requests"]==1


def test_shared_full_plan_sdk_accounting_and_reports(prepared,monkeypatch):
    approve(prepared)
    calls,data=controlled_run(prepared,monkeypatch)
    report=prepared.report()
    assert report["completed_jobs"]==132 and report["reserved_requests"]==len(calls)
    assert report["sealed_physical"]==len(calls) and report["pending_physical"]==0 and not report["paused"]
    assert report["artifact_evidence_kinds"]=={"controlled-transport":132}
    assert report["quality_pass_claim"] is report["real_comparison_ready"] is False
    assert report["review_ready"] and not report["comparison_ready"]
    assert {c.case_id for c in calls}=={s["case_id"] for s in module.job_specs()}
    assert all(c.phase!="judge" for c in calls)
    assert report["results_by_split"]["development"]["intents"]["llm_only"]["accuracy"]["denominator"]==24
    assert report["results_by_split"]["validation"]["intents"]["fusion"]["accuracy"]["denominator"]==36
    assert report["physical_resources"]["usage"]["physical_requests"]["known_sum"]==len(calls)
    before=len(calls)
    asyncio.run(prepared.execute(delegate_factory=lambda frame:(_ for _ in ()).throw(AssertionError("must not resend")),controlled_data=Data()))
    assert len(calls)==before


def test_execution_stop_preserves_prefix_no_restart(prepared,monkeypatch):
    approve(prepared)
    calls,_=controlled_run(prepared,monkeypatch,stop_at="A-et01")
    report=prepared.report()
    assert report["completed_jobs"]==24 and report["reserved_requests"]==24
    assert report["paused"] and report["comparison_ready"] is False
    count=len(calls)
    with pytest.raises(Exception):asyncio.run(prepared.execute(delegate_factory=lambda frame:httpx.MockTransport(lambda request:None),controlled_data=Data()))
    assert len(calls)==count
    assert report["results_by_split"]["validation"]["intents"]["llm_only"]["accuracy"]["numerator"]==0


def test_real_sdk_task_native_handlers_join_shared_ledger(prepared,monkeypatch):
    class CombinedData(Data):
        def __init__(self):
            super().__init__();self.knowledge=ObservationData()
        async def knowledge_handler(self,params,context):
            return await self.knowledge.knowledge_handler(params,context)
    approve(prepared,60);data=CombinedData();observed=[]
    async def handle(request):
        frame=CURRENT.get();payload=json.loads(request.content);observed.append(frame)
        if frame.phase=="intent":
            intent="other" if frame.case_id.startswith("I-") else "order_status"
            return response(json.dumps({"intent":intent,"confidence":.99,"reasoning":"controlled"}))
        if frame.phase=="rag_rewrite":return response("[]")
        if frame.phase=="rag_rerank":return response("[0,1,2,3,4,5]")
        assert frame.phase=="agent"
        content=payload["messages"][-1]["content"]
        required=payload.get("tool_choice",{}).get("name")
        if not required and isinstance(content,list) and any(v.get("type")=="tool_result" for v in content):
            return response("Controlled natural summary; semantics unreviewed.")
        tool=required or "read_demo_order"
        assert tool in ("read_demo_order","search_knowledge_base")
        args={"order_id":"DEMO-1008"} if tool=="read_demo_order" else {"query":"包裹记录与签收争议"}
        return response(content=[{"type":"tool_use","id":f"controlled_{len(observed)}","name":tool,"input":args}])
    actual=module.execute_task
    async def stop_after_pair(arm,case,*args,**kwargs):
        if case["id"]=="et02":raise ValueError("predeclared stop after one native pair")
        return await actual(arm,case,*args,**kwargs)
    monkeypatch.setattr(module,"execute_task",stop_after_pair)
    with pytest.raises(ValueError,match="predeclared stop"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame:httpx.MockTransport(handle),controlled_data=data))
    report=prepared.report()
    assert report["completed_jobs"]==26 and report["reserved_requests"]==len(observed)
    for arm in ("A","D"):
        case=bench.read_json(prepared.root/f"case-{arm}-et01.json")
        assert all(case["shared_checks"]["checks"].values())
        assert case["turns"][0]["tool_evidence"]["tool_calls"]>=1
    assert data.orders and all(v==("demo-customer-a","DEMO-1008") for v in data.orders)
    assert data.closed and report["comparison_ready"] is False


def test_tampered_payload_or_authorization_never_opens_delegate(prepared,monkeypatch):
    approve(prepared)
    auth=prepared.root/"authorization.json";v=bench.read_json(auth);v["max_requests"]=481;auth.write_text(json.dumps(v))
    with pytest.raises(Exception):asyncio.run(prepared.execute(delegate_factory=lambda frame:(_ for _ in ()).throw(AssertionError("must not send")),controlled_data=Data()))
    assert prepared.ledger(prepared.verify()).status()["attempts"]==0


@pytest.mark.parametrize("mutation",["approval","data","source"])
def test_local_callback_context_mutation_blocks_before_any_model(prepared,monkeypatch,mutation):
    approve(prepared)
    async def changed(case,sdk,data,*,ensure_healthy,frame_case_id):
        if mutation=="approval":
            path=prepared.root/"authorization.json";v=bench.read_json(path);v["max_requests"]=1;path.write_text(json.dumps(v))
        elif mutation=="data":data.proof["changed"]=True
        else:monkeypatch.setattr(module.ancestry,"_current_sources",lambda:{"changed":"d"*64})
        ensure_healthy()
        raise AssertionError("a changed context must not reach the SDK")
    monkeypatch.setattr(module,"run_intent",changed)
    with pytest.raises(Exception,match="expanded_runtime_context_changed"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame:(_ for _ in ()).throw(AssertionError("must not send")),controlled_data=Data()))
    p=bench.read_json(prepared.root/"protocol.json")
    assert prepared.ledger(p).status()["attempts"]==0
    assert (prepared.root/"worker-stop.json").exists()


def test_restored_context_does_not_erase_observed_failure(prepared,monkeypatch):
    approve(prepared);original=module.ancestry._current_sources
    async def transient(case,sdk,data,*,ensure_healthy,frame_case_id):
        monkeypatch.setattr(module.ancestry,"_current_sources",lambda:{"changed":"d"*64})
        with pytest.raises(Exception,match="expanded_runtime_context_changed"):ensure_healthy()
        monkeypatch.setattr(module.ancestry,"_current_sources",original)
        ensure_healthy()
        raise AssertionError("restoration cannot re-enable a send")
    monkeypatch.setattr(module,"run_intent",transient)
    with pytest.raises(Exception,match="expanded_runtime_context_previously_failed"):
        asyncio.run(prepared.execute(delegate_factory=lambda frame:(_ for _ in ()).throw(AssertionError("must not send")),controlled_data=Data()))
    assert prepared.ledger(bench.read_json(prepared.root/"protocol.json")).status()["attempts"]==0


def test_persisted_review_is_strictly_rechecked(prepared,monkeypatch):
    approve(prepared);controlled_run(prepared,monkeypatch,stop_at="D-et01")
    key="A-et01";artifact=prepared.root/f"case-{key}.json"
    value={"logical_case_id":key,"artifact_sha256":bench.sha256(artifact),"reviewer":"controlled reviewer",
           "method":"controlled-fixture","dimensions":{k:True for k in module.questions.DIMENSIONS},
           "blocking_errors":[],"reading_burden":"none","reason":"Controlled schema test; not real quality."}
    prepared.review(value)
    review=prepared.root/f"review-{key}.json";changed=deepcopy(value);changed["dimensions"]={};review.write_text(json.dumps(changed))
    with pytest.raises(Exception,match="expanded_review_dimensions"):prepared.report()


def test_completed_source_freeze_rejects_later_business_repairs_without_rebinding_hashes():
    # The completed 2b46ce6 trial remains pinned. New business fixes must not
    # silently become that candidate or consume its already closed allowance.
    # Positive closure and mutation checks still run against synthetic pinned
    # trees; the real working tree is intentionally a different version now.
    with pytest.raises(Exception, match="expanded_unapproved_source_set"):
        module.source_proof()
    # The current supervisor has a new narrowly scoped worker; the historical
    # source still matches its original freeze, without rewriting old hashes.
    import hashlib
    import subprocess
    historical = subprocess.check_output(["git", "show", "2b46ce69e3e6dd6705f9119b5ffb9926bc57b299:backend/"+module.HOOK],
                                         cwd=bench.BACKEND_ROOT.parent)
    assert hashlib.sha256(historical).hexdigest()==module.HOOK_HASH
    assert bench.sha256(module.QUESTION_PATH)==module.QUESTION_HASH


def test_old_execution_freeze_is_not_relaxed_for_new_sources():
    from evaluation import current_task_ablation_proof as old
    with pytest.raises(Exception,match="current_task_ablation_unapproved_source_set"):
        old.source_proof()


def test_supervisor_new_worker_has_exact_root_and_same_ceiling(monkeypatch):
    if os.name!="posix":pytest.skip("POSIX supervisor")
    class Reached(Exception):pass
    monkeypatch.setattr(task_supervisor,"_open_root",lambda root:(_ for _ in ()).throw(Reached()))
    command=[sys.executable,"-B","-u","-m","evaluation.expanded_trial","_worker"]
    with pytest.raises(task_supervisor.SupervisorError,match="expanded_worker_root_invalid"):
        task_supervisor.supervise(command,root=module.ROOT.parent/"wrong",deadline_s=3600)
    with pytest.raises(task_supervisor.SupervisorError,match="supervisor_deadline_invalid"):
        task_supervisor.supervise(command,root=module.ROOT,deadline_s=3601)
    with pytest.raises(Reached):task_supervisor.supervise(command,root=module.ROOT,deadline_s=3600)
