"""Offline preparation for a fixed, current-code A/D task comparison.

The twelve authored tasks are already visible and partly used for repair.
Preparing this protocol grants no model allowance and resumes no old trial.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

from evaluation import benchmark as bench, intent_trial as base, task_data, task_trial as engine, task_recheck as history
from evaluation import current_task_ablation_proof as proof
from evaluation import prospective_task_runtime as runtime
from evaluation.benchmark_metrics import aggregate_resources
from evaluation.current_task_ablation_budget import TaskAblationBudget, fixed_plan, LIMITS, MAX_REQUESTS, MATHEMATICAL_REQUESTS
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.prospective_trial import ProspectiveTrial, _binding, STOP_FILES
from evaluation.task_transport import PHASE_PARAMETERS, AGENT_PARAMETERS
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines


ROOT = proof.ROOT
TRIAL_KIND = "current-task-ablation-v1"
ARMS = ("A", "D")
POLICY = {
    "scope": "12 author-visible seen single-turn tasks x A/D once; full current protected-order bundle, not a component, multi-agent, memory or generalization experiment",
    "contrast": "A: no mandatory fresh read, free order prose and no finite order answers; D: mandatory fresh read + typed order projection + finite question explanations; A may voluntarily read",
    "shared": "same Flash, Skills, role/router, tools and whitelists, source/identity/read-only/policy/diagnostic safeguards, data/index and configured model-stage parameters; isolated runtime/cache per arm-case",
    "prompt_scope": "same base role prompts and Skills; forced tool choice, available schemas and output instructions may differ only as frozen mechanism switches require; first intent bytes identical, not every later request",
    "order": "p01 A,D; p02 D,A; continue alternating through p12; same first intent bytes but independently send in both arms, no output sharing",
    "budget": "new shared hard120; lower explicit allowance allowed; phase mathematical envelope552 is not extra allowance; SDK/HTTP retry0; no Judge or standalone retrieval jobs; task-internal rewrite/rerank counted",
    "clock": "unchanged stage intent8/agent15/rewrite10/rerank30/composer8; SDK ceiling30; audit30; fixed outer3600 for all24 jobs, not a longer model request",
    "errors": "ordinary attributable semantic/structure failures stay and continue; accounting/version/data/auth/quota/network/unknown send pauses without retry or automatic resume",
    "grading": "same source-grounded seven dimensions both arms, actual native-handler evidence required; no template or named-route advantage; output/gold/arm labels excluded from model input",
    "status": "not-prescribed admits accurate success or partial; unknown alone does not force partial; p12 must remain partial and no business write is completed",
    "p11": "safe refusal before a tool is valid, not a required tool task; a claimed not-found lookup still requires actual trusted-customer-B read and no foreign disclosure",
    "coverage": "p01 needs explicit unknown dispatch date; p09 needs direct payment/fulfilment/API permission independence; fields alone do not cover an explicit inference question",
    "natural_wording": "p02 accepts correct natural arithmetic, no compare_amounts or equation template required; p05 finite expiry may express nonpermanence without an exact phrase; UTC or explicitly equivalent timezone acceptable; p04 unasked expiry/renewal fields not mandatory",
    "failure_categories": "source-time omissions recorded separately from wrong amounts, unsupported promises and permission violations; no blanket hallucination count",
    "exposure": "old p01-p09 had real outputs, p01/p07/p09 and related rephrases used for repairs; old p10-p12 not completed in paused batch; all twelve author-visible, shared six policies/ten synthetic orders, not a blind holdout",
    "interpretation": "no delta until every pair is sealed and reviewed; old57 and either six-case trial never fill current slots; a single known-task paired run does not establish population reliability or per-component causality",
}


def cases():
    return runtime.cases()


async def execute_arm_case(arm, case_id, sdk, data, *, ensure_healthy, frame_case_id):
    artifact = await runtime.execute_arm_case(arm, case_id, sdk, data,
        ensure_healthy=ensure_healthy, frame_case_id=frame_case_id)
    artifact["trial_kind"] = TRIAL_KIND
    return artifact


class TaskAblationTransport(PhaseDeadlineTaskHTTPTransport):
    def _validate(self, request, frame):
        if (frame is None or frame.case_id not in {r["case_id"] for r in fixed_plan()}
                or frame.turn_id != "turn-1" or frame.phase == "judge"):
            raise TrialTransportError("protocol")
        return super()._validate(request, frame)


class CurrentTaskAblation:
    protocol_schema = "serveflow-current-task-ablation-protocol-v1"
    authorization_schema = "serveflow-current-task-ablation-authorization-v1"
    _sealed, _review, review = ProspectiveTrial._sealed, ProspectiveTrial._review, ProspectiveTrial.review

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT/"data/benchmarks"):
            base._fail("current_task_ablation_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return TaskAblationBudget(self.root/"ledger", bench.sha256(self.root/"protocol.json"), p["plan"],
                                 initialize=initialize, send_limit=send_limit)

    def expected(self):
        config = base._configuration()
        return {"schema_version":self.protocol_schema, "root_binding_sha256":base._root_hash(self.root),
            "source_proof":proof.source_proof(), "artifacts":proof.artifacts(), "prior_proof":proof.prior_proof(),
            "cases":cases(), "plan":fixed_plan(), "model":base.MODEL, "base_url":config["base_url"],
            "messages_url":messages_url(config["base_url"]), "dependencies":engine._dependencies(),
            "phase_parameters":{k:v for k,v in PHASE_PARAMETERS.items() if k!="judge"},
            "agent_parameters":AGENT_PARAMETERS, "limits":LIMITS, "http_limits":PHASE_HTTP_LIMITS,
            "policy":POLICY, "offline_data_proof":task_data.offline_data_proof(),
            "effective_environment":history.effective_environment(), "authorized_model_requests":0,
            "proposed_model_requests":MAX_REQUESTS, "judge_requests":0, "external_business_requests":0}

    async def prepare(self):
        if self.root.exists(): base._fail("current_task_ablation_root_exists")
        p = self.expected(); initial = []; captured = {}
        for case in p["cases"]:
            value = await base._capture(case,p["base_url"]); value.pop("operation_id")
            captured[case["id"]] = value
        for row in p["plan"]:
            source = row["case_id"][2:]
            initial.append({**deepcopy(captured[source]), "source_case_id":source, "case_id":row["case_id"]})
        p = json.loads(json.dumps({**p,"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=bench.BACKEND_ROOT,text=True).strip(),
            "created_at":base._now(),"initial_intent_requests":initial}))
        self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        base._write(self.root,"protocol.json",p); self.ledger(p,initialize=True)
        return p

    def verify(self):
        p = bench.read_json(base._path(self.root,"protocol.json")); expected = json.loads(json.dumps(self.expected()))
        bench._exact(p,set(expected)|{"git_head","created_at","initial_intent_requests"},"current_task_ablation_protocol_fields")
        if any(p[k]!=v for k,v in expected.items()): base._fail("current_task_ablation_freeze_changed")
        initial = p["initial_intent_requests"]; source = {c["id"]:c for c in p["cases"]}
        if type(initial) is not list or [v.get("case_id") for v in initial]!=[r["case_id"] for r in p["plan"]]:
            base._fail("current_task_ablation_initial_changed")
        pairs = {}
        for row in initial:
            bench._exact(row,{"case_id","source_case_id","input_sha256","payload","payload_sha256","request_bytes_sha256"},"current_task_ablation_initial_fields")
            key = row["case_id"][2:]
            if (row["source_case_id"]!=key or row["input_sha256"]!=canonical_payload_hash(bench.model_input(source[key]))
                    or row["payload_sha256"]!=canonical_payload_hash(row["payload"])
                    or type(row["request_bytes_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}",row["request_bytes_sha256"]) is None):
                base._fail("current_task_ablation_initial_changed")
            value = {k:v for k,v in row.items() if k!="case_id"}
            if key in pairs and pairs[key]!=value: base._fail("current_task_ablation_pair_input_changed")
            pairs[key] = value
        self.ledger(p).status(); self._sealed(p)
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p=self.verify(); state=self.ledger(p).status()
        if (confirmed is not True or type(max_requests) is not int or not 1<=max_requests<=MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference)>500):
            base._fail("current_task_ablation_explicit_authorization_required")
        if state["attempts"] or state["active_case"] or state["completed_cases"]: base._fail("current_task_ablation_already_started")
        value={"schema_version":self.authorization_schema,"reference":reference,"recorded_at":base._now(),
            "protocol_sha256":bench.sha256(self.root/"protocol.json"),"root_binding_sha256":base._root_hash(self.root),
            "initial_chain_hash":self.ledger(p).initial_chain_hash,"max_requests":max_requests,"plan":p["plan"]}
        base._write(self.root,"authorization.json",value); return value

    def _approval(self,p):
        v=bench.read_json(base._path(self.root,"authorization.json"))
        bench._exact(v,{"schema_version","reference","recorded_at","protocol_sha256","root_binding_sha256","initial_chain_hash","max_requests","plan"},"current_task_ablation_approval_fields")
        if (v["schema_version"]!=self.authorization_schema or v["protocol_sha256"]!=bench.sha256(self.root/"protocol.json")
                or v["root_binding_sha256"]!=base._root_hash(self.root) or v["initial_chain_hash"]!=self.ledger(p).initial_chain_hash
                or v["plan"]!=p["plan"] or type(v["max_requests"]) is not int or not 1<=v["max_requests"]<=MAX_REQUESTS):
            base._fail("current_task_ablation_approval_changed")
        return v

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p=self.verify(); approval=self._approval(p)
        if (delegate_factory is None)!=(controlled_data is None): base._fail("current_task_ablation_controlled_pair_required")
        if delegate_factory is None and self.root!=ROOT: base._fail("current_task_ablation_fixed_live_root_required")
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("current_task_ablation_paused")
        if self.ledger(p).status()["completed_cases"]==[r["case_id"] for r in p["plan"]]: return self.report()
        if (self.root/"data-proof.json").exists(): base._fail("current_task_ablation_previous_run_requires_reconciliation")
        ledger=self.ledger(p,send_limit=approval["max_requests"]); data,sdk,transport=controlled_data,None,None
        approval_hash=bench.sha256(self.root/"authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                config=base._configuration(secret=delegate_factory is None)
                if data is None: data=await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind")!="controlled-fixture": base._fail("current_task_ablation_controlled_data_required")
                if controlled_data is None: _binding(data.proof,p["offline_data_proof"])
                data.assert_healthy(); base._write(self.root,"data-proof.json",data.proof)
                data_hash=bench.sha256(self.root/"data-proof.json")
                base._write(self.root,"data-proof-seal.json",{"protocol_sha256":bench.sha256(self.root/"protocol.json"),"data_proof_sha256":data_hash})
                def healthy():
                    if transport is not None and transport.fault: base._fail("current_task_ablation_transport_stopped")
                    self.verify(); self._approval(p)
                    if bench.sha256(self.root/"authorization.json")!=approval_hash or bench.sha256(self.root/"data-proof.json")!=data_hash:
                        base._fail("current_task_ablation_run_identity_changed")
                    data.assert_healthy()
                transport=TaskAblationTransport(config["base_url"],self.root,ledger,lease,healthy,delegate_factory=delegate_factory)
                import httpx
                from anthropic import AsyncAnthropic
                http=httpx.AsyncClient(transport=transport,timeout=SDK_TIMEOUT_S,follow_redirects=False,trust_env=False,headers={"Accept-Encoding":"identity"})
                sdk=AsyncAnthropic(api_key=config.get("api_key","controlled-only"),base_url=config["base_url"],max_retries=0,timeout=SDK_TIMEOUT_S,http_client=http)
                for row in p["plan"]:
                    key=row["case_id"]; healthy(); ledger.start_case(key,lease=lease)
                    with base._quiet_clients(),eval_deadlines(EvaluationDeadlinePolicy()):
                        artifact=await execute_arm_case(key[0],key[2:],sdk,data,ensure_healthy=healthy,frame_case_id=key)
                    healthy(); artifact.update(logical_case_id=key,protocol_sha256=bench.sha256(self.root/"protocol.json"),
                        evidence_kind=transport.kind,data_proof_sha256=data_hash)
                    base._write(self.root,f"case-{key}.json",artifact)
                    ledger.finish_case(key,bench.sha256(self.root/f"case-{key}.json"),outcome=artifact["execution_status"],lease=lease)
            except BaseException:
                if not (self.root/"worker-stop.json").exists(): base._write(self.root,"worker-stop.json",{"reason":"execution_stopped_no_auto_resume","recorded_at":base._now()})
                raise
            finally:
                try:
                    try:
                        if sdk is not None: await task_data._owned(sdk.close())
                    finally:
                        if data is not None: await task_data._owned(data.close())
                    if data is not None: base._write(self.root,"cleanup-complete.json",{"owned_resources_closed":True,"protocol_sha256":bench.sha256(self.root/"protocol.json")})
                except BaseException:
                    if not (self.root/"worker-stop.json").exists(): base._write(self.root,"worker-stop.json",{"reason":"cleanup_failed_no_auto_resume","recorded_at":base._now()})
                    raise
        return self.report()

    def run(self):
        if self.root!=ROOT or Path.cwd()!=bench.BACKEND_ROOT: base._fail("current_task_ablation_fixed_worker_required")
        p=self.verify(); self._approval(p); state=self.ledger(p).status()
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("current_task_ablation_paused")
        if state["completed_cases"]==[r["case_id"] for r in p["plan"]]: return {"exit_code":0,"already_complete":True}
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root/"data-proof.json").exists():
            base._fail("current_task_ablation_previous_run_requires_reconciliation")
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable,"-B","-u","-m","evaluation.current_task_ablation","_worker"],root=ROOT,deadline_s=3600)

    def report(self):
        p=self.verify(); state=self.ledger(p).status(); sealed=self._sealed(p); rows=[]; bycase={c["id"]:c for c in p["cases"]}
        for planned in p["plan"]:
            key=planned["case_id"]; case=bycase[key[2:]]
            row={"logical_case_id":key,"status":"unknown" if state["active_case"]==key else "not_executed","tool_task_applicable":case["expected"]["tool_task"]}
            if key in sealed["cases"]:
                artifact=bench.read_json(self.root/f"case-{key}.json")
                if artifact.get("trial_kind")!=TRIAL_KIND: base._fail("current_task_ablation_artifact_identity_changed")
                observed=runtime.checks(key[2:],artifact)
                if observed!=artifact["shared_checks"]: base._fail("current_task_ablation_checks_changed")
                row.update(artifact_sha256=sealed["cases"][key]["artifact_sha256"],checks=observed,evidence_kind=artifact["evidence_kind"],
                    status="pending_review" if all(observed["checks"].values()) and not observed["blockers"] else "failed")
                row["tool_status"]="not_applicable" if not row["tool_task_applicable"] else "pending_review" if observed["checks"].get("tool_task_correct") else "failed"
                review=self.root/f"review-{key}.json"
                if review.exists():
                    v=bench.read_json(base._path(self.root,review.name)); self._review(v,p)
                    row.update(review=v,status="passed" if row["status"]=="pending_review" and all(v["dimensions"].values()) and not v["blocking_errors"] else "failed")
                    if row["tool_task_applicable"]: row["tool_status"]="passed" if row["tool_status"]=="pending_review" and v["dimensions"]["tool_permissions"] else "failed"
            rows.append(row)
        states=("passed","failed","pending_review","unknown","not_executed")
        def metric(arm,predicate,key="status"):
            group=[r for r in rows if r["logical_case_id"].startswith(arm+"-") and predicate(bycase[r["logical_case_id"][2:]])]
            return {"numerator":sum(r.get(key,r["status"])=="passed" for r in group),"denominator":len(group),"counts":{s:sum(r.get(key,r["status"])==s for r in group) for s in states}}
        metrics={a:{"correct_handling":metric(a,lambda c:True),"supported_completion":metric(a,lambda c:not c["expected"]["boundary"]),
            "tool_tasks":metric(a,lambda c:c["expected"]["tool_task"],"tool_status"),"boundary_handling":metric(a,lambda c:c["expected"]["boundary"])} for a in ARMS}
        indexed={r["logical_case_id"]:r for r in rows}
        pairs=[{"case_id":c["id"],**{a:indexed[a+"-"+c["id"]]["status"] for a in ARMS}} for c in p["cases"]]
        pair_counts={k:0 for k in ("both_passed","only_A_passed","only_D_passed","both_failed","unpaired_pending_unknown_or_unexecuted")}
        for row in pairs:
            key=("unpaired_pending_unknown_or_unexecuted" if any(row[a] not in {"passed","failed"} for a in ARMS)
                 else "both_passed" if row["A"]==row["D"]=="passed" else "only_A_passed" if row["A"]=="passed"
                 else "only_D_passed" if row["D"]=="passed" else "both_failed")
            pair_counts[key]+=1
        paused=state["paused"] or any((self.root/n).exists() for n in STOP_FILES)
        execution_complete=len(sealed["cases"])==24 and not state["pending"] and not paused and (self.root/"cleanup-complete.json").exists()
        comparison_ready=bool(execution_complete and all(r["status"] in {"passed","failed"} and "review" in r for r in rows))
        resources=[]; perarm={a:[] for a in ARMS}
        for op in sealed["operations"].values():
            physical=bench.read_json(self.root/f"physical-{op['attempt']:03d}.json") if op["output_sha256"] else None
            resource={"case_id":str(op["attempt"]),"status":"success" if op["outcome"]=="received" else "failed",
                "latency_ms":physical["latency_ms"] if physical else None,
                "usage":{**(physical["response"]["usage"] if physical else {"input_tokens":None,"output_tokens":None}),"physical_requests":1}}
            resources.append(resource); perarm[op["case_id"][0]].append(resource)
        # An occupied but unclosed claim is not an unattempted zero-cost item.
        # Keep its token/time values unknown while counting the reservation.
        for operation_id in state["pending"]:
            resource={"case_id":"pending:"+operation_id,"status":"unknown","latency_ms":None,
                "usage":{"input_tokens":None,"output_tokens":None,"physical_requests":1}}
            resources.append(resource); perarm[operation_id[0]].append(resource)
        return {"schema_version":"serveflow-current-task-ablation-progress-v1","trial_kind":TRIAL_KIND,"protocol_sha256":bench.sha256(self.root/"protocol.json"),
            "authorization_present":(self.root/"authorization.json").exists(),"trial_request_limit":MAX_REQUESTS,"mathematical_ceiling":MATHEMATICAL_REQUESTS,
            "reserved_requests":state["attempts"],"sealed_physical":sum(v["output_sha256"] is not None for v in sealed["operations"].values()),"pending_physical":len(state["pending"]),
            "planned_jobs":24,"completed_jobs":len(sealed["cases"]),"paused":paused,"cases":rows,"task_metrics":metrics,"task_pairs":pairs,"task_pair_counts":pair_counts,
            "comparison_ready":comparison_ready,"review_ready":bool(execution_complete),"semantic_reviews_imported":sum("review" in r for r in rows),
            "task_delta_percentage_points":(pair_counts["only_D_passed"]-pair_counts["only_A_passed"])*100/12 if comparison_ready else None,
            "physical_resources":aggregate_resources(resources),"arm_resources":{a:aggregate_resources(v) for a,v in perarm.items()},
            "resource_scope":"all reserved attempts, including pending claims with unknown token use/time","pending_usage_unknown_count":len(state["pending"]),
            "all_cases_previously_exposed":True,"quality_pass_claim":False,
            "note":"current protected-order bundle only; all24 positions retained; old outputs never fill slots; not independent component/multi-agent/freshness/generalization evidence"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify","status","report","authorize","run","_worker","review"))
    parser.add_argument("--reference"); parser.add_argument("--max-requests",type=int)
    parser.add_argument("--confirm-explicit-authorization",action="store_true"); parser.add_argument("--review-file",type=Path)
    args=parser.parse_args(); trial=CurrentTaskAblation()
    try:
        if args.command=="prepare": value=asyncio.run(trial.prepare())
        elif args.command=="authorize": value=trial.authorize(args.reference,max_requests=args.max_requests,confirmed=args.confirm_explicit_authorization)
        elif args.command=="run":
            result=trial.run()
            raise SystemExit(0 if result["exit_code"]==0 and not result.get("timed_out",False) and not result.get("cancelled",False) else 2)
        elif args.command=="_worker": value=asyncio.run(trial.execute())
        elif args.command=="review": value=trial.review(bench.read_json(args.review_file))
        elif args.command in {"status","report"}: value=trial.report()
        else: value=trial.verify()
        print(json.dumps({"result":args.command,"reserved_requests":value.get("reserved_requests",0),"proposed_upper_bound":MAX_REQUESTS,
            "authorized_by_prepare":0,"completed_jobs":value.get("completed_jobs",0),"comparison_ready":value.get("comparison_ready",False)}))
    except Exception:
        parser.exit(2,"Current A/D trial stopped; inspect private records; no overwrite, retry or reused allowance.\n")


if __name__=="__main__": main()
