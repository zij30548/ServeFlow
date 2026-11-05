"""Fixed expanded evaluation: offline prepare first, separately authorize sends.

No web endpoint uses this runner. Questions/gold, SDK captures, actual dynamic
payloads, physical attempts and manual task reviews remain distinct evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from evaluation import benchmark as bench, benchmark_metrics as metrics
from evaluation import intent_trial as base, expanded_validation as questions
from evaluation import task_data, task_trial as engine, task_recheck as history
from evaluation import diagnostic_repair_proof as ancestry
from evaluation.expanded_budget import ExpandedBudget, ROOT, fixed_plan, job_specs, LIMITS, MAX_REQUESTS
from evaluation.expanded_observation_runtime import run_intent, run_retrieval
from evaluation.expanded_task_runtime import execute_task, checks as task_checks
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.prospective_trial import ProspectiveTrial, STOP_FILES, _binding
from evaluation.prospective_repaired_proof import _git_blobs
from evaluation.task_transport import PHASE_PARAMETERS, AGENT_PARAMETERS
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines


BASELINE = "9617dca8735c343b646e4c8418ae4d8d6982b99b"
QUESTION_HASH = "c7db489c3e778012134493d9b2f0a84d1bdeeb42c6d8dd241a9bb0a7fd8e7f66"
QUESTION_PATH = questions.ROOT / "questions.json"
PREFLIGHT = bench.BACKEND_ROOT / "data/benchmarks/expanded-evaluation-preflight-v1"
HOOK = "evaluation/task_supervisor.py"
HOOK_HASH = "c052399bbd379590e7262a30747c4063a2056f28d351fdd69e33752e53f619f2"
ADDITIONS = tuple(f"{folder}/{name}.py" for folder, names in (
    ("evaluation", ("expanded_budget", "expanded_observation_runtime", "expanded_task_runtime", "expanded_trial")),
    ("tests", ("test_expanded_budget", "test_expanded_observation_runtime", "test_expanded_task_runtime", "test_expanded_trial"))) for name in names)
KIND = "expanded-evaluation-v1"
POLICY = {
    "scope": "108 authored questions, 132 jobs, development before validation; existing synthetic facts, not independent blind or enterprise evaluation",
    "intent": "one shared real observation per question; LLM-only versus unchanged local fusion; resource counted once",
    "intent_parsing": "existing normalized production decisions; parser_failed=false does not prove strict raw JSON/enum conformity; raw whitelist text retained, no post-hoc repair",
    "tasks": "current A/D full protected-order bundle; same current common safeguards; A may read voluntarily; no role-name or exact-template grading advantage",
    "retrieval": "same local index; original-query vector Top3 versus unchanged full rewrite/recall/deduplicate/rerank; no answer generation or refusal fix",
    "history": "provided histories are fixed test inputs, not demonstrated live memory or recovery; trusted identity stays out of model input",
    "freeze": "first intent/rewrite actual SDK bytes frozen; later tool/rerank payloads depend on observed outputs and are bound before reserve, not claimed preknown",
    "execution": "each fixed job once, no automatic retry/resume/repair; semantic/structure failures retained; accounting/version/data/auth/quota/network/unknown sends pause",
    "quota": "one fresh grant <=480; 1212 is mathematical phase envelope only, not permission; all failures/cancelled/unknown attempts occupy allowance",
    "clock": "existing intent8/agent15/rewrite10/rerank30/composer8, SDK30, audit30, in-flight3; fixed outer3600; no longer model deadlines",
    "checks": "full source/history/ledger/payload audit on every upstream admission and job boundary; local callbacks recheck current source/env/config, protocol/approval/data hashes and stop, without replaying all old evidence inside local retrieval deadlines",
    "grading": "frozen labels/gold; task actual native tool evidence plus same seven executor-reviewed dimensions; no Judge; correct refusal/clarification differs from completed business operation",
    "results": "development and validation separate; every planned item remains in denominators; no old output fills new slots or old score changes",
    "known_risk": "et22 quoted order IDs currently resolve ambiguous; retain frozen required own-order read, do not relabel to pass",
}


def cases():
    questions.verify(bench.read_json(QUESTION_PATH))
    if bench.sha256(QUESTION_PATH) != QUESTION_HASH:
        base._fail("expanded_question_freeze_changed")
    return {s: bench.read_json(questions._path(bench.BACKEND_ROOT, s))["cases"] for s in questions.SUITES}


def source_proof():
    result = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
                            cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15)
    if len(result.stdout) > 2*1024*1024:
        base._fail("expanded_source_paths")
    tracked = {v.removeprefix("backend/") for v in result.stdout.decode().split("\0") if v}
    if any(not re.fullmatch(r"[A-Za-z0-9_./-]+", p) or ".." in Path(p).parts for p in tracked):
        base._fail("expanded_source_paths")
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    original = {p for p in tracked if (Path(p).parts[0] in ancestry._RUNTIME_FOLDERS
        and Path(p).suffix in ancestry._SOURCE_SUFFIXES and "__pycache__" not in Path(p).parts)
        or (Path(p).parts[0] == "tests" and Path(p).suffix == ".py") or p in fixed}
    current = ancestry._current_sources()
    if set(current) != original | set(ADDITIONS) or not fixed <= original or set(ADDITIONS) & tracked:
        base._fail("expanded_unapproved_source_set")
    paths, blobs = ["backend/"+p for p in sorted(original)], {}
    for start in range(0, len(paths), 256):
        blobs.update(_git_blobs(BASELINE, paths[start:start+256]))
    original_hashes = {p.removeprefix("backend/"): hashlib.sha256(v).hexdigest() for p, v in blobs.items()}
    if any(current[p] != (HOOK_HASH if p == HOOK else digest) for p, digest in original_hashes.items()):
        base._fail("expanded_business_or_source_changed")
    return {"baseline": BASELINE, "baseline_files": len(original), "current_files": len(current),
            "original_sha256": canonical_payload_hash(original_hashes), "current_sha256": canonical_payload_hash(current),
            "closed_additions": list(ADDITIONS), "exact_hook": {HOOK: HOOK_HASH}, "business_modified": False}


def prior_proof():
    path = PREFLIGHT / "before-preservation.json"
    value = bench.read_json(path)
    root = bench.BACKEND_ROOT / "data/benchmarks"
    actual = {str(p.resolve()) for p in root.rglob("*") if p.is_file()
              and not p.is_relative_to(ROOT) and not p.is_relative_to(PREFLIGHT)}
    if actual != set(value["old_evidence_files"]):
        base._fail("expanded_prior_evidence_set_changed")
    for group in ("old_evidence_files", "original_caches"):
        for name, digest in value[group].items():
            p = Path(name)
            if p.is_symlink() or bench.sha256(p) != digest:
                base._fail("expanded_prior_evidence_changed")
    return {"snapshot_sha256": bench.sha256(path), "files": len(actual), "caches": len(value["original_caches"]),
            "old_allowance_reused": False, "old_scores_recomputed": False}


class ExpandedTransport(PhaseDeadlineTaskHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = bench.read_json(self.root/"protocol.json")
        self._rewrites = {v["case_id"]: v for v in p["initial_rewrite_requests"]}

    def _validate(self, request, frame):
        spec = next((r for r in job_specs() if frame is not None and r["case_id"] == frame.case_id), None)
        if spec is None or frame.turn_id != "turn-1" or frame.phase == "judge":
            raise TrialTransportError("protocol")
        if spec["suite"] == "intents" and (frame.phase != "intent" or frame.agent is not None):
            raise TrialTransportError("protocol")
        if spec["suite"] == "retrieval" and (frame.phase not in ("rag_rewrite", "rag_rerank") or frame.agent is not None):
            raise TrialTransportError("protocol")
        canonical, wire = super()._validate(request, frame)
        if spec["suite"] == "retrieval" and frame.phase == "rag_rewrite":
            expected = self._rewrites.get(frame.case_id)
            if expected is None or (canonical, wire) != (expected["payload_sha256"], expected["request_bytes_sha256"]):
                raise TrialTransportError("protocol")
        return canonical, wire


class ExpandedTrial:
    protocol_schema = "serveflow-expanded-execution-protocol-v1"
    authorization_schema = "serveflow-expanded-execution-authorization-v1"
    _sealed = ProspectiveTrial._sealed

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT/"data/benchmarks"):
            base._fail("expanded_fixed_live_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return ExpandedBudget(self.root/"ledger", bench.sha256(self.root/"protocol.json"), p["plan"],
                              initialize=initialize, send_limit=send_limit)

    def expected(self):
        config = base._configuration()
        return {"schema_version": self.protocol_schema, "root_binding_sha256": base._root_hash(self.root),
            "source_proof": source_proof(), "artifacts": ancestry._current_sources(), "prior_proof": prior_proof(),
            "question_manifest_sha256": QUESTION_HASH, "cases": cases(), "plan": fixed_plan(), "jobs": job_specs(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "dependencies": engine._dependencies(), "phase_parameters": {k:v for k,v in PHASE_PARAMETERS.items() if k != "judge"},
            "agent_parameters": AGENT_PARAMETERS, "limits": LIMITS, "http_limits": PHASE_HTTP_LIMITS, "policy": POLICY,
            "offline_data_proof": task_data.offline_data_proof(), "effective_environment": history.effective_environment(),
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "judge_requests": 0, "external_business_requests": 0}

    async def prepare(self):
        if self.root.exists(): base._fail("expanded_root_exists")
        p, initial, rewrites, observed = self.expected(), [], [], {}
        from evaluation.retrieval_trial import capture_rewrite
        for spec in p["jobs"]:
            case = next(c for c in p["cases"][spec["suite"]] if c["id"] == spec["source_case_id"])
            if case["split"] != spec["split"]: base._fail("expanded_job_split_changed")
            key = spec["source_case_id"]
            if key not in observed:
                observed[key] = await (capture_rewrite(case, p["base_url"]) if spec["suite"] == "retrieval" else base._capture(case, p["base_url"]))
            row = deepcopy(observed[key]); row.pop("operation_id", None)
            row.update(source_case_id=key, case_id=spec["case_id"])
            (rewrites if spec["suite"] == "retrieval" else initial).append(row)
        p = json.loads(json.dumps({**p, "created_at": base._now(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT, text=True).strip(),
            "initial_intent_requests": initial, "initial_rewrite_requests": rewrites}))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        base._write(self.root, "protocol.json", p); self.ledger(p, initialize=True)
        return p

    def verify(self):
        p = bench.read_json(base._path(self.root,"protocol.json")); expected = json.loads(json.dumps(self.expected()))
        bench._exact(p, set(expected)|{"created_at","git_head","initial_intent_requests","initial_rewrite_requests"}, "expanded_protocol_fields")
        if any(p[k] != v for k,v in expected.items()): base._fail("expanded_execution_freeze_changed")
        for field, retrieval in (("initial_intent_requests",False),("initial_rewrite_requests",True)):
            specs = [s for s in p["jobs"] if (s["suite"] == "retrieval") == retrieval]
            rows = p[field]
            if type(rows) is not list or [r.get("case_id") for r in rows] != [s["case_id"] for s in specs]:
                base._fail("expanded_initial_order_changed")
            pair = {}
            for row, spec in zip(rows, specs):
                bench._exact(row,{"case_id","source_case_id","input_sha256","payload","payload_sha256","request_bytes_sha256"},"expanded_initial_fields")
                case = next(c for c in p["cases"][spec["suite"]] if c["id"] == spec["source_case_id"])
                if (row["source_case_id"] != case["id"] or case["split"] != spec["split"]
                        or row["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                        or row["payload_sha256"] != canonical_payload_hash(row["payload"])
                        or type(row["request_bytes_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}",row["request_bytes_sha256"])):
                    base._fail("expanded_initial_binding_changed")
                identity = {k:v for k,v in row.items() if k != "case_id"}
                if case["id"] in pair and pair[case["id"]] != identity: base._fail("expanded_pair_input_changed")
                pair[case["id"]] = identity
        self.ledger(p).status(); sealed = self._sealed(p)
        for key in sealed["cases"]:
            v = bench.read_json(self.root/f"case-{key}.json")
            if v.get("trial_kind") != KIND: base._fail("expanded_artifact_trial_changed")
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p = self.verify(); state = self.ledger(p).status()
        if (confirmed is not True or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference) > 500):
            base._fail("expanded_explicit_authorization_required")
        if state["attempts"] or state["active_case"] or state["completed_cases"]: base._fail("expanded_already_started")
        value = {"schema_version":self.authorization_schema,"reference":reference,"recorded_at":base._now(),
            "protocol_sha256":bench.sha256(self.root/"protocol.json"),"root_binding_sha256":base._root_hash(self.root),
            "initial_chain_hash":self.ledger(p).initial_chain_hash,"max_requests":max_requests,"plan":p["plan"]}
        base._write(self.root,"authorization.json",value); return value

    def _approval(self,p):
        value = bench.read_json(base._path(self.root,"authorization.json"))
        bench._exact(value,{"schema_version","reference","recorded_at","protocol_sha256","root_binding_sha256","initial_chain_hash","max_requests","plan"},"expanded_approval_fields")
        if (value["schema_version"] != self.authorization_schema or value["protocol_sha256"] != bench.sha256(self.root/"protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root) or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash
                or value["plan"] != p["plan"] or type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= MAX_REQUESTS):
            base._fail("expanded_approval_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p = self.verify(); approval = self._approval(p)
        if (delegate_factory is None) != (controlled_data is None): base._fail("expanded_controlled_pair_required")
        if delegate_factory is None and self.root != ROOT: base._fail("expanded_fixed_live_root_required")
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("expanded_paused")
        state = self.ledger(p).status()
        if state["completed_cases"] == [s["case_id"] for s in p["jobs"]]: return self.report()
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root/"data-proof.json").exists():
            base._fail("expanded_previous_run_requires_reconciliation")
        ledger = self.ledger(p,send_limit=approval["max_requests"]); data,sdk,transport = controlled_data,None,None
        approval_hash = bench.sha256(self.root/"authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                config = base._configuration(secret=delegate_factory is None)
                if data is None: data = await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture": base._fail("expanded_controlled_data_required")
                if controlled_data is None: _binding(data.proof,p["offline_data_proof"])
                data.assert_healthy(); base._write(self.root,"data-proof.json",data.proof)
                data_hash = bench.sha256(self.root/"data-proof.json")
                data_canonical = canonical_payload_hash(data.proof)
                protocol_hash = bench.sha256(self.root/"protocol.json")
                runtime_faults = []
                base._write(self.root,"data-proof-seal.json",{"protocol_sha256":bench.sha256(self.root/"protocol.json"),"data_proof_sha256":data_hash})
                def healthy():
                    if runtime_faults: base._fail("expanded_runtime_context_previously_failed")
                    if transport is not None and transport.fault: base._fail("expanded_transport_stopped")
                    self.verify(); self._approval(p)
                    if bench.sha256(self.root/"authorization.json") != approval_hash or bench.sha256(self.root/"data-proof.json") != data_hash:
                        base._fail("expanded_run_identity_changed")
                    data.assert_healthy()
                def runtime_checks():
                    # Local read-only callbacks must not repeatedly charge old
                    # archive/ledger parsing to the knowledge tool's 8s clock.
                    # Actual upstream admission still executes full healthy().
                    if (transport is not None and transport.fault
                            or any((self.root/n).exists() for n in STOP_FILES)
                            or bench.sha256(self.root/"protocol.json") != protocol_hash
                            or bench.sha256(self.root/"authorization.json") != approval_hash
                            or bench.sha256(self.root/"data-proof.json") != data_hash
                            or canonical_payload_hash(data.proof) != data_canonical
                            or ancestry._current_sources() != p["artifacts"]
                            or history.effective_environment() != p["effective_environment"]
                            or base._configuration() != {"base_url":p["base_url"],"model":p["model"]}):
                        base._fail("expanded_runtime_context_changed")
                    data.assert_healthy()
                def runtime_healthy():
                    if runtime_faults: base._fail("expanded_runtime_context_previously_failed")
                    try:
                        runtime_checks()
                    except BaseException:
                        # A->B->A restoration must not erase an observed local
                        # context failure or allow another model send.
                        runtime_faults.append(True)
                        raise
                transport = ExpandedTransport(config["base_url"],self.root,ledger,lease,healthy,delegate_factory=delegate_factory)
                import httpx
                from anthropic import AsyncAnthropic
                http = httpx.AsyncClient(transport=transport,timeout=SDK_TIMEOUT_S,follow_redirects=False,trust_env=False,headers={"Accept-Encoding":"identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key","controlled-only"),base_url=config["base_url"],max_retries=0,timeout=SDK_TIMEOUT_S,http_client=http)
                for spec in p["jobs"]:
                    key = spec["case_id"]; healthy(); ledger.start_case(key,lease=lease)
                    case = next(c for c in p["cases"][spec["suite"]] if c["id"] == spec["source_case_id"])
                    with base._quiet_clients(),eval_deadlines(EvaluationDeadlinePolicy()):
                        call = {"intents":run_intent,"retrieval":run_retrieval}.get(spec["suite"])
                        args = (case,sdk,data) if call else (spec["arm"],case,sdk,data)
                        artifact = await (call or execute_task)(*args,ensure_healthy=runtime_healthy,frame_case_id=key)
                    healthy(); artifact.update(logical_case_id=key,trial_kind=KIND,protocol_sha256=bench.sha256(self.root/"protocol.json"),
                        evidence_kind=transport.kind,data_proof_sha256=data_hash)
                    base._write(self.root,f"case-{key}.json",artifact)
                    outcome = artifact.get("execution_status",artifact.get("status"))
                    if outcome not in ("success","partial","failed"): base._fail("expanded_case_status")
                    ledger.finish_case(key,bench.sha256(self.root/f"case-{key}.json"),outcome=outcome,lease=lease)
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
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT: base._fail("expanded_fixed_worker_required")
        p=self.verify(); self._approval(p); state=self.ledger(p).status()
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("expanded_paused")
        if state["completed_cases"] == [s["case_id"] for s in p["jobs"]]: return {"exit_code":0,"already_complete":True}
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root/"data-proof.json").exists():
            base._fail("expanded_previous_run_requires_reconciliation")
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable,"-B","-u","-m","evaluation.expanded_trial","_worker"],root=ROOT,deadline_s=3600)

    def _validate_review(self,value,p):
        bench._exact(value,{"logical_case_id","artifact_sha256","reviewer","method","dimensions","blocking_errors","reading_burden","reason"},"expanded_review_fields")
        keys={s["case_id"] for s in p["jobs"] if s["suite"]=="tasks"}
        if type(value["logical_case_id"]) is not str or value["logical_case_id"] not in keys:
            base._fail("expanded_review_identity")
        if value["artifact_sha256"] != bench.sha256(self.root/f"case-{value['logical_case_id']}.json"): base._fail("expanded_review_hash")
        bench._exact(value["dimensions"],set(questions.DIMENSIONS),"expanded_review_dimensions")
        if (any(type(v) is not bool for v in value["dimensions"].values()) or value["method"] not in ("executor-rubric-review","controlled-fixture","independent-human-review")
                or value["reading_burden"] not in ("none","some","heavy")): base._fail("expanded_review_invalid")
        bench._text(value["reviewer"],4000); bench._text(value["reason"],4000); bench._strings(value["blocking_errors"])
        return value

    def review(self,value):
        p=self.verify();self._validate_review(value,p)
        if value["logical_case_id"] not in self._sealed(p)["cases"]:base._fail("expanded_review_requires_sealed_case")
        base._write(self.root,f"review-{value['logical_case_id']}.json",value); return value

    def report(self):
        p=self.verify(); state=self.ledger(p).status(); sealed=self._sealed(p); artifacts={}
        for key in sealed["cases"]: artifacts[key]=bench.read_json(self.root/f"case-{key}.json")
        split_reports={}; task_rows=[]
        for split in ("development","validation"):
            selected={s:[c for c in p["cases"][s] if c["split"]==split] for s in questions.SUITES}
            intent={}
            for arm in ("llm_only","fusion"):
                records=[]
                for case in selected["intents"]:
                    v=artifacts.get("I-"+case["id"])
                    if v:
                        decision=v["llm"] if arm=="llm_only" else v["fusion"]
                        records.append({"case_id":case["id"],"status":v["status"],"predicted_intent":decision["intent"],"latency_ms":None,
                                        "usage":{"input_tokens":None,"output_tokens":None,"physical_requests":None}})
                intent[arm]=metrics.score_intent(selected["intents"],records)
            retrieval={}
            for arm in ("baseline","enhanced"):
                records=[artifacts["R-"+c["id"]][arm]["record"] for c in selected["retrieval"] if "R-"+c["id"] in artifacts]
                retrieval[arm]=metrics.score_retrieval(selected["retrieval"],records)
            tasks={}
            for arm in ("A","D"):
                rows=[]
                for case in selected["tasks"]:
                    key=arm+"-"+case["id"];v=artifacts.get(key)
                    row={"case_id":key,"source_case_id":case["id"],"split":split,"status":"unknown" if state["active_case"]==key else "not_executed", "tool_task":case["expected"]["tool_task"],"boundary":case["expected"]["boundary"]}
                    if v:
                        observed=task_checks(case,v)
                        if observed!=v["shared_checks"]: base._fail("expanded_task_checks_changed")
                        row.update(checks=observed,status="pending_review" if all(observed["checks"].values()) and not observed["blockers"] else "failed")
                        review=self.root/f"review-{key}.json"
                        if review.exists():
                            rv=bench.read_json(base._path(self.root,review.name))
                            self._validate_review(rv,p)
                            row.update(review=rv,status="passed" if row["status"]=="pending_review" and all(rv["dimensions"].values()) and not rv["blocking_errors"] else "failed")
                        row["tool_status"]="not_applicable" if not row["tool_task"] else "passed" if observed["checks"].get("tool_task_correct") and "review" in row and row["review"]["dimensions"]["tool_permissions"] else "pending_review" if observed["checks"].get("tool_task_correct") else "failed"
                    rows.append(row)
                task_rows.extend(rows)
                def count(predicate,field="status"):
                    group=[r for r in rows if predicate(r)]
                    return {"numerator":sum(r.get(field)=="passed" for r in group),"denominator":len(group),"counts":{s:sum(r.get(field,r["status"])==s for r in group) for s in ("passed","failed","pending_review","unknown","not_executed")}}
                tasks[arm]={"correct_handling":count(lambda r:True),"supported_completion":count(lambda r:not r["boundary"]),"tool_tasks":count(lambda r:r["tool_task"],"tool_status"),"boundary_handling":count(lambda r:r["boundary"])}
            split_reports[split]={"intents":intent,"retrieval":retrieval,"tasks":tasks}
        paused=bool(state["paused"] or any((self.root/n).exists() for n in STOP_FILES)); complete=len(artifacts)==132 and not state["pending"] and not paused and (self.root/"cleanup-complete.json").exists()
        reviewed=all("review" in r for r in task_rows); resources=[]
        for op in sealed["operations"].values():
            row=bench.read_json(self.root/f"physical-{op['attempt']:03d}.json") if op["output_sha256"] else None
            resources.append({"case_id":str(op["attempt"]),"status":"success" if op["outcome"]=="received" else "failed","latency_ms":row["latency_ms"] if row else None,"usage":{**(row["response"]["usage"] if row else {"input_tokens":None,"output_tokens":None}),"physical_requests":1}})
        for op in state["pending"]: resources.append({"case_id":"pending:"+op,"status":"unknown","latency_ms":None,"usage":{"input_tokens":None,"output_tokens":None,"physical_requests":1}})
        ready=bool(complete and reviewed)
        for split,values in split_reports.items():
            relevant=[r for r in task_rows if r["split"]==split];indexed={r["case_id"]:r for r in relevant}
            pairs={"both_passed":0,"only_A_passed":0,"only_D_passed":0,"both_failed":0,"unpaired_pending_unknown_or_unexecuted":0}
            for case in (c for c in p["cases"]["tasks"] if c["split"]==split):
                a,d=(indexed[arm+"-"+case["id"]]["status"] for arm in ("A","D"))
                key="unpaired_pending_unknown_or_unexecuted" if a not in ("passed","failed") or d not in ("passed","failed") else "both_passed" if a==d=="passed" else "only_A_passed" if a=="passed" else "only_D_passed" if d=="passed" else "both_failed"
                pairs[key]+=1
            values["task_pair_counts"]=pairs
            values["task_delta_percentage_points"]=(pairs["only_D_passed"]-pairs["only_A_passed"])*100/(len(relevant)//2) if ready else None
        evidence_counts=dict(Counter(v["evidence_kind"] for v in artifacts.values()))
        return {"schema_version":"serveflow-expanded-progress-v1","protocol_sha256":bench.sha256(self.root/"protocol.json"),"authorization_present":(self.root/"authorization.json").exists(),
            "reserved_requests":state["attempts"],"sealed_physical":sum(v["output_sha256"] is not None for v in sealed["operations"].values()),"pending_physical":len(state["pending"]),"planned_jobs":132,"completed_jobs":len(artifacts),"paused":paused,
            "results_by_split":split_reports,"task_rows":task_rows,"review_ready":bool(complete),"comparison_ready":ready,"semantic_task_reviews":sum("review" in r for r in task_rows),
            "physical_resources":metrics.aggregate_resources(resources),"resource_scope":"all reserved physical attempts once; shared intent observations are not doubled",
            "artifact_evidence_kinds":evidence_counts,"real_comparison_ready":bool(ready and evidence_counts=={"live-model":132}),
            "quality_pass_claim":False,"live_scope":"known-authored questions with synthetic read-only business data; no browser/real merchant claim"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify","report","authorize","run","_worker","review"))
    parser.add_argument("--reference");parser.add_argument("--max-requests",type=int);parser.add_argument("--confirm-explicit-authorization",action="store_true");parser.add_argument("--review-file",type=Path)
    args=parser.parse_args(argv);trial=ExpandedTrial()
    try:
        if args.command=="prepare":value=asyncio.run(trial.prepare())
        elif args.command=="verify":value=trial.verify()
        elif args.command=="report":value=trial.report()
        elif args.command=="authorize":value=trial.authorize(args.reference,max_requests=args.max_requests,confirmed=args.confirm_explicit_authorization)
        elif args.command=="review":value=trial.review(bench.read_json(args.review_file))
        elif args.command=="run":
            value=trial.run()
            if value.get("exit_code")!=0 or value.get("timed_out") or value.get("cancelled"):base._fail("expanded_supervised_worker_failed")
        else:value=asyncio.run(trial.execute())
        print(json.dumps(value if args.command=="report" else {"result":args.command,"reserved_requests":value.get("reserved_requests",0),"authorization_present":(trial.root/"authorization.json").exists()},ensure_ascii=False,indent=2));return 0
    except Exception:
        print(json.dumps({"error":"expanded_execution_stopped_inspect_private_evidence","no_auto_retry":True}));return 2


if __name__=="__main__":raise SystemExit(main())
