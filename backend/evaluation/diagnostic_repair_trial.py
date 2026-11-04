"""Six fixed production-D checks; offline preparation never authorizes sends.

No model/root/URL overrides. The previous 57-send trial remains paused. This is
targeted repair validation, not a new A/D gain or retrieval benchmark.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from agents.agent_orchestrator import Request
from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from core.order_evidence import resolve_order_id
from evaluation import benchmark as bench, intent_trial as base, prospective_validation as design
from evaluation import diagnostic_repair_proof as proof, task_data, task_trial as engine, task_recheck as history
from evaluation.benchmark_metrics import aggregate_resources
from evaluation.diagnostic_repair_budget import DiagnosticRepairBudget, fixed_plan, MAX_REQUESTS, LIMITS
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.prospective_task_runtime import _source_case
from evaluation.prospective_trial import ProspectiveTrial, _binding, STOP_FILES
from evaluation.quality_arm_runtime import build_arm_runtime
from evaluation.quality_scoring import shared_checks
from evaluation.task_evidence import CaptureTools
from evaluation.task_runtime import _result
from evaluation.task_transport import Frame, context, PHASE_PARAMETERS, AGENT_PARAMETERS


ROOT = proof.ROOT
MANIFEST = bench.BACKEND_ROOT / "evaluation/benchmark/diagnostic-repair-v1.json"
IDS = ("p07", "n01", "p01", "n02", "p09", "n03")
GROUPS = ("seen_regression", "new_authored_rephrase")
POLICY = {
    "scope": "six once-only production-D single turns; three known regressions and three authored rephrases of shared known facts; not blind/generalization or A/D/RAG gain",
    "development_exposure": "all six used controlled wiring; p09/n03 exposed an explicit diagnostic routing gap fixed before freeze; candidate includes that exact routing patch after e42, not a held-out unchanged-e42 trial",
    "parameters": "unchanged Flash, non-thinking/non-streaming and existing role parameters; SDK30/intent8/agent15/rewrite10/rerank30/composer8; audit30/cleanup.25; outer900",
    "errors": "ordinary attributable output/semantic failure stays in denominator and plan continues; version/accounting/source/auth/quota/network/unknown send pauses without retry/resume",
    "budget": "6x23=138 mathematical ceiling, not authorization or expected spend; any lower explicit authorization is checked under the reservation lock",
    "review": "actual native/handler/source checks plus seven-dimensional executor review; no Judge, no semantic auto-pass",
    "payload": "first intent frozen offline; all dependent requests recorded as exact SDK bytes before atomic reservation; no headers/secrets; no labels/gold/group sent",
}


def cases():
    value = bench.read_json(MANIFEST)
    bench._exact(value, {"schema_version", "dataset_id", "provenance", "cases"}, "diagnostic_repair_case_fields")
    if (value["schema_version"] != "serveflow-diagnostic-repair-cases-v1"
            or value["dataset_id"] != "diagnostic-repair-v1"
            or value["provenance"] != "targeted-known-regressions-and-new-authored-rephrases-shared-sources"
            or type(value["cases"]) is not list or [c.get("id") for c in value["cases"]] != list(IDS)):
        base._fail("diagnostic_repair_cases_invalid")
    originals = {c["id"]: c for c in design._cases(bench.BACKEND_ROOT, "tasks")["cases"]}
    seen = set()
    for index, case in enumerate(value["cases"]):
        original = originals[IDS[index-index % 2]]
        bench._exact(case, set(original) | {"group", "base_case_id"}, "diagnostic_repair_case_fields")
        group = GROUPS[index % 2]
        if case["group"] != group or case["base_case_id"] != original["id"]:
            base._fail("diagnostic_repair_group_invalid")
        if any(case[k] != original[k] for k in set(original)-{"id", "input"}):
            base._fail("diagnostic_repair_gold_changed")
        bench._exact(case["input"], set(original["input"]), "diagnostic_repair_input_fields")
        if any(case["input"][k] != original["input"][k] for k in ("history", "customer_id")):
            base._fail("diagnostic_repair_actor_context_changed")
        bench._text(case["input"]["message"], 2000)
        if group == "seen_regression" and case["input"] != original["input"]:
            base._fail("diagnostic_repair_regression_changed")
        normalized = bench._normal(case["input"]["message"])
        if normalized in seen or group == "new_authored_rephrase" and normalized in {
                bench._normal(c["input"]["message"]) for c in originals.values()}:
            base._fail("diagnostic_repair_duplicate_question")
        seen.add(normalized)
    return deepcopy(value["cases"])


def checks(case, artifact):
    observed = shared_checks(_source_case(case), artifact)
    result = deepcopy(observed)
    turns = artifact.get("turns", [])
    turn = turns[0] if type(turns) is list and len(turns) == 1 else {}
    actual = turn.get("result", {})
    status = actual.get("status")
    bound = (artifact.get("schema_version") == "serveflow-diagnostic-repair-artifact-v1"
        and artifact.get("case_id") == "D-"+case["id"] and artifact.get("source_case_id") == case["id"]
        and artifact.get("arm") == "D" and artifact.get("data_kind") in {"controlled-fixture", "live-mysql-and-isolated-chroma"}
        and turn.get("request_id") == "diagnostic-repair-"+case["id"]+"-1")
    eligible = bool(bound and status in {"success", "partial"} and artifact.get("execution_status") == status
                    and type(actual.get("response")) is str and actual["response"].strip())
    result["checks"].update(artifact_binding=bool(result["checks"]["artifact_binding"] and bound),
        execution_eligible=eligible, expected_status=eligible and (
            case["expected"]["expected_status"] == "not-prescribed" or status == case["expected"]["expected_status"]))
    result["blockers"] = list(dict.fromkeys([*(b for b in observed["blockers"] if b != "execution_eligible"),
        *(key for key, passed in result["checks"].items() if not passed)]))
    result["semantic_review_required"] = True
    return result


async def execute_case(case, sdk, data, healthy):
    started = time.perf_counter()
    healthy()
    runtime, _manager = await build_arm_runtime("D", sdk, data)
    actor = bench.model_input(case)
    req = Request(message=actor["message"], history=None, context="", conversation_state={},
        user_id="diagnostic-repair-"+case["input"]["customer_id"], conv_id="diagnostic-repair-"+case["id"],
        request_id="diagnostic-repair-"+case["id"]+"-1", demo_customer_id=case["input"]["customer_id"],
        demo_order_reader=data.order_reader)
    selected = resolve_order_id(req.message)
    capture = CaptureTools(case["id"], 1, call_context=lambda role: context(agent=role),
        expected_order_id=selected.order_id if selected.status == "resolved" else None)
    frame = Frame("D-"+case["id"], "turn-1", evidence_guard=capture.assert_send_allowed)
    capture.install(runtime)
    try:
        with context(frame):
            actual = await runtime.run(req)
            trace = capture.reconcile(actual)
            capture.assert_healthy(); healthy()
    finally:
        capture.restore()
    kind = data.proof.get("kind", "live-mysql-and-isolated-chroma")
    artifact = {"schema_version": "serveflow-diagnostic-repair-artifact-v1", "case_id": frame.case_id,
        "source_case_id": case["id"], "arm": "D", "data_kind": kind,
        "turns": [{"turn_id": "turn-1", "question": req.message, "request_id": req.request_id,
                   "result": _result(actual), "tool_evidence": trace}],
        "answer": actual.response, "execution_status": actual.status, "judge": None,
        "semantic_review": None, "wall_latency_ms": (time.perf_counter()-started)*1000,
        "actor_path_latency_ms": actual.latency_ms}
    artifact["shared_checks"] = checks(case, artifact)
    return artifact


class FocusedTransport(PhaseDeadlineTaskHTTPTransport):
    def _validate(self, request, frame):
        if frame is None or frame.case_id not in {r["case_id"] for r in fixed_plan()} or frame.phase == "judge":
            raise TrialTransportError("protocol")
        return super()._validate(request, frame)


class FocusedTrial:
    protocol_schema = "serveflow-diagnostic-repair-protocol-v1"
    authorization_schema = "serveflow-diagnostic-repair-authorization-v1"
    _sealed, _review, review = ProspectiveTrial._sealed, ProspectiveTrial._review, ProspectiveTrial.review

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT/"data/benchmarks"):
            base._fail("diagnostic_repair_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return DiagnosticRepairBudget(self.root/"ledger", bench.sha256(self.root/"protocol.json"), p["plan"],
                                      initialize=initialize, send_limit=send_limit)

    def expected(self):
        config = base._configuration()
        return {"schema_version": self.protocol_schema, "root_binding_sha256": base._root_hash(self.root),
            "source_proof": proof.source_proof(), "artifacts": proof.artifacts(), "prior_proof": proof.prior_proof(),
            "cases": cases(), "plan": fixed_plan(), "dependencies": engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "phase_parameters": {k:v for k,v in PHASE_PARAMETERS.items() if k != "judge"},
            "agent_parameters": AGENT_PARAMETERS, "limits": LIMITS, "http_limits": PHASE_HTTP_LIMITS,
            "policy": POLICY, "offline_data_proof": task_data.offline_data_proof(),
            "effective_environment": history.effective_environment(), "authorized_model_requests": 0,
            "proposed_model_requests": MAX_REQUESTS, "judge_requests": 0, "external_business_requests": 0}

    async def prepare(self):
        if self.root.exists(): base._fail("diagnostic_repair_root_exists")
        p = self.expected(); initial = []
        for case in p["cases"]:
            value = await base._capture(case, p["base_url"]); value.pop("operation_id")
            initial.append({**value, "source_case_id": case["id"], "case_id": "D-"+case["id"]})
        p = json.loads(json.dumps({**p, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=bench.BACKEND_ROOT, text=True).strip(), "created_at": base._now(), "initial_intent_requests": initial}))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        base._write(self.root, "protocol.json", p); self.ledger(p, initialize=True)
        return p

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        expected = json.loads(json.dumps(self.expected()))
        bench._exact(p, set(expected) | {"git_head", "created_at", "initial_intent_requests"}, "diagnostic_repair_protocol_fields")
        if any(p[k] != v for k,v in expected.items()): base._fail("diagnostic_repair_freeze_changed")
        values = p["initial_intent_requests"]
        if type(values) is not list or [v.get("case_id") for v in values] != [v["case_id"] for v in p["plan"]]:
            base._fail("diagnostic_repair_initial_requests_changed")
        for case, value in zip(p["cases"], values):
            bench._exact(value, {"case_id", "source_case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}, "diagnostic_repair_initial_fields")
            if (value["source_case_id"] != case["id"] or value["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                    or value["payload_sha256"] != canonical_payload_hash(value["payload"])
                    or type(value["request_bytes_sha256"]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", value["request_bytes_sha256"]) is None):
                base._fail("diagnostic_repair_initial_requests_changed")
        self.ledger(p).status(); self._sealed(p)
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p = self.verify(); state = self.ledger(p).status()
        if (confirmed is not True or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference)>500):
            base._fail("diagnostic_repair_explicit_authorization_required")
        if state["attempts"] or state["active_case"] or state["completed_cases"]:
            base._fail("diagnostic_repair_already_started")
        value = {"schema_version": self.authorization_schema, "reference": reference, "recorded_at": base._now(),
            "protocol_sha256": bench.sha256(self.root/"protocol.json"), "root_binding_sha256": base._root_hash(self.root),
            "initial_chain_hash": self.ledger(p).initial_chain_hash, "max_requests": max_requests, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value); return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256", "initial_chain_hash", "max_requests", "plan"}, "diagnostic_repair_approval_fields")
        if (value["schema_version"] != self.authorization_schema or value["protocol_sha256"] != bench.sha256(self.root/"protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root) or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash
                or value["plan"] != p["plan"] or type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= MAX_REQUESTS):
            base._fail("diagnostic_repair_approval_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p = self.verify(); approval = self._approval(p)
        if (delegate_factory is None) != (controlled_data is None): base._fail("diagnostic_repair_controlled_pair_required")
        if delegate_factory is None and self.root != ROOT: base._fail("diagnostic_repair_fixed_live_root_required")
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("diagnostic_repair_paused")
        if self.ledger(p).status()["completed_cases"] == [c["case_id"] for c in p["plan"]]: return self.report()
        if (self.root/"data-proof.json").exists(): base._fail("diagnostic_repair_previous_run_requires_reconciliation")
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        data, sdk, transport = controlled_data, None, None
        approval_hash = bench.sha256(self.root/"authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                config = base._configuration(secret=delegate_factory is None)
                if data is None: data = await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture": base._fail("diagnostic_repair_controlled_data_required")
                if controlled_data is None: _binding(data.proof, p["offline_data_proof"])
                data.assert_healthy()
                base._write(self.root, "data-proof.json", data.proof)
                proof_hash = bench.sha256(self.root/"data-proof.json")
                base._write(self.root, "data-proof-seal.json", {"protocol_sha256": bench.sha256(self.root/"protocol.json"), "data_proof_sha256": proof_hash})
                def healthy():
                    if transport is not None and transport.fault: base._fail("diagnostic_repair_transport_stopped")
                    self.verify(); self._approval(p)
                    if bench.sha256(self.root/"authorization.json") != approval_hash: base._fail("diagnostic_repair_run_identity_changed")
                    if bench.sha256(self.root/"data-proof.json") != proof_hash: base._fail("diagnostic_repair_data_changed")
                    data.assert_healthy()
                transport = FocusedTransport(config["base_url"], self.root, ledger, lease, healthy, delegate_factory=delegate_factory)
                import httpx
                from anthropic import AsyncAnthropic
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False, trust_env=False,
                                        headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                                     max_retries=0, timeout=SDK_TIMEOUT_S, http_client=http)
                for case in p["cases"]:
                    key = "D-"+case["id"]
                    healthy(); ledger.start_case(key, lease=lease)
                    with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                        artifact = await execute_case(case, sdk, data, healthy)
                    healthy()
                    artifact.update(logical_case_id=key, protocol_sha256=bench.sha256(self.root/"protocol.json"),
                                    evidence_kind=transport.kind, data_proof_sha256=proof_hash)
                    base._write(self.root, f"case-{key}.json", artifact)
                    ledger.finish_case(key, bench.sha256(self.root/f"case-{key}.json"), outcome=artifact["execution_status"], lease=lease)
            except BaseException:
                if not (self.root/"worker-stop.json").exists():
                    base._write(self.root, "worker-stop.json", {"reason": "execution_stopped_no_auto_resume", "recorded_at": base._now()})
                raise
            finally:
                try:
                    try:
                        if sdk is not None: await task_data._owned(sdk.close())
                    finally:
                        if data is not None: await task_data._owned(data.close())
                    if data is not None: base._write(self.root, "cleanup-complete.json", {"owned_resources_closed": True,
                        "protocol_sha256": bench.sha256(self.root/"protocol.json")})
                except BaseException:
                    if not (self.root/"worker-stop.json").exists():
                        base._write(self.root, "worker-stop.json", {"reason": "cleanup_failed_no_auto_resume", "recorded_at": base._now()})
                    raise
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT: base._fail("diagnostic_repair_fixed_worker_required")
        p = self.verify(); self._approval(p)
        if any((self.root/n).exists() for n in STOP_FILES): base._fail("diagnostic_repair_paused")
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.diagnostic_repair_trial", "_worker"],
                         root=ROOT, deadline_s=900)

    def report(self):
        p = self.verify(); state = self.ledger(p).status(); sealed = self._sealed(p); rows = []
        for case in p["cases"]:
            key = "D-"+case["id"]
            row = {"logical_case_id": key, "group": case["group"], "status": "unknown" if state["active_case"]==key else "not_executed"}
            if key in sealed["cases"]:
                artifact = bench.read_json(self.root/f"case-{key}.json"); observed = checks(case, artifact)
                if observed != artifact["shared_checks"]: base._fail("diagnostic_repair_checks_changed")
                row.update(artifact_sha256=sealed["cases"][key]["artifact_sha256"], evidence_kind=artifact["evidence_kind"],
                    checks=observed, status="pending_review" if all(observed["checks"].values()) and not observed["blockers"] else "failed")
                row["tool_status"] = "pending_review" if observed["checks"].get("tool_task_correct") else "failed"
                review = self.root/f"review-{key}.json"
                if review.exists():
                    value = bench.read_json(base._path(self.root, review.name)); self._review(value, p)
                    row.update(review=value, status="passed" if row["status"]=="pending_review" and all(value["dimensions"].values()) and not value["blocking_errors"] else "failed")
                    row["tool_status"] = "passed" if row["tool_status"]=="pending_review" and value["dimensions"]["tool_permissions"] else "failed"
            rows.append(row)
        def counts(selected, key="status"):
            return {"planned": len(selected), **{s:sum(r.get(key,r["status"])==s for r in selected) for s in ("passed","failed","pending_review","unknown","not_executed")}}
        resources = []
        for op in sealed["operations"].values():
            physical = bench.read_json(self.root/f"physical-{op['attempt']:03d}.json") if op["output_sha256"] else None
            resources.append({"case_id": str(op["attempt"]), "status": "success" if op["outcome"]=="received" else "failed",
                "latency_ms": physical["latency_ms"] if physical else None,
                "usage": {**(physical["response"]["usage"] if physical else {"input_tokens":None,"output_tokens":None}), "physical_requests":1}})
        paused = state["paused"] or any((self.root/n).exists() for n in STOP_FILES)
        return {"schema_version":"serveflow-diagnostic-repair-progress-v1", "protocol_sha256":bench.sha256(self.root/"protocol.json"),
            "authorization_present":(self.root/"authorization.json").exists(), "mathematical_ceiling":MAX_REQUESTS,
            "reserved_requests":state["attempts"], "sealed_physical":sum(v["output_sha256"] is not None for v in sealed["operations"].values()),
            "pending_physical":len(state["pending"]), "completed_jobs":len(sealed["cases"]), "planned_jobs":6, "paused":paused,
            "cases":rows, "counts":counts(rows), "tool_counts":counts(rows,"tool_status"),
            "groups":{g:counts([r for r in rows if r["group"]==g]) for g in GROUPS},
            "resources":aggregate_resources(resources), "review_ready":len(sealed["cases"])==6 and not paused and not state["pending"] and (self.root/"cleanup-complete.json").exists(),
            "semantic_reviews_imported":sum("review" in r for r in rows), "quality_pass_claim":False,
            "note":"targeted production-D repair check, not A/D gain or RAG quality; all failures/unknowns/unexecuted remain in plan"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","verify","status","report","authorize","run","_worker","review"))
    parser.add_argument("--reference"); parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-explicit-authorization", action="store_true"); parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(); trial = FocusedTrial()
    try:
        if args.command == "prepare": value = asyncio.run(trial.prepare())
        elif args.command == "authorize": value = trial.authorize(args.reference, max_requests=args.max_requests, confirmed=args.confirm_explicit_authorization)
        elif args.command == "run": raise SystemExit(0 if trial.run()["exit_code"] == 0 else 2)
        elif args.command == "_worker": value = asyncio.run(trial.execute())
        elif args.command == "review": value = trial.review(bench.read_json(args.review_file))
        elif args.command in {"status","report"}: value = trial.report()
        else: value = trial.verify()
        print(json.dumps({"result":args.command,"reserved_requests":value.get("reserved_requests",0),"proposed_upper_bound":MAX_REQUESTS,"authorized_by_prepare":0,"completed_jobs":value.get("completed_jobs",0)}))
    except Exception:
        parser.exit(2,"Targeted repair trial stopped; inspect private evidence; no overwrite, retry or automatic allowance.\n")


if __name__ == "__main__": main()
