"""One separately authorized execution of the frozen sixteen repair positions.

Preparation and ordinary tests grant zero real calls. No application endpoint
uses this runner. Historical question/protocol/ledger bytes remain immutable.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, benchmark_metrics as metrics
from evaluation import intent_trial as base, repair_validation as preparation
from evaluation import diagnostic_repair_proof as ancestry
from evaluation import task_data, task_trial as engine, task_recheck as history
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import PhaseDeadlineTaskHTTPTransport, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S
from evaluation.prospective_repaired_proof import _git_blobs
from evaluation.prospective_trial import ProspectiveTrial, STOP_FILES, _binding
from evaluation.repair_execution_budget import RepairExecutionBudget, ROOT, LIMITS, MAX_REQUESTS, fixed_plan, job_specs
from evaluation.repair_execution_runtime import rows, checks, execute_task, run_retrieval, SCHEMA, source_rows
from evaluation.task_transport import PHASE_PARAMETERS, AGENT_PARAMETERS


BASELINE = "2d1645a291e348daa80d375050991f7ac2a326bd"
PREPARATION_SHA256 = "18fd46bab820b2dbb9ec13b73fd3f84d35ba5f86c8bf2136b0f5ba63ca103172"
DATASET_SHA256 = "1962adb23bb85d054aa0a0129fe0f0621df4e38382419f9725335ef0be0a6707"
PREFLIGHT = bench.BACKEND_ROOT / "data/benchmarks/repair-execution-preflight-v1"
ADDITIONS = tuple(f"{folder}/{name}.py" for folder, names in (
    ("evaluation", ("repair_execution_budget", "repair_execution_runtime", "repair_execution_trial")),
    ("tests", ("test_repair_execution_budget", "test_repair_execution_runtime", "test_repair_execution_trial"))) for name in names)
# Exact, reviewed evaluation hooks only. No business, prompt or Schema changes.
HOOKS = {
    "evaluation/task_supervisor.py": "acb0392fff1ede75255e0149ee07ebf9f9ad9f70a70086ee0707534017711adb",
    "tests/test_repair_validation.py": "0401acd55588f1f9bbb7af2ad4a6d038c4dbdeb82185c0f21d5bd57e598a521d",
    "tests/test_expanded_trial.py": "3cdcb0ce0519bd94cc852551d02659d3428edb8f649b514cc854bf9128381ebf",
}
KIND = "repair-execution-v1"
POLICY = {
    "scope": "eight current-production Orchestrator tasks and eight same-query vector/current RAG observations; not HTTP/browser/memory/real merchant verification",
    "selection": "exact frozen sixteen rows; known and new wording separate; no labels, expected facts or trusted identity appended to model input",
    "quota": "one fresh protocol-bound allowance <=96 shared by all physical steps; mathematical 200 is not permission; failures/cancelled/unknown attempts never refunded",
    "execution": "each position once; no sample retry/repair/resume; bounded existing role continuations/fallbacks remain observable and counted",
    "clock": "original phase deadlines intent8/agent15/composer8/rewrite10/rerank30; SDK30; audit30; outer900; no new model wait",
    "stop": "answer/structure failures retained; execution/accounting/data/source/auth/quota/rate-limit/network/timeout/cancel/unknown sends pause without automatic recovery",
    "source": "all upstream admissions and job boundaries fully verify source/history/ledger/payload; local checks latch faults including A-B-A",
    "payload": "first sixteen SDK requests match immutable preparation; dynamic native/rerank canonical and wire bytes sealed before atomic reserve/send",
    "review": "same frozen seven dimensions, actual native handlers/permissions, same original status expectations; no Judge, automatic semantic pass or role-name gate",
    "limits": "known author-visible synthetic facts, one observation each; no general accuracy or stable production quality claim; old scores and balances unchanged",
}


def frozen_preparation():
    path = preparation.ROOT / "protocol.json"
    if bench.sha256(path) != PREPARATION_SHA256 or bench.sha256(bench.BACKEND_ROOT / preparation.DATASET) != DATASET_SHA256:
        base._fail("repair_execution_preparation_changed")
    p = bench.read_json(path)
    if (p["candidate_business_commit"] != preparation.BASELINE or p["authorized_model_requests"] != 0
            or p["executed_model_requests"] != 0 or p["live_sender_ready"] is not False
            or p["budget"]["proposed_shared_ceiling"] != MAX_REQUESTS):
        base._fail("repair_execution_preparation_identity")
    preparation.validate()
    return p


def source_proof():
    result = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
        cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15)
    if len(result.stdout) > 2*1024*1024:
        base._fail("repair_execution_source_paths")
    tracked = {s.removeprefix("backend/") for s in result.stdout.decode().split("\0") if s}
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    original = {s for s in tracked if (Path(s).parts[0] in ancestry._RUNTIME_FOLDERS
        and Path(s).suffix in ancestry._SOURCE_SUFFIXES and "__pycache__" not in Path(s).parts)
        or Path(s).parts[0] == "tests" and Path(s).suffix == ".py" or s in fixed}
    current = ancestry._current_sources()
    if set(current) != original | set(ADDITIONS) or set(ADDITIONS) & tracked or not set(HOOKS) <= original:
        base._fail("repair_execution_unapproved_source_set")
    names, blobs = ["backend/"+s for s in sorted(original)], {}
    for start in range(0, len(names), 256):
        blobs.update(_git_blobs(BASELINE, names[start:start+256]))
    historical = {s.removeprefix("backend/"): hashlib.sha256(raw).hexdigest() for s, raw in blobs.items()}
    if any(current[s] != HOOKS.get(s, digest) for s, digest in historical.items()):
        base._fail("repair_execution_business_or_hook_changed")
    p = frozen_preparation()
    if p["source_proof"]["artifacts"] != historical:
        base._fail("repair_execution_prepared_source_mismatch")
    return {"baseline": BASELINE, "business_candidate": preparation.BASELINE,
        "baseline_files": len(original), "current_files": len(current), "closed_additions": list(ADDITIONS),
        "exact_evaluation_hooks": dict(HOOKS), "original_sha256": canonical_payload_hash(historical),
        "current_sha256": canonical_payload_hash(current), "business_modified": False}


def prior_proof():
    path = PREFLIGHT / "before-preservation.json"
    snapshot = bench.read_json(path)
    actual = {str(p.resolve()) for p in (bench.BACKEND_ROOT / "data/benchmarks").rglob("*")
              if p.is_file() and not p.is_relative_to(ROOT) and not p.is_relative_to(PREFLIGHT)}
    if actual != set(snapshot["old_evidence_files"]):
        base._fail("repair_execution_prior_set_changed")
    for group in ("old_evidence_files", "original_caches"):
        for name, digest in snapshot[group].items():
            p = Path(name)
            if p.is_symlink() or bench.sha256(p) != digest:
                base._fail("repair_execution_prior_changed")
    return {"snapshot_sha256": bench.sha256(path), "files": len(actual), "caches": len(snapshot["original_caches"]),
            "preparation_sha256": PREPARATION_SHA256, "previous_batch": frozen_preparation()["prior_proof"],
            "old_allowance_reused": False, "old_scores_changed": False}


def initial_requests(p):
    result = {"initial_intent_requests": [], "initial_rewrite_requests": []}
    for row in p["sdk_first_captures"]:
        key = "initial_intent_requests" if row["suite"] == "tasks" else "initial_rewrite_requests"
        result[key].append({"case_id": row["position_id"], "source_case_id": row["source_case_id"],
            **{name: deepcopy(row[name]) for name in ("input_sha256", "payload", "payload_sha256", "request_bytes_sha256")}})
    return result


def artifact_checks(row, artifact):
    """Check identity/source records, never infer that a reply answered a user."""
    if row["id"].startswith("RT"):
        return checks(row, artifact)
    gates = dict.fromkeys(("artifact_binding", "sources_valid", "records_consistent", "rankings_consistent"), False)
    try:
        case, identity = row["case"], row["id"]
        gates["artifact_binding"] = bool(
            artifact["schema_version"] == SCHEMA and artifact["suite"] == "retrieval"
            and artifact["case_id"] == artifact["logical_case_id"] == identity
            and artifact["source_case_id"] == case["id"] and artifact["group"] == row["group"]
            and artifact["wrapper_sha256"] == canonical_payload_hash(row)
            and artifact["case_sha256"] == canonical_payload_hash(case)
            and artifact["input_sha256"] == canonical_payload_hash(bench.model_input(case))
            and artifact["data_kind"] in {"controlled-fixture", "live-mysql-and-isolated-chroma"})
        projected = {}
        source_keys = {"title", "content", "score", "chunk", "document_id", "version", "content_sha256"}
        for arm in ("baseline", "enhanced"):
            sources = artifact[arm]["sources"]
            if type(sources) is not list or any(type(v) is not dict or set(v) != source_keys for v in sources):
                raise ValueError("invalid_projected_sources")
            projected[arm] = source_rows([{k: v for k, v in source.items() if k != "content_sha256"}
                                          for source in sources])
        gates["sources_valid"] = all(projected[arm] == artifact[arm]["sources"] and len(projected[arm]) <= 3
                                    for arm in projected)
        gates["records_consistent"] = bool(all(
            artifact[arm]["record"]["case_id"] == case["id"]
            and artifact[arm]["record"]["status"] in {"success", "failed"}
            and artifact[arm]["record"]["ranked_titles"] == [v["title"] for v in projected[arm]]
            for arm in projected) and artifact["baseline"]["record"]["status"] == "success"
            and artifact["status"] == artifact["enhanced"]["record"]["status"])
        gates["rankings_consistent"] = all(
            artifact["ranking_observations"][arm] == metrics.score_retrieval([case], [artifact[arm]["record"]])["per_case"][0]
            for arm in projected)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
        pass
    return {"checks": gates, "blockers": [k for k, value in gates.items() if not value],
            "semantic_review_required": True, "quality_pass_claim": False}


class RepairTransport(PhaseDeadlineTaskHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = bench.read_json(self.root / "protocol.json")
        self._rewrites = {v["case_id"]: v for v in p["initial_rewrite_requests"]}

    def _validate(self, request, frame):
        spec = next((s for s in job_specs() if frame is not None and frame.case_id == s["case_id"]), None)
        if spec is None or frame.turn_id != "turn-1" or frame.phase == "judge":
            raise TrialTransportError("protocol")
        if spec["suite"] == "retrieval" and (frame.phase not in {"rag_rewrite", "rag_rerank"} or frame.agent is not None):
            raise TrialTransportError("protocol")
        canonical, wire = super()._validate(request, frame)
        if spec["suite"] == "retrieval" and frame.phase == "rag_rewrite":
            expected = self._rewrites.get(frame.case_id)
            if expected is None or (canonical, wire) != (expected["payload_sha256"], expected["request_bytes_sha256"]):
                raise TrialTransportError("protocol")
        return canonical, wire


class RepairExecutionTrial:
    protocol_schema = "serveflow-repair-execution-protocol-v1"
    authorization_schema = "serveflow-repair-execution-authorization-v1"
    _sealed = ProspectiveTrial._sealed

    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != ROOT and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            base._fail("repair_execution_fixed_root_required")

    def ledger(self, p, *, initialize=False, send_limit=MAX_REQUESTS):
        return RepairExecutionBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), p["plan"],
                                     initialize=initialize, send_limit=send_limit)

    def expected(self):
        preparation_record, config = frozen_preparation(), base._configuration()
        if (config != {"base_url": preparation_record["base_url"], "model": preparation_record["model"]}
                or history.effective_environment() != preparation_record["effective_environment"]
                or engine._dependencies() != preparation_record["dependencies"]
                or task_data.offline_data_proof() != preparation_record["offline_data_proof"]):
            base._fail("repair_execution_candidate_configuration_changed")
        return {"schema_version": self.protocol_schema, "root_binding_sha256": base._root_hash(self.root),
            "preparation_sha256": PREPARATION_SHA256, "dataset_sha256": DATASET_SHA256,
            "source_proof": source_proof(), "artifacts": ancestry._current_sources(), "prior_proof": prior_proof(),
            "cases": rows(), "plan": fixed_plan(), "jobs": job_specs(), "model": base.MODEL,
            "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "dependencies": engine._dependencies(), "effective_environment": history.effective_environment(),
            "offline_data_proof": preparation_record["offline_data_proof"], "limits": LIMITS,
            "http_limits": PHASE_HTTP_LIMITS, "phase_parameters": {k: v for k, v in PHASE_PARAMETERS.items() if k != "judge"},
            "agent_parameters": AGENT_PARAMETERS, "policy": POLICY, "rubric": preparation_record["rubric"],
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "judge_requests": 0, "external_business_requests": 0, **initial_requests(preparation_record)}

    async def prepare(self):
        if self.root.exists():
            base._fail("repair_execution_root_exists")
        p = self.expected()
        # Re-capture the sixteen first requests locally before this new freeze;
        # never loosen or overwrite the immutable question preparation.
        if await preparation.capture_requests(frozen_preparation()) != frozen_preparation()["sdk_first_captures"]:
            base._fail("repair_execution_first_requests_changed")
        if p != self.expected():
            base._fail("repair_execution_preparation_context_changed")
        p = json.loads(json.dumps({**p, "created_at": base._now(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT, text=True).strip()}))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        base._write(self.root, "protocol.json", p)
        self.ledger(p, initialize=True)
        return p

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        expected = json.loads(json.dumps(self.expected()))
        bench._exact(p, set(expected) | {"created_at", "git_head"}, "repair_execution_protocol_fields")
        if any(p[k] != v for k, v in expected.items()):
            base._fail("repair_execution_freeze_changed")
        self.ledger(p).status()
        sealed = self._sealed(p)
        for key in sealed["cases"]:
            artifact = bench.read_json(self.root / f"case-{key}.json")
            row = next(v for v in p["cases"] if v["id"] == key)
            if artifact.get("trial_kind") != KIND or artifact.get("shared_checks") != artifact_checks(row, artifact):
                base._fail("repair_execution_artifact_changed")
        return p

    def authorize(self, reference, *, max_requests, confirmed=False):
        p = self.verify()
        if (confirmed is not True or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS
                or type(reference) is not str or not reference.strip() or len(reference) > 500):
            base._fail("repair_execution_explicit_authorization_required")
        state = self.ledger(p).status()
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root / "data-proof.json").exists():
            base._fail("repair_execution_already_started")
        value = {"schema_version": self.authorization_schema, "reference": reference, "recorded_at": base._now(),
            "protocol_sha256": bench.sha256(self.root / "protocol.json"), "root_binding_sha256": base._root_hash(self.root),
            "initial_chain_hash": self.ledger(p).initial_chain_hash, "max_requests": max_requests, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256",
                            "initial_chain_hash", "max_requests", "plan"}, "repair_execution_approval_fields")
        if (value["schema_version"] != self.authorization_schema
                or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash or value["plan"] != p["plan"]
                or type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= MAX_REQUESTS):
            base._fail("repair_execution_approval_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p, approval = self.verify(), None
        approval = self._approval(p)  # Before secret=True, data, SDK or transport.
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("repair_execution_controlled_pair_required")
        if delegate_factory is None and self.root != ROOT or delegate_factory is not None and self.root == ROOT:
            base._fail("repair_execution_live_controlled_root_mismatch")
        if any((self.root / n).exists() for n in STOP_FILES):
            base._fail("repair_execution_paused")
        state = self.ledger(p).status()
        if state["completed_cases"] == [s["case_id"] for s in p["jobs"]]:
            return self.report()
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root / "data-proof.json").exists():
            base._fail("repair_execution_previous_run_requires_reconciliation")
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        data, sdk, transport = controlled_data, None, None
        approval_hash = bench.sha256(self.root / "authorization.json")
        with ledger.exclusive_run() as lease:
            try:
                config = base._configuration(secret=delegate_factory is None)
                if data is None:
                    data = await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("repair_execution_controlled_data_required")
                if controlled_data is None:
                    _binding(data.proof, p["offline_data_proof"])
                data.assert_healthy()
                base._write(self.root, "data-proof.json", data.proof)
                data_hash = bench.sha256(self.root / "data-proof.json")
                data_canonical, protocol_hash = canonical_payload_hash(data.proof), bench.sha256(self.root / "protocol.json")
                base._write(self.root, "data-proof-seal.json", {"protocol_sha256": protocol_hash, "data_proof_sha256": data_hash})
                faults = []

                def healthy():
                    if faults or transport is not None and transport.fault:
                        base._fail("repair_execution_context_previously_failed")
                    self.verify()
                    self._approval(p)
                    if bench.sha256(self.root / "authorization.json") != approval_hash or bench.sha256(self.root / "data-proof.json") != data_hash:
                        base._fail("repair_execution_run_identity_changed")
                    data.assert_healthy()

                def runtime_healthy():
                    if faults:
                        base._fail("repair_execution_context_previously_failed")
                    try:
                        if (transport is not None and transport.fault or any((self.root / n).exists() for n in STOP_FILES)
                                or bench.sha256(self.root / "protocol.json") != protocol_hash
                                or bench.sha256(self.root / "authorization.json") != approval_hash
                                or bench.sha256(self.root / "data-proof.json") != data_hash
                                or canonical_payload_hash(data.proof) != data_canonical
                                or ancestry._current_sources() != p["artifacts"]
                                or history.effective_environment() != p["effective_environment"]
                                or base._configuration() != {"base_url": p["base_url"], "model": p["model"]}):
                            base._fail("repair_execution_runtime_context_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise

                transport = RepairTransport(config["base_url"], self.root, ledger, lease, healthy, delegate_factory=delegate_factory)
                import httpx
                from anthropic import AsyncAnthropic
                http = httpx.AsyncClient(transport=transport, timeout=SDK_TIMEOUT_S, follow_redirects=False,
                                        trust_env=False, headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                                     max_retries=0, timeout=SDK_TIMEOUT_S, http_client=http)
                for spec in p["jobs"]:
                    key = spec["case_id"]
                    healthy()
                    ledger.start_case(key, lease=lease)
                    row = next(v for v in p["cases"] if v["id"] == key)
                    with base._quiet_clients(), eval_deadlines(EvaluationDeadlinePolicy()):
                        artifact = await (execute_task if spec["suite"] == "tasks" else run_retrieval)(
                            row, sdk, data, ensure_healthy=runtime_healthy, frame_case_id=key)
                    healthy()
                    artifact.update(logical_case_id=key, trial_kind=KIND, protocol_sha256=protocol_hash,
                                    data_proof_sha256=data_hash, evidence_kind=transport.kind)
                    artifact["shared_checks"] = artifact_checks(row, artifact)
                    base._write(self.root, f"case-{key}.json", artifact)
                    outcome = artifact.get("execution_status", artifact.get("status"))
                    if outcome not in {"success", "partial", "failed"}:
                        base._fail("repair_execution_case_status")
                    ledger.finish_case(key, bench.sha256(self.root / f"case-{key}.json"), outcome=outcome, lease=lease)
            except BaseException:
                if not (self.root / "worker-stop.json").exists():
                    base._write(self.root, "worker-stop.json", {"reason": "execution_stopped_no_auto_resume", "recorded_at": base._now()})
                raise
            finally:
                try:
                    try:
                        if sdk is not None:
                            await task_data._owned(sdk.close())
                    finally:
                        if data is not None:
                            await task_data._owned(data.close())
                    if data is not None:
                        base._write(self.root, "cleanup-complete.json", {"owned_resources_closed": True,
                                                                      "protocol_sha256": bench.sha256(self.root / "protocol.json")})
                except BaseException:
                    if not (self.root / "worker-stop.json").exists():
                        base._write(self.root, "worker-stop.json", {"reason": "cleanup_failed_no_auto_resume", "recorded_at": base._now()})
                    raise
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("repair_execution_fixed_worker_required")
        p = self.verify()
        self._approval(p)
        state = self.ledger(p).status()
        if any((self.root / n).exists() for n in STOP_FILES):
            base._fail("repair_execution_paused")
        if state["completed_cases"] == [s["case_id"] for s in p["jobs"]]:
            return {"exit_code": 0, "already_complete": True}
        if state["attempts"] or state["active_case"] or state["completed_cases"] or (self.root / "data-proof.json").exists():
            base._fail("repair_execution_previous_run_requires_reconciliation")
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.repair_execution_trial", "_worker"],
                         root=ROOT, deadline_s=900)

    def review(self, value):
        p = self.verify()
        self._validate_review(value, p)
        if value["logical_case_id"] not in self._sealed(p)["cases"]:
            base._fail("repair_execution_review_requires_sealed_case")
        base._write(self.root, f"review-{value['logical_case_id']}.json", value)
        return value

    def _validate_review(self, value, p):
        bench._exact(value, {"logical_case_id", "artifact_sha256", "reviewer", "method", "dimensions",
                            "blocking_errors", "reading_burden", "reason"}, "repair_execution_review_fields")
        if value["logical_case_id"] not in {r["id"] for r in p["cases"] if r["id"].startswith("RT")}:
            base._fail("repair_execution_review_identity")
        if value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{value['logical_case_id']}.json")):
            base._fail("repair_execution_review_hash")
        bench._exact(value["dimensions"], set(preparation.RUBRIC["tasks"]), "repair_execution_review_dimensions")
        if (any(type(v) is not bool for v in value["dimensions"].values())
                or value["method"] not in {"executor-rubric-review", "controlled-fixture", "independent-human-review"}
                or value["reading_burden"] not in {"none", "some", "heavy"}):
            base._fail("repair_execution_review_invalid")
        artifact = bench.read_json(self.root / f"case-{value['logical_case_id']}.json")
        if ((artifact["evidence_kind"] == "controlled-transport") != (value["method"] == "controlled-fixture")):
            base._fail("repair_execution_review_evidence_kind")
        bench._text(value["reviewer"], 4000)
        bench._text(value["reason"], 4000)
        bench._strings(value["blocking_errors"])

    def report(self):
        p = self.verify()
        state, sealed = self.ledger(p).status(), self._sealed(p)
        artifacts = {k: bench.read_json(self.root / f"case-{k}.json") for k in sealed["cases"]}
        task_rows, retrieval_rows, resources = [], [], []
        for row in p["cases"]:
            if not row["id"].startswith("RT"):
                continue
            key, artifact = row["id"], artifacts.get(row["id"])
            item = {"id": key, "group": row["group"], "status": "unknown" if state["active_case"] == key else "not_executed",
                    "tool_task": row["case"]["expected"]["tool_task"], "boundary": row["case"]["expected"]["boundary"]}
            if artifact:
                observed = artifact_checks(row, artifact)
                operations = [op for op in sealed["operations"].values() if op["case_id"] == key]
                physical_ok = all(op["outcome"] == "received" and op["output_sha256"] is not None for op in operations)
                item.update(checks=observed, physical_structure_valid=physical_ok,
                            status="pending_review" if all(observed["checks"].values()) and not observed["blockers"] and physical_ok else "failed")
                path = self.root / f"review-{key}.json"
                if path.exists():
                    value = bench.read_json(base._path(self.root, path.name))
                    self._validate_review(value, p)
                    item.update(review=value, status="passed" if item["status"] == "pending_review"
                                and all(value["dimensions"].values()) and not value["blocking_errors"] else "failed")
                item["tool_status"] = ("not_applicable" if not item["tool_task"] else
                    "passed" if observed["checks"].get("tool_task_correct") and physical_ok and "review" in item
                    and item["review"]["dimensions"]["tool_permissions"] else
                    "pending_review" if observed["checks"].get("tool_task_correct") and physical_ok else "failed")
            task_rows.append(item)
        grouped = {}
        for group in ("known_regression", "new_wording"):
            selected = [r for r in task_rows if r["group"] == group]
            grouped[group] = {"correct_handling": {"numerator": sum(r["status"] == "passed" for r in selected), "denominator": len(selected)},
                              "statuses": dict(Counter(r["status"] for r in selected))}
        retrieval = {}
        for row in p["cases"]:
            if not row["id"].startswith("RR"):
                continue
            key, artifact = row["id"], artifacts.get(row["id"])
            item = {"id": key, "group": row["group"],
                    "status": "unknown" if state["active_case"] == key else "not_executed"}
            if artifact:
                observed = artifact_checks(row, artifact)
                operations = [op for op in sealed["operations"].values() if op["case_id"] == key]
                physical_ok = all(op["outcome"] == "received" and op["output_sha256"] is not None for op in operations)
                item.update(checks=observed, physical_structure_valid=physical_ok,
                            physical_outcomes=[op["outcome"] for op in operations],
                            rewrite_queries=artifact["method_observation"]["rewrite_queries"],
                            rerank_applied=artifact["method_observation"]["rerank_applied"],
                            status="recorded" if all(observed["checks"].values()) and not observed["blockers"]
                                and physical_ok and artifact["status"] == "success" else "failed")
            retrieval_rows.append(item)
        for group in ("all", "known_regression", "new_wording"):
            selected = [r for r in p["cases"] if r["id"].startswith("RR") and (group == "all" or r["group"] == group)]
            retrieval[group] = {}
            for arm in ("baseline", "enhanced"):
                records = [artifacts[r["id"]][arm]["record"] for r in selected if r["id"] in artifacts]
                retrieval[group][arm] = metrics.score_retrieval([r["case"] for r in selected], records)
        for op in sealed["operations"].values():
            row = bench.read_json(self.root / f"physical-{op['attempt']:03d}.json") if op["output_sha256"] else None
            resources.append({"case_id": str(op["attempt"]), "status": "success" if op["outcome"] == "received" else "failed",
                "latency_ms": row["latency_ms"] if row else None, "usage": {**(row["response"]["usage"] if row else
                {"input_tokens": None, "output_tokens": None}), "physical_requests": 1}})
        for op in state["pending"]:
            resources.append({"case_id": "pending:"+op, "status": "unknown", "latency_ms": None,
                "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": 1}})
        complete = (len(artifacts) == 16 and not state["pending"] and not state["paused"]
                    and not any((self.root / n).exists() for n in STOP_FILES) and (self.root / "cleanup-complete.json").exists())
        reviewed = sum("review" in r for r in task_rows)
        evidence = dict(Counter(v["evidence_kind"] for v in artifacts.values()))
        retrieval_valid = all(r["status"] == "recorded" for r in retrieval_rows)
        authorized = self._approval(p)["max_requests"] if (self.root / "authorization.json").exists() else 0
        return {"schema_version": "serveflow-repair-execution-progress-v1", "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "authorization_present": (self.root / "authorization.json").exists(), "reserved_requests": state["attempts"],
            "authorized_requests": authorized, "maximum_requests": MAX_REQUESTS, "planned_jobs": 16, "completed_jobs": len(artifacts),
            "not_executed_jobs": [r["id"] for r in p["cases"] if r["id"] not in artifacts and state["active_case"] != r["id"]],
            "unknown_jobs": [state["active_case"]] if state["active_case"] else [],
            "sealed_physical": sum(op["output_sha256"] is not None for op in sealed["operations"].values()),
            "pending_physical": len(state["pending"]), "paused": state["paused"] or any((self.root / n).exists() for n in STOP_FILES),
            "task_rows": task_rows, "task_groups": grouped, "retrieval": retrieval, "retrieval_rows": retrieval_rows,
            "semantic_task_reviews": reviewed, "execution_complete": bool(complete),
            "review_ready": bool(complete and retrieval_valid), "comparison_ready": bool(complete and retrieval_valid and reviewed == 8),
            "real_comparison_ready": bool(complete and retrieval_valid and reviewed == 8 and evidence == {"live-model": 16}),
            "artifact_evidence_kinds": evidence, "physical_resources": metrics.aggregate_resources(resources),
            "resource_scope": "all reserved attempts once, including failed/unknown; no inferred tokens",
            "quality_pass_claim": False, "scope": POLICY["scope"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "report", "authorize", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-explicit-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args, trial = parser.parse_args(argv), RepairExecutionTrial()
    try:
        if args.command == "prepare":
            value = asyncio.run(trial.prepare())
        elif args.command == "verify":
            value = trial.verify()
        elif args.command == "report":
            value = trial.report()
        elif args.command == "authorize":
            value = trial.authorize(args.reference, max_requests=args.max_requests, confirmed=args.confirm_explicit_authorization)
        elif args.command == "review":
            value = trial.review(bench.read_json(args.review_file))
        elif args.command == "run":
            value = trial.run()
            if value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled"):
                base._fail("repair_execution_supervised_worker_failed")
        else:
            value = asyncio.run(trial.execute())
        print(json.dumps(value if args.command == "report" else {"result": args.command,
            "reserved_requests": value.get("reserved_requests", 0),
            "authorization_present": (trial.root / "authorization.json").exists()}, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"error": "repair_execution_stopped_inspect_private_evidence", "no_auto_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
