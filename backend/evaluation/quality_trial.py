"""Fixed four-arm quality trial, with one durable global send budget.

Preparation grants zero sends. A separately recorded applicable authorization
is required; historical unused allowances, outputs and scores are never reused.
The original authored questions/design stay immutable. This tool is not an
application feature, a public model selector or a product quota service.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, intent_trial as base
from evaluation import quality_ablation as design
from evaluation import task_business_regression as business, task_deadline_recheck as clock
from evaluation.benchmark_metrics import REVIEW_METHODS, _resources
from evaluation.intent_transport import canonical_payload_hash, messages_url, _unique_pairs, _reject_constant
from evaluation.quality_budget import QualityBudget, MAX_REQUESTS, LIMITS, fixed_plan
from evaluation.quality_scoring import SCORING_POLICY
from evaluation.task_deadline_transport import DeadlineTaskHTTPTransport
from evaluation.task_recheck import _tree, effective_environment, history


ROOT = bench.BACKEND_ROOT / "data/benchmarks/quality-factorial-v1"
BATCH = "serveflow-quality-factorial-v1"
CANDIDATE = "fresh-read-by-typed-order-evidence-2x2"
V6 = {
    "commit": "d79cc6789ee676c1a5f4d928011502b702fc3829",
    "protocol": "9a044d5d6090c8cf02c7b8de0c25b6f1ddd1286d6782bb6474e86df0e2cdd94b",
    "authorization": "f495db0703fa3e385a0db5d87d5b3ef876676b920ea697dd02fdb4282574192b",
    "journal": "09be5e35795419a3027f36841884a1dd4a455b60a9add599d97575731628ef21",
    "chain": "50335ffdb7d3f43d49f4c77b88798cc686a934c19f8fa2b90e75ac26a42806df",
    "report": "8440e7c26157e7609b340be6a80e0aa99ebe1f24b6b9cc0003634ec945f1026a",
    "tree": "ac3e1cc49c74da818a1b115795eb61411afd7356d30ee25bcf529a9b094bd051",
}
REVIEW_POLICY = {
    "version": "quality-common-semantic-v1",
    "dimensions": list(design.DIMENSIONS),
    "primary_gate": "shared execution/evidence/permission checks plus seven-dimensional semantic review; no blockers",
    "mechanisms_not_primary": ["fresh_read_each_relevant_turn", "exact_projection", "required_agent_names"],
    "natural_summary": "a correct supported natural summary may pass; exact program text is not required",
    "tool_correctness": "actual required tools, trusted identity, target, schema and execution; not merely handler return",
    "judge": "auxiliary only, unchanged 0.75 legacy threshold reported separately",
    "denominators": {"correct_handling": 12, "supported_completion": 8, "tool_tasks": 10, "boundary_handling": 4},
    "boundary": "correct refusal/handoff preparation is not actual refund/human connection completion",
    "source_scope": "known author-visible questions and static synthetic business records; not generalization proof",
    "shared_scoring_policy": deepcopy(SCORING_POLICY),
}
POLICY = {**deepcopy(business.POLICY), "outer_process_deadline_s": 3600,
    "stage_clock": deepcopy(clock.CLOCK_POLICY), "grading_version": REVIEW_POLICY["version"],
    "global_arms_budget": True, "arm_case_runtime_isolation": True,
    "payload_persistence": "validated full SDK JSON and exact UTF-8 request bytes, before reserve/send; no headers",
    "order": "fixed manifest rotation, once per arm/case; no retries or best-output selection",
    "factor_scope": "only force fresh order reads and typed order rendering; all security/policy boundaries shared",
    "history_visibility": "A/C may receive the same explicitly marked static history of validated real tools from this arm-case, never assistant text as evidence; B/D force a fresh read and do not inject that old cache; these necessary message differences are frozen, not described as identical per-message payloads",
    "ordinary_failure": "attributable actor/schema/judge/semantic failure is retained and the fixed plan continues; unaccountable native tool trace or fatal evidence mismatch pauses as an execution-integrity failure",
    "operation_identity": "protocol-bound batch/root plus arm-case, turn, phase, role and ordinal; no arm identity in model content"}
PROTOCOL_FIELDS = business.PROTOCOL_FIELDS | {"design_proof", "arms"}
REQUEST_FIELDS = {"schema_version", "payload_sha256", "request_bytes_sha256", "payload", "wire_text"}


def cases():
    return business.cases()


def arm_cases():
    source = {case["id"]: case for case in cases()}
    return [(row["case_id"][0], source[row["case_id"][2:]], row["case_id"]) for row in fixed_plan()]


def _design_proof():
    tool = design.QualityAblation()
    frozen = tool.verify()
    return {"freeze_sha256": bench.sha256(tool.root / "design-freeze.json"),
            "design_sha256": canonical_payload_hash(design.load_manifest()),
            "manifest_sha256": bench.sha256(design.MANIFEST),
            "preparation_source": frozen["git_head"], "original_preparation_authorized_requests": 0,
            "original_preparation_actual_requests": 0}


def prior_proof():
    """Check old bytes and original Git objects, not current-code old verify."""
    root = clock.ROOT
    if _tree(root) != V6["tree"]:
        base._fail("quality_old_v6_tree_changed")
    for name, key in (("protocol.json", "protocol"), ("authorization.json", "authorization"),
                      ("ledger/journal.jsonl", "journal"), ("reviewed-report.json", "report")):
        if bench.sha256(base._path(root, name)) != V6[key]:
            base._fail("quality_old_v6_record_changed")
    p = bench.read_json(base._path(root, "protocol.json"))
    history._archived_sources(p, "artifacts")
    if (p["git_head"] != V6["commit"] or p["root_binding_sha256"] != base._root_hash(root)
            or p["plan"] != business.fixed_plan()):
        base._fail("quality_old_v6_identity_changed")
    ledger = business.BusinessTaskBudget(root / "ledger", V6["protocol"], p["plan"])
    state, sealed = ledger.status(), ledger.results()
    if (state["attempts"] != 45 or state["paused"] or state["pending"] or state["active_case"] is not None
            or state["completed_cases"] != list(business.CASE_IDS) or state["chain_hash"] != V6["chain"]
            or len(sealed["operations"]) != 45 or len(sealed["cases"]) != 12):
        base._fail("quality_old_v6_ledger_changed")
    for op in sealed["operations"].values():
        if bench.sha256(base._path(root, f"physical-{op['attempt']:03d}.json")) != op["output_sha256"]:
            base._fail("quality_old_v6_physical_changed")
    for key, item in sealed["cases"].items():
        if bench.sha256(base._path(root, f"case-{key}.json")) != item["artifact_sha256"]:
            base._fail("quality_old_v6_case_changed")
    older = business.prior_proof()
    _, v4 = clock._archived(clock.V4)
    _, v5 = clock._archived(clock.V5)
    if (p["prior_proof"]["v4"] != v4 or p["prior_proof"]["v5"] != v5
            or p["prior_proof"]["older_proof_sha256"] != canonical_payload_hash(older)
            or _tree(root) != V6["tree"]):
        base._fail("quality_old_history_changed")
    return {"old_v6": {**deepcopy(V6), "reserved_requests": 45, "sealed_requests": 45},
            "older_proof_sha256": canonical_payload_hash(p["prior_proof"]),
            "old_scores_recomputed": False, "old_allowance_reused": False}


def _artifacts():
    paths = [p for folder in ("evaluation", "tests") for p in (bench.BACKEND_ROOT / folder).glob("*quality*.py")]
    paths += [bench.BACKEND_ROOT / "evaluation/benchmark/quality-ablation-v1.json"]
    extra = bench.BACKEND_ROOT / "tests/test_order_policy_hooks.py"
    if extra.exists():
        paths.append(extra)
    return {**business._artifacts(), **clock._artifacts(),
            **{str(path.relative_to(bench.BACKEND_ROOT)): bench.sha256(path) for path in sorted(paths)}}


def _read_request(root, wire_hash):
    path = base._path(root, "request-" + wire_hash + ".json")
    value = bench.read_json(path)
    bench._exact(value, REQUEST_FIELDS, "quality_request_fields")
    if (value["schema_version"] != "serveflow-quality-request-v1"
            or type(value["wire_text"]) is not str or len(value["wire_text"].encode()) > business.engine.HTTP_LIMITS["request_bytes"]
            or hashlib.sha256(value["wire_text"].encode()).hexdigest() != wire_hash
            or value["request_bytes_sha256"] != wire_hash
            or canonical_payload_hash(value["payload"]) != value["payload_sha256"]
            or json.loads(value["wire_text"], object_pairs_hook=_unique_pairs, parse_constant=_reject_constant) != value["payload"]):
        base._fail("quality_request_changed")
    return value


class QualityHTTPTransport(DeadlineTaskHTTPTransport):
    """Retain full validated payload before send, without credentials/headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = bench.read_json(base._path(self.root, "protocol.json"))
        self._initial = {item["case_id"]: item for item in p["initial_intent_requests"]}

    def _validate(self, request, frame):
        canonical, wire_hash = super()._validate(request, frame)
        if frame.phase == "intent" and frame.turn_id == "turn-1":
            expected = self._initial.get(frame.case_id)
            if expected is None or (canonical, wire_hash) != (expected["payload_sha256"], expected["request_bytes_sha256"]):
                from evaluation.intent_transport import TrialTransportError
                raise TrialTransportError("protocol")
        raw = request.content.decode("utf-8")
        value = {"schema_version": "serveflow-quality-request-v1", "payload_sha256": canonical,
                 "request_bytes_sha256": wire_hash,
                 "payload": json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant),
                 "wire_text": raw}
        path = self.root / ("request-" + wire_hash + ".json")
        if path.exists():
            if _read_request(self.root, wire_hash) != value:
                base._fail("quality_request_changed")
        else:
            base._write(self.root, path.name, value)
        return canonical, wire_hash


class QualityTrial(business.BusinessRegression):
    root_path = ROOT
    batch, candidate = BATCH, CANDIDATE
    protocol_schema = "serveflow-quality-protocol-v1"
    authorization_schema = "serveflow-quality-authorization-v1"
    report_schema = "serveflow-quality-report-v1"

    def artifact_proof(self): return _artifacts()
    def historical_proof(self): return prior_proof()
    def execution_policy(self): return deepcopy(POLICY)
    def http_limits(self): return deepcopy(business.engine.HTTP_LIMITS)
    def make_transport(self, *args, **kwargs): return QualityHTTPTransport(*args, **kwargs)
    def sdk_timeout_s(self): return 8
    def deadline_policy(self): return EvaluationDeadlinePolicy(overhead_limit_s=30, cleanup_grace_s=.25)
    def ledger(self, protocol):
        return QualityBudget(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")), protocol["plan"])

    def _sealed(self, p):
        result = super()._sealed(p)
        for op in result["operations"].values():
            if _read_request(self.root, op["request_bytes_sha256"])["payload_sha256"] != op["payload_sha256"]:
                base._fail("quality_request_changed")
        return result

    async def prepare(self):
        if self.root.exists(): base._fail("quality_root_already_exists")
        manifest, config, values = design.load_manifest(), base._configuration(), cases()
        captures = {}
        for case in values:
            item = await base._capture({"id": case["id"], "input": {"message": case["turns"][0], "history": []}}, config["base_url"])
            item.pop("operation_id")
            captures[case["id"]] = item
        initial = [{**deepcopy(captures[case["id"]]), "case_id": logical, "source_case_id": case["id"], "arm": arm}
                   for arm, case, logical in arm_cases()]
        p = {"schema_version": self.protocol_schema, "batch": self.batch, "candidate": self.candidate,
            "created_at": base._now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": self.artifact_proof(),
            "cases_sha256": canonical_payload_hash({"cases": values}), "dependencies": business.engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(LIMITS), "http_limits": self.http_limits(), "policy": self.execution_policy(),
            "review_policy": deepcopy(REVIEW_POLICY), "phase_parameters": deepcopy(business.engine.PHASE_PARAMETERS),
            "agent_parameters": deepcopy(business.engine.AGENT_PARAMETERS), "plan": fixed_plan(), "initial_intent_requests": initial,
            "offline_data_proof": business.engine.offline_data_proof(), "prior_proof": self.historical_proof(),
            "effective_environment": effective_environment(), "data_proof_sha256": business.repair.data_binding(),
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "design_proof": _design_proof(), "arms": deepcopy(manifest["arms"])}
        p = json.loads(json.dumps(p))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", p)
        QualityBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), p["plan"], initialize=True)
        return p

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        bench._exact(p, PROTOCOL_FIELDS, "quality_protocol_fields")
        config, values = base._configuration(), cases()
        if (p["schema_version"] != self.protocol_schema or p["batch"] != self.batch or p["candidate"] != self.candidate
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["messages_url"] != messages_url(config["base_url"])
                or p["limits"] != LIMITS or p["http_limits"] != self.http_limits()
                or p["policy"] != self.execution_policy() or p["review_policy"] != REVIEW_POLICY
                or p["phase_parameters"] != json.loads(json.dumps(business.engine.PHASE_PARAMETERS))
                or p["agent_parameters"] != json.loads(json.dumps(business.engine.AGENT_PARAMETERS))
                or p["plan"] != fixed_plan() or p["authorized_model_requests"] != 0
                or type(p["authorized_model_requests"]) is not int or p["proposed_model_requests"] != MAX_REQUESTS
                or type(p["proposed_model_requests"]) is not int or p["dependencies"] != business.engine._dependencies()
                or p["effective_environment"] != effective_environment() or p["data_proof_sha256"] != business.repair.data_binding()
                or p["design_proof"] != _design_proof() or p["arms"] != design.load_manifest()["arms"]):
            base._fail("quality_protocol_changed")
        if (p["artifacts"] != self.artifact_proof() or p["cases_sha256"] != canonical_payload_hash({"cases": values})
                or p["offline_data_proof"] != business.engine.offline_data_proof() or p["prior_proof"] != self.historical_proof()):
            base._fail("quality_frozen_evidence_changed")
        if type(p["initial_intent_requests"]) is not list or len(p["initial_intent_requests"]) != 48:
            base._fail("quality_initial_requests_changed")
        shared = {}
        for item, (arm, case, logical) in zip(p["initial_intent_requests"], arm_cases()):
            if (type(item) is not dict or set(item) != {"case_id", "source_case_id", "arm", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or (item["arm"], item["case_id"], item["source_case_id"]) != (arm, logical, case["id"])
                    or item["input_sha256"] != canonical_payload_hash({"message": case["turns"][0], "history": []})
                    or item["payload_sha256"] != canonical_payload_hash(item["payload"])
                    or type(item["request_bytes_sha256"]) is not str or re.fullmatch(r"[a-f0-9]{64}", item["request_bytes_sha256"]) is None):
                base._fail("quality_initial_requests_changed")
            pair = {k: item[k] for k in ("input_sha256", "payload", "payload_sha256", "request_bytes_sha256")}
            if case["id"] in shared and shared[case["id"]] != pair:
                base._fail("quality_arm_initial_payload_mismatch")
            shared[case["id"]] = pair
        self.ledger(p).status(); self._sealed(p)
        actual = self.root / "data-proof.json"
        if actual.exists() and canonical_payload_hash(bench.read_json(base._path(self.root, actual.name))) != p["data_proof_sha256"]:
            base._fail("quality_data_identity_changed")
        return p

    def authorize(self, reference, *, confirmed=False):
        p = self.verify()
        if confirmed is not True or type(reference) is not str or not reference.strip() or len(reference) > 500:
            base._fail("quality_explicit_authorization_required")
        ledger = self.ledger(p)
        if ledger.status()["attempts"] or ledger.status()["active_case"]: base._fail("quality_already_started")
        value = {"schema_version": self.authorization_schema, "reference": reference, "recorded_at": base._now(),
            "protocol_sha256": bench.sha256(self.root / "protocol.json"), "root_binding_sha256": base._root_hash(self.root),
            "initial_chain_hash": ledger.initial_chain_hash, "max_requests": MAX_REQUESTS, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256",
                            "initial_chain_hash", "max_requests", "plan"}, "quality_authorization_fields")
        if (value["schema_version"] != self.authorization_schema or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash or type(value["max_requests"]) is not int
                or value["max_requests"] != MAX_REQUESTS or value["plan"] != p["plan"]):
            base._fail("quality_authorization_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        from anthropic import AsyncAnthropic
        import httpx
        from evaluation.quality_arm_runtime import execute_arm_case
        from evaluation.quality_scoring import shared_checks
        p = self.verify(); self._approval(p)
        if (delegate_factory is None) != (controlled_data is None): base._fail("quality_incomplete_controlled_boundary")
        if any((self.root / name).exists() for name in ("stopped.json", "supervisor-stop.json", "worker-stop.json")):
            base._fail("quality_paused")
        config = base._configuration(secret=delegate_factory is None)
        ledger, data, sdk = self.ledger(p), controlled_data, None
        with ledger.exclusive_run() as lease:
            try:
                if data is None:
                    url = business.engine._mysql_url()
                    if not url: base._fail("dedicated_eval_mysql_not_configured")
                    data = await business.engine.build_live_data(url)
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("quality_controlled_data_required")
                data.assert_healthy()
                if canonical_payload_hash(data.proof) != p["data_proof_sha256"]: base._fail("quality_data_identity_changed")
                if not (self.root / "data-proof.json").exists(): base._write(self.root, "data-proof.json", data.proof)
                elif bench.read_json(base._path(self.root, "data-proof.json")) != data.proof: base._fail("quality_data_proof_changed")
                proof_hash = bench.sha256(self.root / "data-proof.json")
                def admission():
                    self.verify(); self._approval(p)
                    if bench.sha256(base._path(self.root, "data-proof.json")) != proof_hash: base._fail("quality_data_proof_changed")
                    data.assert_healthy()
                transport = self.make_transport(config["base_url"], self.root, ledger, lease, admission, delegate_factory=delegate_factory)
                http = httpx.AsyncClient(transport=transport, timeout=self.sdk_timeout_s(), follow_redirects=False, trust_env=False,
                                        headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                                     max_retries=0, timeout=self.sdk_timeout_s(), http_client=http)
                def healthy():
                    if transport.fault: base._fail("quality_transport_stopped")
                    admission()
                for arm, case, logical in arm_cases():
                    if logical in ledger.status()["completed_cases"]: continue
                    healthy(); ledger.start_case(logical, lease=lease)
                    with base._quiet_clients(), eval_deadlines(self.deadline_policy()):
                        artifact = await execute_arm_case(arm, case, sdk, data, ensure_healthy=healthy, frame_case_id=logical)
                    artifact.update({"arm": arm, "logical_case_id": logical,
                        "protocol_sha256": bench.sha256(self.root / "protocol.json"), "evidence_kind": transport.kind,
                        "data_proof_sha256": proof_hash, "grading_version": REVIEW_POLICY["version"]})
                    artifact["shared_checks"] = shared_checks(case, artifact)
                    name = f"case-{logical}.json"; base._write(self.root, name, artifact)
                    ledger.finish_case(logical, bench.sha256(self.root / name),
                        outcome="success" if artifact["execution_status"] == "success" else "failed", lease=lease)
            except BaseException as error:
                if not (self.root / "worker-stop.json").exists():
                    from evaluation.task_evidence import EvidenceError, FATAL_EVIDENCE_CODES
                    stopped = {"schema_version": "serveflow-quality-worker-stop-v1",
                        "reason": "cancelled" if isinstance(error, asyncio.CancelledError) else "execution_stopped", "recorded_at": base._now()}
                    if isinstance(error, EvidenceError) and error.reason in FATAL_EVIDENCE_CODES:
                        stopped["evidence_error"] = error.reason
                    base._write(self.root, "worker-stop.json", stopped)
                raise
            finally:
                try:
                    if sdk is not None: await sdk.close()
                finally:
                    if data is not None: await data.close()
        return self.report()

    def _review(self, value):
        bench._exact(value, {"case_id", "artifact_sha256", "reviewer", "method", "dimensions", "reason",
                            "blocking_errors", "reading_burden"}, "quality_review_fields")
        if value["case_id"] not in {row["case_id"] for row in fixed_plan()}: base._fail("quality_review_case")
        for key in ("reviewer", "method", "reason"): bench._text(value[key], 4000)
        bench._strings(value["blocking_errors"])
        if value["method"] not in REVIEW_METHODS or value["reading_burden"] not in {"none", "some", "heavy"}:
            base._fail("quality_review_method_or_burden")
        bench._exact(value["dimensions"], set(design.DIMENSIONS), "quality_review_dimensions")
        if not all(type(v) is bool for v in value["dimensions"].values()): base._fail("quality_review_dimensions")
        if value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{value['case_id']}.json")):
            base._fail("quality_review_binding")

    def report(self):
        from evaluation.quality_scoring import shared_checks
        p = self.verify(); ledger = self.ledger(p); state, sealed = ledger.status(), self._sealed(p)
        rows = []
        for arm, case, logical in arm_cases():
            row = {"arm": arm, "case_id": case["id"], "logical_case_id": logical,
                   "status": "unknown" if state["active_case"] == logical else "not_executed"}
            if logical in sealed["cases"]:
                artifact = bench.read_json(base._path(self.root, f"case-{logical}.json"))
                check = shared_checks(case, artifact)
                if (artifact.get("shared_checks") != check or artifact.get("arm") != arm
                        or artifact.get("logical_case_id") != logical or artifact.get("case_id") != case["id"]
                        or artifact.get("protocol_sha256") != bench.sha256(self.root / "protocol.json")
                        or artifact.get("grading_version") != REVIEW_POLICY["version"]):
                    base._fail("quality_case_binding_changed")
                hard = all(check["checks"].values()) and not check["blockers"]
                judges = [turn["judge_overall"] for turn in artifact["turns"]]
                row.update(shared_checks=check, status="pending_review" if hard else "failed", judge_scores=judges,
                           execution_status=artifact["execution_status"], data_kind=artifact["data_kind"],
                           wall_latency_ms=artifact["wall_latency_ms"], actor_path_latency_ms=artifact["actor_path_latency_ms"])
                review_path = self.root / f"review-{logical}.json"
                if review_path.exists():
                    review = bench.read_json(base._path(self.root, review_path.name)); self._review(review)
                    row["review"] = {**review, "review_sha256": bench.sha256(review_path)}
                    row["status"] = "passed" if hard and all(review["dimensions"].values()) and not review["blocking_errors"] else "failed"
                legacy_hard = (artifact["execution_status"] == "success" and not artifact["deterministic"]["blockers"]
                               and all(artifact["deterministic"]["checks"].values()))
                row["legacy_success_role_judge_status"] = ("pending_review" if "review" not in row else "passed"
                    if legacy_hard and row["status"] == "passed" and artifact["judge_ready"]
                       and all(type(s) in (int, float) and s >= .75 for s in judges) else "failed")
            rows.append(row)
        def metric(items, predicate, *, require_answer=True):
            def status(row):
                if not require_answer and "shared_checks" in row:
                    # A wrong financial sentence is an answer failure, not
                    # evidence that an otherwise correct tool call was wrong.
                    # Keep this task-level tool dimension independently scored.
                    if "review" not in row: return "pending_review"
                    return "passed" if predicate(row) else "failed"
                return ("passed" if row["status"] == "passed" and predicate(row) else
                        "failed" if row["status"] in {"passed", "failed"} else row["status"])
            counts = Counter(status(row) for row in items)
            return {"numerator": counts["passed"], "denominator": len(items), "value": counts["passed"] / len(items),
                    "counts": {s: counts[s] for s in ("passed", "failed", "pending_review", "unknown", "not_executed")}}
        physical = [bench.read_json(base._path(self.root, f"physical-{op['attempt']:03d}.json")) for op in sealed["operations"].values()]
        def usage_for(records, reserved):
            return {key: {"known_sum": sum(item["response"]["usage"][key] for item in records if item["response"]["usage"][key] is not None),
                          "known_requests": sum(item["response"]["usage"][key] is not None for item in records),
                          "unknown_requests": reserved - sum(item["response"]["usage"][key] is not None for item in records)}
                    for key in ("input_tokens", "output_tokens")}
        arms = {}
        for arm in design.ARMS:
            items = [row for row in rows if row["arm"] == arm]
            arms[arm] = {"correct_handling": metric(items, lambda r: True),
                "supported_completion": metric([r for r in items if r["case_id"] in design.SUPPORTED], lambda r: True),
                "tool_tasks": metric([r for r in items if r["case_id"] in design.TOOLS],
                    lambda r: r["shared_checks"]["checks"]["tool_task_correct"] and r["review"]["dimensions"]["tool_permissions"], require_answer=False),
                "boundary_handling": metric([r for r in items if r["case_id"] in design.BOUNDARY], lambda r: True),
                "reading_burden": dict(Counter(row.get("review", {}).get("reading_burden", "unreviewed") for row in items))}
            arm_physical = [item for item in physical if item["case_id"].startswith(arm + "-")]
            arm_reserved = sum(number for key, number in state["turn_attempts"].items() if key.startswith(arm + "-"))
            arms[arm].update({"reserved_requests": arm_reserved, "sealed_requests": len(arm_physical),
                "usage": usage_for(arm_physical, arm_reserved),
                "physical_resources": _resources([{"latency_ms": item["latency_ms"], "usage": {**item["response"]["usage"], "physical_requests": 1}} for item in arm_physical]),
                "phase_counts": dict(Counter(item["phase"] for item in arm_physical)),
                "outcomes": dict(Counter(item["outcome"] for item in arm_physical)),
                "request_models": dict(Counter(item["request_model"] for item in arm_physical)),
                "upstream_models": dict(Counter(item["response"]["upstream_model"] or "unknown" for item in arm_physical))})
        pairs = {}
        indexed = {(row["arm"], row["case_id"]): row for row in rows}
        for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D")):
            counts = Counter()
            for cid in design.CASE_IDS:
                a, b = indexed[left, cid], indexed[right, cid]
                if a["status"] not in {"passed", "failed"} or b["status"] not in {"passed", "failed"}:
                    counts["unpaired_unknown_or_unexecuted"] += 1
                elif a["status"] == "passed" and b["status"] == "passed": counts["both_passed"] += 1
                elif a["status"] == "passed": counts["only_left_passed"] += 1
                elif b["status"] == "passed": counts["only_right_passed"] += 1
                else: counts["neither_passed"] += 1
            pairs[left + "_vs_" + right] = {**{k: counts[k] for k in ("both_passed", "only_left_passed", "only_right_passed", "neither_passed", "unpaired_unknown_or_unexecuted")},
                "planned_denominator": 12, "full_plan_delta_percentage_points": None if counts["unpaired_unknown_or_unexecuted"] else
                100 * (arms[right]["correct_handling"]["value"] - arms[left]["correct_handling"]["value"])}
        usage = usage_for(physical, state["attempts"])
        return {"schema_version": self.report_schema, "batch": self.batch, "candidate": self.candidate,
            "protocol_sha256": bench.sha256(self.root / "protocol.json"), "review_policy": deepcopy(REVIEW_POLICY),
            "planned_arm_cases": 48, "planned_arm_turns": 52, "actual_requests": state["attempts"],
            "authorized_requests": MAX_REQUESTS if (self.root / "authorization.json").exists() and self._approval(p) else 0,
            "ledger": state, "sealed_requests": len(physical), "arms": arms, "pairs": pairs, "cases": rows, "usage": usage,
            "physical_resources": _resources([{"latency_ms": item["latency_ms"], "usage": {**item["response"]["usage"], "physical_requests": 1}} for item in physical]),
            "stage_clock_policy": deepcopy(self.execution_policy()["stage_clock"]),
            "legacy_gate_scope": "diagnostic only: original success status, deterministic role/tool expectations and valid Judge >=0.75; not the old v6 full fresh/exact-projection contract, and no historical score is recomputed",
            "model_quality_evidence": bool(physical) and all(item["evidence_kind"] == "live-model" for item in physical)
                and all(row.get("data_kind", "live-mysql-and-isolated-chroma") == "live-mysql-and-isolated-chroma" for row in rows),
            "limitations": ["known author-visible questions and static synthetic records, each arm/case once; not independent blind or production/generalization rate",
                "appropriate refusal/handoff preparation is not completed mutation/human connection",
                "required-agent names/freshness/exact projection remain diagnostics, not shared outcome gates",
                "unknown, pending review and unexecuted stay in full denominator; no uplift before full terminal pairs",
                "legacy Judge scores retained, no score from old v6 inserted into new D",
                "no HTTP/browser/Redis/background memory quality acceptance"]}


def main(argv=None):
    return business._main(QualityTrial, "evaluation.quality_trial", argv)


if __name__ == "__main__":
    raise SystemExit(main())
