"""Fixed business-quality regression, independent of every earlier trial.

Preparation is offline and grants zero sends. The twelve authored cases retain
their original questions and expectations. Business acceptance and the legacy
Judge-gated score are deliberately separate; this never rewrites old scores.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_trial as engine, task_repair as repair
from evaluation import task_budget as accounting
from evaluation.benchmark_metrics import REVIEW_METHODS, _resources
from evaluation.intent_transport import canonical_payload_hash, messages_url
from evaluation.task_recheck import effective_environment


ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-business-v4"
BATCH = "serveflow-task-business-v4"
CANDIDATE = "native-order-read-deterministic-business-evidence"
CASE_IDS = tuple([f"e{i:02d}" for i in range(1, 7)] + [f"t{i:02d}" for i in range(7, 13)])
MAX_REQUESTS = 312
LIMITS = {**deepcopy(engine.LIMITS), "cases": 12, "turns": 13, "physical_requests": MAX_REQUESTS}
REVIEW_POLICY = {
    "version": "task-business-v4",
    "dimensions": list(engine.REVIEW_FIELDS),
    "primary_gate": "execution + deterministic evidence/projection + seven-dimension executor review; no blockers",
    "judge_role": "auxiliary; raw original scores retained; a separate legacy Judge-gated result is reported",
    "legacy_gate": "original execution/deterministic gates + seven-dimension review + every valid Judge score >= 0.75",
    "projection_scope": "exact program-rendered order evidence only; additional prose still requires semantic review",
    "partial_write_refusal": "complete actual turns may be partial only for a requested business mutation; all read evidence/permissions/projections must remain valid and seven-dimension review must confirm an explicit correct refusal with no missed query",
    "partial_exclusions": "failed tools, absent evidence, timeouts, missing turns and non-mutation partial answers never qualify",
    "freshness": "a case requiring an order read must read its uniquely resolved order in each relevant actual turn",
    "record_time": "source as_of/record_updated_at; query wall time is not business record time",
    "denominator": "all twelve planned cases; unknown and unexecuted cases remain in the denominator",
    "groups": "six known regression cases and six existing author-visible validation cases, reported separately",
    "comparison": "different grading version; never describe historical 1/6 to a new business score as a measured uplift",
    "semantics": "unchanged source text or valid structure does not prove relevance, coverage or absence of extra invented facts",
}
POLICY = {**deepcopy(engine.POLICY), "grading_version": "task-business-v4"}
PROTOCOL_FIELDS = repair.PROTOCOL_FIELDS
OLD_ROOT = repair.ROOT
OLD_PROTOCOL = "e08044cf73fdbafa47dcd6351ad8fc176af50ffbb5e5a12498e0b36bc60d2c3f"
OLD_AUTHORIZATION = "c95d3c0e580d83eb89a57b95f9978f822310524f674246d24d4bfa87a1e986c7"
OLD_JOURNAL = "931118a7265840593450a7789294f13760c1dfdda10d0de7c004a389c04d5b61"
OLD_CHAIN = "98101c74c470a41292e257adb30a5fe3969ea2d895fd4b3193dac86d87cc17d0"
OLD_TREE = "8e93698ec3d43f64e9fc728f466ac6d5b680c7570e66696ade9534966129eaef"


def cases():
    """Reuse, never relabel, the original six and the existing validation six."""
    original = engine.cases()
    suite = bench.load_suites()["tasks"]
    bench.validate_suite(suite)
    additions = [case for case in suite["cases"] if case["split"] == "validation"]
    if [c["id"] for c in additions] != list(CASE_IDS[6:]):
        base._fail("task_business_fixed_validation_changed")
    values = deepcopy(original)
    for case in additions:
        if case["input"].get("history"):
            base._fail("task_business_validation_history_changed")
        values.append({"id": case["id"], "source_case_id": case["id"], "variant": False,
            "customer_id": case["input"]["customer_id"], "turns": [case["input"]["message"]],
            "expected": deepcopy(case["expected"]), "references": deepcopy(case["references"]),
            "tags": deepcopy(case["tags"])})
    if tuple(c["id"] for c in values) != CASE_IDS or sum(len(c["turns"]) for c in values) != 13:
        base._fail("task_business_fixed_plan_changed")
    return values


def fixed_plan():
    return [{"case_id": key, "turn_ids": ["turn-1", "turn-2"] if key == "e04" else ["turn-1"]}
            for key in CASE_IDS]


class BusinessTaskBudget(accounting.TaskBudget):
    """New fixed 12-case/13-turn identity; reuse old durable journal mechanics.

    No old limits, ledger bytes or genesis are changed. Only this exact ordered
    plan admits up to 24 sends per turn / 312 overall / three in flight. A
    premarked or unfinished operation is never refunded or auto-resumed.
    """

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False):
        if type(initialize) is not bool or case_plan != fixed_plan() or type(case_plan) is not list:
            raise accounting.TaskBudgetError("configuration")
        try:
            self.root = Path(root).absolute()
            if self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents):
                raise accounting.TaskBudgetError("configuration")
        except (TypeError, ValueError, OSError):
            raise accounting.TaskBudgetError("configuration") from None
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise accounting.TaskBudgetError("protocol")
        plan = deepcopy(case_plan)
        self.case_plan = tuple(plan)
        self._case_ids = tuple(p["case_id"] for p in plan)
        self._turns = {p["case_id"]: tuple(p["turn_ids"]) for p in plan}
        self.protocol_sha256 = accounting._hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        self.operation_ids = tuple(f'{p["case_id"]}:{t}' for p in plan for t in p["turn_ids"])
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-task-business:identity:v4\n" + accounting._canonical(self._identity)).hexdigest()
        self._lease = self._lease_fd = None
        self._initializing = initialize
        try:
            with self._locked(create=initialize) as (directory, _identity, journal, claims):
                self._replay(journal, claims)
                if initialize:
                    fd = os.open("run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
                    try:
                        accounting._private_file(fd)
                        accounting._write(fd, self._run_identity())
                    finally:
                        os.close(fd)
                    os.fsync(directory)
        finally:
            self._initializing = False

    @staticmethod
    def _event_hash(unsigned):
        return hashlib.sha256(b"serveflow-task-business:event:v4\n" + accounting._canonical(unsigned)).hexdigest()

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        self._metadata(case_id, turn_id, phase, agent)
        if state["paused"]:
            raise accounting.TaskBudgetError("paused")
        if state["active_case"] != case_id:
            raise accounting.TaskBudgetError("protocol")
        if len(state["pending"]) >= accounting.MAX_IN_FLIGHT:
            raise accounting.TaskBudgetError("ledger_busy")
        turn = (case_id, turn_id)
        if (state["attempts"] >= MAX_REQUESTS or state["turn_attempts"].get(turn, 0) >= accounting.PER_TURN
                or state["phase_counts"].get((*turn, phase), 0) >= accounting.PHASE_CAPS[phase]):
            raise accounting.TaskBudgetError("quota")
        active, planned = state["active_turn"].get(case_id), self._turns[case_id]
        if (active is None and turn_id != planned[0] or active is not None and active != turn_id
                and (planned.index(turn_id) != planned.index(active) + 1 or state["pending"])):
            raise accounting.TaskBudgetError("protocol")

    def _apply(self, state, event):
        if event["event"] != "case_start":
            return super()._apply(state, event)
        key = event["case_id"]
        if (type(key) is not str or key not in self._turns or type(event["attempt"]) is not int
                or state["active_case"] is not None or state["paused"] or state["pending"]
                or len(state["completed_cases"]) >= 12
                or key != self._case_ids[len(state["completed_cases"])] or key in state["starts"]
                or event["operation_id"] != "case:" + key or event["attempt"] != state["attempts"]
                or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256",
                    "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
            raise accounting.TaskBudgetError("protocol")
        state["active_case"] = key
        state["starts"].append(key)


def prior_proof():
    """Read exact old bytes/claims; never call old report, repair or execute."""
    from evaluation.task_recheck import _tree, history
    older = repair.prior_proof()
    before = _tree(OLD_ROOT)
    if before != OLD_TREE:
        base._fail("task_business_old_tree_changed")
    for name, digest in {"protocol.json": OLD_PROTOCOL, "authorization.json": OLD_AUTHORIZATION,
                         "ledger/journal.jsonl": OLD_JOURNAL}.items():
        if bench.sha256(base._path(OLD_ROOT, name)) != digest:
            base._fail("task_business_old_evidence_changed")
    protocol = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
    history._archived_sources(protocol, "artifacts")
    if protocol.get("prior_proof") != older:
        base._fail("task_business_old_history_changed")
    budget = accounting.TaskBudget(OLD_ROOT / "ledger", OLD_PROTOCOL, protocol["plan"])
    state, sealed = budget.status(), budget.results()
    if (state["attempts"] != 31 or state["pending"] or state["paused"] is not True
            or state["active_case"] != "e05" or state["completed_cases"] != ["e01", "e02", "e03", "e04"]
            or state["chain_hash"] != OLD_CHAIN or len(sealed["operations"]) != 31):
        base._fail("task_business_old_ledger_changed")
    for operation in sealed["operations"].values():
        if bench.sha256(base._path(OLD_ROOT, f"physical-{operation['attempt']:03d}.json")) != operation["output_sha256"]:
            base._fail("task_business_old_physical_changed")
    for key, item in sealed["cases"].items():
        if bench.sha256(base._path(OLD_ROOT, f"case-{key}.json")) != item["artifact_sha256"]:
            base._fail("task_business_old_case_changed")
    if _tree(OLD_ROOT) != before:
        base._fail("task_business_old_tree_changed_during_read")
    return {"schema_version": "serveflow-task-v3-readonly-proof-v1", "tree_sha256": before,
        "protocol_sha256": OLD_PROTOCOL, "authorization_sha256": OLD_AUTHORIZATION,
        "journal_sha256": OLD_JOURNAL, "chain_hash": OLD_CHAIN,
        "reserved_requests": 31, "sealed_requests": 31, "old_strict_results": {
            "passed": 1, "failed": 3, "unknown": 1, "not_executed": 1},
        "older_proof_sha256": canonical_payload_hash(older), "old_allowance_reused": False,
        "old_scores_recomputed": False}


def _artifacts():
    result = repair._artifacts()
    additions = ["evaluation/task_business_regression.py", "tests/test_task_business_regression.py",
                 "tests/test_order_runtime.py", "tests/test_diagnostic_evidence.py",
                 "tests/test_skill_business_grounding.py"]
    # The evidence/diagnostic modules are new production boundaries, not part of
    # the old benchmark's fixed list. Freeze every current Python module here.
    for folder in ("core", "agents"):
        additions += [str(p.relative_to(bench.BACKEND_ROOT))
                      for p in sorted((bench.BACKEND_ROOT / folder).glob("*.py"))]
    additions += [str(p.relative_to(bench.BACKEND_ROOT)) for p in sorted(
        (bench.BACKEND_ROOT / "tests").glob("test_*evidence*.py"))]
    return {**result, **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in additions}}


def projection_checks(case, artifact):
    """Check actual native evidence blocks, not semantic correctness of prose."""
    from core.order_evidence import extract_order_evidence, merge_order_evidence, render_order_evidence, resolve_order_id
    checks = {"fresh_native_order_read": True, "order_projection_valid": True, "order_projection_present": True}
    records, history = [], []
    for index, turn in enumerate(artifact.get("turns", []), 1):
        calls = [call for call in turn["tool_evidence"]["calls"] if call["name"] == "read_demo_order"]
        selection = resolve_order_id(turn["question"], history=history)
        if "read_demo_order" in case["expected"]["required_tools"] and selection.order_id is not None:
            checks["fresh_native_order_read"] &= any(call.get("success") and call.get("tool_use_id")
                and call.get("input") == {"order_id": selection.order_id} for call in calls)
        evidence = []
        for call in calls:
            identity = (call.get("input") or {}).get("order_id")
            value = extract_order_evidence(call.get("result"), expected_order_id=identity,
                                          trusted_customer_id=case["customer_id"])
            checks["order_projection_valid"] &= value.status not in {"invalid", "unavailable", "partial"}
            evidence.append(value)
        if evidence:
            merged = merge_order_evidence(evidence)
            text = render_order_evidence(merged)
            valid = merged.status not in {"invalid", "unavailable", "partial"}
            present = bool(text and text in turn["result"]["response"])
            checks["order_projection_valid"] &= valid
            checks["order_projection_present"] &= present
            records.append({"turn_id": f"turn-{index}", "status": merged.status,
                "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(), "present_in_answer": present})
        history += [{"role": "user", "content": turn["question"]},
                    {"role": "assistant", "content": turn["result"]["response"]}]
    return {"checks": checks, "blocks": records,
        "scope": "program-rendered source facts only; extra prose and relevance require executor review"}


def business_execution_checks(case, artifact):
    """Separate a completed read plus refused write from execution failure.

    A mutation request is eligibility, never proof of a correct refusal. Source
    checks, valid projection and the seven-dimension review remain mandatory.
    The legacy execution_complete/success gate is not changed or rescored.
    """
    from core.order_evidence import requests_business_mutation
    turns = artifact.get("turns", [])
    checks = {"actual_turns_complete": type(turns) is list and len(turns) == len(case["turns"]),
              "actual_turns_bound": True, "no_failed_turn": True, "partial_only_requested_mutation": True}
    partials = []
    for index, turn in enumerate(turns, 1):
        question = case["turns"][index - 1] if index <= len(case["turns"]) else None
        checks["actual_turns_bound"] &= turn.get("turn_id") == f"turn-{index}" and turn.get("question") == question
        status = turn.get("result", {}).get("status")
        checks["no_failed_turn"] &= status in {"success", "partial"}
        if status == "partial":
            eligible = question is not None and requests_business_mutation(question)
            checks["partial_only_requested_mutation"] &= eligible
            partials.append({"turn_id": f"turn-{index}", "requested_mutation": eligible,
                            "semantic_refusal_review_required": True})
    return {"checks": checks, "partial_turns": partials,
        "scope": "execution eligibility only; every read/evidence/permission gate and semantic review is still required"}


class BusinessRegression(repair.TaskRepair):
    # Fixed subclasses may select a new, separately authorized observation
    # batch. No CLI/env switch can change the transport or evaluation policy.
    root_path = ROOT
    batch = BATCH
    candidate = CANDIDATE
    protocol_schema = "serveflow-task-business-protocol-v4"
    authorization_schema = "serveflow-task-business-authorization-v4"
    report_schema = "serveflow-task-business-report-v4"

    def __init__(self, root=None):
        self.root = Path(self.root_path if root is None else root).absolute()
        base._path(self.root, "protocol.json")
        if self.root != self.root_path and self.root.is_relative_to(bench.BACKEND_ROOT / "data/benchmarks"):
            base._fail("fixed_task_business_root_required")

    def artifact_proof(self):
        return _artifacts()

    def historical_proof(self):
        return prior_proof()

    def execution_policy(self):
        return POLICY

    def make_transport(self, *args, **kwargs):
        return engine.TaskHTTPTransport(*args, **kwargs)

    def ledger(self, protocol):
        return BusinessTaskBudget(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")), protocol["plan"])

    async def prepare(self):
        if self.root.exists():
            base._fail("task_root_already_exists")
        values, config = cases(), base._configuration()
        initial = []
        for case in values:
            item = await base._capture({"id": case["id"], "input": {"message": case["turns"][0], "history": []}}, config["base_url"])
            item.pop("operation_id")
            initial.append(item)
        protocol = {"schema_version": self.protocol_schema, "batch": self.batch, "candidate": self.candidate,
            "created_at": base._now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                cwd=bench.BACKEND_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": self.artifact_proof(),
            "cases_sha256": canonical_payload_hash({"cases": values}), "dependencies": engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(LIMITS), "http_limits": deepcopy(engine.HTTP_LIMITS), "policy": deepcopy(self.execution_policy()),
            "review_policy": deepcopy(REVIEW_POLICY), "phase_parameters": deepcopy(engine.PHASE_PARAMETERS),
            "agent_parameters": deepcopy(engine.AGENT_PARAMETERS), "plan": fixed_plan(), "initial_intent_requests": initial,
            "offline_data_proof": engine.offline_data_proof(), "prior_proof": self.historical_proof(),
            "effective_environment": effective_environment(), "data_proof_sha256": repair.data_binding(),
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS}
        protocol = json.loads(json.dumps(protocol))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", protocol)
        BusinessTaskBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), protocol["plan"], initialize=True)
        return protocol

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        bench._exact(p, PROTOCOL_FIELDS, "task_business_protocol_fields")
        config, values = base._configuration(), cases()
        if (p["schema_version"] != self.protocol_schema or p["batch"] != self.batch or p["candidate"] != self.candidate
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["messages_url"] != messages_url(config["base_url"])
                or p["limits"] != LIMITS or p["http_limits"] != engine.HTTP_LIMITS or p["policy"] != self.execution_policy()
                or p["review_policy"] != REVIEW_POLICY or p["dependencies"] != engine._dependencies()
                or p["phase_parameters"] != json.loads(json.dumps(engine.PHASE_PARAMETERS))
                or p["agent_parameters"] != json.loads(json.dumps(engine.AGENT_PARAMETERS))
                or p["plan"] != fixed_plan() or type(p["authorized_model_requests"]) is not int
                or p["authorized_model_requests"] != 0 or type(p["proposed_model_requests"]) is not int
                or p["proposed_model_requests"] != MAX_REQUESTS or p["effective_environment"] != effective_environment()
                or p["data_proof_sha256"] != repair.data_binding()):
            base._fail("task_business_protocol_changed")
        if (p["artifacts"] != self.artifact_proof() or p["cases_sha256"] != canonical_payload_hash({"cases": values})
                or p["offline_data_proof"] != engine.offline_data_proof() or p["prior_proof"] != self.historical_proof()):
            base._fail("task_business_frozen_evidence_changed")
        if type(p["initial_intent_requests"]) is not list or len(p["initial_intent_requests"]) != 12:
            base._fail("task_business_initial_requests_changed")
        for item, case in zip(p["initial_intent_requests"], values):
            if (type(item) is not dict or set(item) != {"case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or item["case_id"] != case["id"]
                    or item["input_sha256"] != canonical_payload_hash({"message": case["turns"][0], "history": []})
                    or item["payload_sha256"] != canonical_payload_hash(item["payload"])
                    or type(item["request_bytes_sha256"]) is not str
                    or re.fullmatch(r"[a-f0-9]{64}", item["request_bytes_sha256"]) is None):
                base._fail("task_business_initial_requests_changed")
        self.ledger(p).status()
        self._sealed(p)
        actual = base._path(self.root, "data-proof.json")
        if actual.exists() and canonical_payload_hash(bench.read_json(actual)) != p["data_proof_sha256"]:
            base._fail("task_business_data_identity_changed")
        return p

    def authorize(self, reference, *, confirmed=False):
        p = self.verify()
        if confirmed is not True or type(reference) is not str or not reference.strip() or len(reference) > 500:
            base._fail("task_explicit_authorization_required")
        ledger = self.ledger(p)
        if ledger.status()["attempts"] or ledger.status()["active_case"]:
            base._fail("task_already_started")
        value = {"schema_version": self.authorization_schema, "reference": reference,
            "recorded_at": base._now(), "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "root_binding_sha256": base._root_hash(self.root), "initial_chain_hash": ledger.initial_chain_hash,
            "max_requests": MAX_REQUESTS, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256",
                            "initial_chain_hash", "max_requests", "plan"}, "task_authorization_fields")
        if (value["schema_version"] != self.authorization_schema
                or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash
                or type(value["max_requests"]) is not int or value["max_requests"] != MAX_REQUESTS
                or value["plan"] != p["plan"]):
            base._fail("task_authorization_changed")
        return value

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        from anthropic import AsyncAnthropic
        import httpx
        p = self.verify()
        self._approval(p)  # Before credentials, stores or SDK construction.
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("task_incomplete_controlled_boundary")
        if any((self.root / name).exists() for name in ("stopped.json", "supervisor-stop.json", "worker-stop.json")):
            base._fail("task_paused")
        config = base._configuration(secret=delegate_factory is None)
        ledger, data, sdk = self.ledger(p), controlled_data, None
        with ledger.exclusive_run() as lease:
            try:
                if data is None:
                    url = engine._mysql_url()
                    if not url:
                        base._fail("dedicated_eval_mysql_not_configured")
                    data = await engine.build_live_data(url)
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("task_controlled_data_required")
                data.assert_healthy()
                if canonical_payload_hash(data.proof) != p["data_proof_sha256"]:
                    base._fail("task_business_data_identity_changed")
                name = "data-proof.json"
                if not (self.root / name).exists():
                    base._write(self.root, name, data.proof)
                elif bench.read_json(base._path(self.root, name)) != data.proof:
                    base._fail("task_data_proof_changed")
                proof_hash = bench.sha256(self.root / name)
                def admission():
                    self.verify()
                    self._approval(p)
                    if bench.sha256(base._path(self.root, name)) != proof_hash:
                        base._fail("task_data_proof_changed")
                    data.assert_healthy()
                transport = self.make_transport(config["base_url"], self.root, ledger, lease, admission,
                                                delegate_factory=delegate_factory)
                http = httpx.AsyncClient(transport=transport, timeout=8, follow_redirects=False,
                    trust_env=False, headers={"Accept-Encoding": "identity"})
                sdk = AsyncAnthropic(api_key=config.get("api_key", "controlled-only"), base_url=config["base_url"],
                    max_retries=0, timeout=8, http_client=http)
                def healthy():
                    if transport.fault:
                        base._fail("task_transport_stopped")
                    admission()
                for case in cases():
                    if case["id"] in ledger.status()["completed_cases"]:
                        continue
                    healthy()
                    ledger.start_case(case["id"], lease=lease)
                    with base._quiet_clients():
                        result = await engine.execute_case(case, sdk, data, ensure_healthy=healthy)
                    result.update({"protocol_sha256": bench.sha256(self.root / "protocol.json"),
                        "evidence_kind": transport.kind, "data_proof_sha256": proof_hash,
                        "order_projection": projection_checks(case, result),
                        "business_execution": business_execution_checks(case, result), "grading_version": "task-business-v4"})
                    filename = f"case-{case['id']}.json"
                    base._write(self.root, filename, result)
                    ledger.finish_case(case["id"], bench.sha256(self.root / filename),
                        outcome="success" if result["execution_status"] == "success" else "failed", lease=lease)
            except BaseException as exc:
                if not (self.root / "worker-stop.json").exists():
                    from evaluation.task_evidence import EvidenceError, FATAL_EVIDENCE_CODES
                    stopped = {"schema_version": "serveflow-task-worker-stop-v1",
                        "reason": "cancelled" if isinstance(exc, asyncio.CancelledError) else "execution_stopped",
                        "recorded_at": base._now()}
                    if isinstance(exc, EvidenceError) and exc.reason in FATAL_EVIDENCE_CODES:
                        stopped["evidence_error"] = exc.reason
                    base._write(self.root, "worker-stop.json", stopped)
                raise
            finally:
                try:
                    if sdk is not None:
                        await sdk.close()
                finally:
                    if data is not None:
                        await data.close()
        return self.report()

    def _review(self, value):
        bench._exact(value, {"case_id", "artifact_sha256", "reviewer", "method", "dimensions", "reason",
                            "blocking_errors", "reading_burden"}, "task_review_fields")
        if value["case_id"] not in CASE_IDS:
            base._fail("task_review_case")
        for key in ("reviewer", "method", "reason"):
            bench._text(value[key], 4000)
        bench._strings(value["blocking_errors"])
        if value["method"] not in REVIEW_METHODS or value["reading_burden"] not in {"none", "some", "heavy"}:
            base._fail("task_review_method_or_burden")
        bench._exact(value["dimensions"], set(engine.REVIEW_FIELDS), "task_review_dimensions")
        if not all(type(v) is bool for v in value["dimensions"].values()):
            base._fail("task_review_dimensions")
        if value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{value['case_id']}.json")):
            base._fail("task_review_binding")

    def report(self):
        p = self.verify()
        ledger = self.ledger(p)
        state, sealed = ledger.status(), self._sealed(p)
        rows, artifacts = [], []
        for case in cases():
            key = case["id"]
            initial = "unknown" if state["active_case"] == key else "not_executed"
            row = {"case_id": key, "group": "regression" if key.startswith("e") else "validation",
                "planned_turns": len(case["turns"]), "business_status": initial, "legacy_judge_gated_status": initial}
            if key in sealed["cases"]:
                artifact = bench.read_json(base._path(self.root, f"case-{key}.json"))
                projections = projection_checks(case, artifact)
                execution = business_execution_checks(case, artifact)
                if (artifact.get("order_projection") != projections or artifact.get("business_execution") != execution
                        or artifact.get("grading_version") != "task-business-v4"):
                    base._fail("task_business_projection_changed")
                legacy_hard = (artifact["execution_status"] == "success" and not artifact["deterministic"]["blockers"]
                               and all(artifact["deterministic"]["checks"].values()))
                required = {name: value for name, value in artifact["deterministic"]["checks"].items()
                            if name != "execution_complete"}
                hard = (all(execution["checks"].values()) and not any(
                    name != "execution_complete" for name in artifact["deterministic"]["blockers"])
                    and all(required.values()) and all(projections["checks"].values()))
                judges = [turn["judge_overall"] for turn in artifact["turns"]]
                row.update({"deterministic_pass": hard, "legacy_deterministic_pass": legacy_hard,
                    "business_execution": execution, "order_projection": projections,
                    "judge_scores": judges, "judge_ready": artifact["judge_ready"],
                    "data_kind": artifact["data_kind"], "evidence_kind": artifact["evidence_kind"]})
                row["business_status"] = "pending_review" if hard else "failed"
                row["legacy_judge_gated_status"] = "failed" if not legacy_hard else "pending_review" if artifact["judge_ready"] else "evaluation_failed"
                path = base._path(self.root, f"review-{key}.json")
                if path.exists():
                    review = bench.read_json(path)
                    self._review(review)
                    row["semantic_review"] = {**review, "review_sha256": bench.sha256(path)}
                    business = hard and all(review["dimensions"].values()) and not review["blocking_errors"]
                    legacy = (legacy_hard and all(review["dimensions"].values()) and not review["blocking_errors"]
                        and artifact["judge_ready"] and all(type(s) in (int, float) and s >= .75 for s in judges))
                    row["business_status"] = "passed" if business else "failed"
                    row["legacy_judge_gated_status"] = "passed" if legacy else "failed"
                artifacts.append(artifact)
            rows.append(row)
        statuses = ("passed", "failed", "evaluation_failed", "pending_review", "unknown", "not_executed")
        def counts(items, field):
            return {key: sum(row[field] == key for row in items) for key in statuses}
        physical = [bench.read_json(base._path(self.root, f"physical-{op['attempt']:03d}.json"))
                    for op in sealed["operations"].values() if op["output_sha256"] is not None]
        resources = _resources([{"latency_ms": item["latency_ms"], "usage": {
            **item["response"]["usage"], "physical_requests": 1}} for item in physical])
        usage = {}
        for key in ("input_tokens", "output_tokens"):
            known = [item["response"]["usage"][key] for item in physical if item["response"]["usage"][key] is not None]
            usage[key] = {"known_sum": sum(known) if known else None, "known_requests": len(known),
                          "unknown_requests": state["attempts"] - len(known)}
        return {"schema_version": self.report_schema, "batch": self.batch, "candidate": self.candidate,
            "protocol_sha256": bench.sha256(self.root / "protocol.json"), "review_policy": deepcopy(REVIEW_POLICY),
            "planned_cases": 12, "planned_turns": 13, "planned_denominator": 12,
            "authorized_requests": MAX_REQUESTS if (self.root / "authorization.json").exists() and self._approval(p) else 0,
            "ledger": state, "cases": rows,
            "business_counts": counts(rows, "business_status"),
            "legacy_judge_gated_counts": counts(rows, "legacy_judge_gated_status"),
            "groups": {group: {"planned_denominator": 6,
                "business_counts": counts([row for row in rows if row["group"] == group], "business_status"),
                "legacy_judge_gated_counts": counts([row for row in rows if row["group"] == group], "legacy_judge_gated_status")}
                for group in ("regression", "validation")},
            "physical_resources": resources, "usage": usage,
            "reserved_requests": state["attempts"], "sealed_requests": len(physical),
            "review_dimensions": {name: {"true_count": sum(row.get("semantic_review", {}).get("dimensions", {}).get(name) is True for row in rows),
                "reviewed_count": sum("semantic_review" in row for row in rows), "planned_denominator": 12}
                for name in engine.REVIEW_FIELDS},
            "reading_burden": {level: sum(row.get("semantic_review", {}).get("reading_burden") == level for row in rows)
                for level in ("none", "some", "heavy")},
            "limitations": ["known authored cases; shared synthetic records; not independent blind testing or generalization proof",
                "changed primary grading policy; do not compare historical 1/6 to the new business score as uplift",
                "Judge is auxiliary; the separate legacy gate retains its original 0.75 threshold",
                "projection identity is not a semantic correctness proof for extra text",
                "physical latency includes admission/cleanup; network deadline remains eight seconds",
                "unknown and unexecuted remain in denominator; no HTTP/browser/Redis/background-memory acceptance"]}


def _main(trial_class, worker_module, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "status", "report", "preflight", "authorize", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-new-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(argv)
    trial = trial_class()
    try:
        if args.operation == "prepare": value = asyncio.run(trial.prepare())
        elif args.operation == "verify": value = {"verified": bool(trial.verify()), "model_requests": trial.ledger(trial.verify()).status()["attempts"]}
        elif args.operation == "status": value = trial.ledger(trial.verify()).status()
        elif args.operation == "report": value = trial.report()
        elif args.operation == "preflight": value = asyncio.run(trial.preflight())
        elif args.operation == "authorize": value = trial.authorize(args.reference, confirmed=args.confirm_new_authorization)
        elif args.operation == "review":
            if not args.review_file: base._fail("task_review_file_required")
            value = trial.review(bench.read_json(args.review_file))
        elif args.operation == "_worker": value = asyncio.run(trial.execute())
        else:
            trial._approval(trial.verify())
            from evaluation.task_supervisor import supervise
            os.chdir(bench.BACKEND_ROOT)
            value = supervise([sys.executable, "-B", "-u", "-m", worker_module, "_worker"],
                              root=trial.root, deadline_s=trial.execution_policy()["outer_process_deadline_s"])
            print(json.dumps(value, ensure_ascii=False, allow_nan=False))
            return 0 if value["exit_code"] == 0 and not value["timed_out"] and not value["cancelled"] else 2
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        print("STOP: task_business_regression_failed; inspect private receipts, do not retry or reinitialize", file=sys.stderr)
        return 2


def main(argv=None):
    return _main(BusinessRegression, "evaluation.task_business_regression", argv)


if __name__ == "__main__":
    raise SystemExit(main())
