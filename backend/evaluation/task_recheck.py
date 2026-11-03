"""Post-routing-fix task trial, with read-only proof of earlier frozen evidence.

This new batch does not resume the zero-send v1 preparation or reuse any old
allowance. Common execution, native tools, transport and budget remain unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation import intent_recheck as history
from evaluation.intent_transport import canonical_payload_hash, messages_url
from evaluation.intent_validation import _ledger_snapshot
from evaluation import task_trial as engine
from evaluation.task_budget import TaskBudget

ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-e2e-v2-routing-fix"
BATCH = "serveflow-task-e2e-v2-routing-fix"
CANDIDATE = "shared-domain-evidence-routing-fix"
OLD_RECHECK = bench.BACKEND_ROOT / "data/benchmarks/intent-recheck-v2"
OLD_TASK = engine.ROOT
OLD_RECHECK_PROTOCOL = "31725961e8448cf07c1e64b08f846e9cbdeb80befae8ea4d75a2cfdc771291de"
OLD_RECHECK_JOURNAL = "18b8dbc09fb47b2e3e10335616fc430e224bb543d8d32e4bd7249752a05c50e3"
OLD_RECHECK_AUTHORIZATION = "2952aa08ac04238544eba018243271cde5924db82bdfd977f8bb97ba44dbccb7"
OLD_RECHECK_REPORT = "a5001db9147e06a0cff649f72d48f9a5c818a0b44896e9224eb76ff43772c799"
OLD_RECHECK_CHAIN = "6748dd214459951fa2078508947a8ffb3d13096cb400cd40ef099f39cbfdb580"
OLD_TASK_PROTOCOL = "c8e9bcde59c6244c282eb2c3cb7f6a4a17acc1b0f96a12b9632c430205ff5f39"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PROTOCOL_FIELDS = {"schema_version", "batch", "candidate", "created_at", "git_head", "root_binding_sha256", "artifacts",
    "cases_sha256", "dependencies", "model", "base_url", "messages_url", "limits", "http_limits", "policy",
    "phase_parameters", "agent_parameters", "plan", "initial_intent_requests", "offline_data_proof", "prior_proof",
    "authorized_model_requests", "proposed_model_requests", "effective_environment"}


def effective_environment():
    """Freeze effective trial settings without loading unrelated dotenv values."""
    defaults = {"ECHOMIND_INTENT_TIMEOUT_S": 8., "ECHOMIND_LLM_TIMEOUT_S": 15.,
        "ECHOMIND_COMPOSER_TIMEOUT_S": 8., "ECHOMIND_COMPOSER_MAX_TOKENS": 1000,
        "ECHOMIND_COMPOSER_TEMPERATURE": .1, "ECHOMIND_MONITOR_FALLBACK_PENALTY": .5,
        "ECHOMIND_MONITOR_COOLDOWN_SECONDS": 60., "ECHOMIND_TOOL_TRACE_MAX": 200}
    effective = {}
    for key, default in defaults.items():
        try:
            value = int(os.environ.get(key, str(default))) if type(default) is int else float(os.environ.get(key, str(default)))
        except (TypeError, ValueError, OverflowError):
            base._fail("task_recheck_effective_configuration")
        if not math.isfinite(value) or value != default:
            base._fail("task_recheck_effective_configuration")
        effective[key] = value
    for role in ("GENERAL", "TECHNICAL", "BILLING", "ESCALATION"):
        key = "ECHOMIND_" + role + "_MODEL"
        value = os.environ.get(key, "").strip()
        if value:
            base._fail("task_recheck_role_model_override")
        effective[key] = ""
    if (os.environ.get("INTENT_EMBEDDING_PROVIDER", "local").strip().lower() != "local"
            or os.environ.get("ORT_DISABLE_TELEMETRY") != "1"):
        base._fail("task_recheck_effective_configuration")
    return {**effective, "INTENT_EMBEDDING_PROVIDER": "local", "ORT_DISABLE_TELEMETRY": "1"}


def _tree(root):
    """Hash private historical files without reading them as current config."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        base._fail("task_archived_evidence_missing")
    files = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            base._fail("task_archived_evidence_unsafe")
        if item.is_file():
            name = str(item.relative_to(root))
            files[name] = bench.sha256(base._path(root, name))
    if not files:
        base._fail("task_archived_evidence_missing")
    return canonical_payload_hash(files)


def archived_proof():
    """Validate old Git bytes, original hash chains and sealed outputs only.

    No old current-code verify/report, score calculation, initialization, send,
    repair or reservation is permitted along this path.
    """
    legacy = history.archived_evidence_proof()
    before = {"recheck": _tree(OLD_RECHECK), "task_preparation": _tree(OLD_TASK)}
    for name in ("protocol.json", "authorization.json", "ledger/identity.json", "ledger/journal.jsonl"):
        if not base._path(OLD_RECHECK, name).is_file():
            base._fail("task_archived_evidence_missing")
    if (bench.sha256(OLD_RECHECK / "protocol.json") != OLD_RECHECK_PROTOCOL
            or bench.sha256(OLD_RECHECK / "authorization.json") != OLD_RECHECK_AUTHORIZATION
            or bench.sha256(OLD_RECHECK / "ledger/journal.jsonl") != OLD_RECHECK_JOURNAL
            or (OLD_RECHECK / "stopped.json").exists()):
        base._fail("task_archived_evidence_changed")
    protocol = bench.read_json(base._path(OLD_RECHECK, "protocol.json"))
    history._archived_sources(protocol, "artifacts")
    if protocol["archived_evidence_proof"] != legacy or protocol["root_binding_sha256"] != base._root_hash(OLD_RECHECK):
        base._fail("task_archived_evidence_changed")
    ledger = history._ArchivedDevelopmentLedger(OLD_RECHECK / "ledger", OLD_RECHECK_PROTOCOL,
        [op["operation_id"] for op in protocol["operations"]], 24)
    state, sealed = ledger.status(), ledger.results()
    if (state["attempts"] != 24 or state["paused"] or state["pending"] or state["chain_hash"] != OLD_RECHECK_CHAIN
            or state["completed"] != [op["operation_id"] for op in protocol["operations"]] or len(sealed) != 24):
        base._fail("task_archived_recheck_unsealed")
    approval = bench.read_json(base._path(OLD_RECHECK, "authorization.json"))
    if (approval["protocol_sha256"] != OLD_RECHECK_PROTOCOL or approval["max_requests"] != 24
            or approval["root_binding_sha256"] != base._root_hash(OLD_RECHECK)
            or approval["initial_ledger_chain_hash"] != ledger.initial_chain_hash):
        base._fail("task_archived_authorization_changed")
    observations = {}
    for op in protocol["operations"]:
        name = op["case_id"] + "-observation.json"
        digest = bench.sha256(base._path(OLD_RECHECK, name))
        if digest != sealed[op["operation_id"]]["output_sha256"]:
            base._fail("task_archived_observation_changed")
        observations[op["case_id"]] = digest
    report_name = "report-" + OLD_RECHECK_CHAIN + ".json"
    if bench.sha256(base._path(OLD_RECHECK, report_name)) != OLD_RECHECK_REPORT:
        base._fail("task_archived_report_changed")
    # The task predecessor was preparation only: it must remain exactly zero.
    if (bench.sha256(base._path(OLD_TASK, "protocol.json")) != OLD_TASK_PROTOCOL
            or (OLD_TASK / "authorization.json").exists()
            or bench.sha256(base._path(OLD_TASK, "ledger/journal.jsonl")) != EMPTY_SHA256):
        base._fail("task_archived_preparation_changed")
    previous = bench.read_json(base._path(OLD_TASK, "protocol.json"))
    history._archived_sources(previous, "artifacts")
    if (previous["git_head"] != "43720fd797e4ae39c6fe2061f799cf0f98664a4c"
            or previous["authorized_model_requests"] != 0
            or previous["root_binding_sha256"] != base._root_hash(OLD_TASK)):
        base._fail("task_archived_preparation_changed")
    old_budget = TaskBudget(OLD_TASK / "ledger", OLD_TASK_PROTOCOL, previous["plan"])
    zero = old_budget.status()
    if (zero["attempts"] or zero["pending"] or zero["paused"] or zero["active_case"]
            or zero["completed_cases"] or old_budget.results() != {"operations": {}, "cases": {}}):
        base._fail("task_archived_preparation_changed")
    if before != {"recheck": _tree(OLD_RECHECK), "task_preparation": _tree(OLD_TASK)}:
        base._fail("task_archived_changed_during_read")
    return {"schema_version": "serveflow-task-archived-proof-v2", "prior_intent_batches": legacy,
        "recheck": {"protocol_sha256": OLD_RECHECK_PROTOCOL, "authorization_sha256": OLD_RECHECK_AUTHORIZATION,
            "journal_sha256": OLD_RECHECK_JOURNAL, "ledger_sha256": _ledger_snapshot(OLD_RECHECK),
            "chain_hash": OLD_RECHECK_CHAIN, "observation_sha256": observations, "report_sha256": OLD_RECHECK_REPORT,
            "tree_sha256": before["recheck"]},
        "task_preparation": {"protocol_sha256": OLD_TASK_PROTOCOL, "tree_sha256": before["task_preparation"],
            "chain_hash": zero["chain_hash"], "authorized_model_requests": 0, "actual_model_requests": 0},
        "old_allowance_reused": False}


def _artifacts():
    extra = ("evaluation/task_recheck.py", "tests/test_task_recheck.py", "tests/test_routing_evidence.py")
    return {**engine._artifacts(), **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in extra}}


class TaskRecheck(engine.TaskTrial):
    def __init__(self, root=ROOT):
        # Deliberately independent of v1's fixed production directory. No root
        # CLI argument, no arbitrary historical directory or nested re-use.
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        historical = bench.BACKEND_ROOT / "data/benchmarks"
        if self.root != ROOT and self.root.is_relative_to(historical):
            base._fail("fixed_task_recheck_root_required")

    async def prepare(self):
        if self.root.exists():
            base._fail("task_root_already_exists")
        values, config = engine.cases(), base._configuration()
        initial = []
        for case in values:
            item = await base._capture({"id": case["id"], "input": {"message": case["turns"][0], "history": []}}, config["base_url"])
            item.pop("operation_id")
            initial.append(item)
        protocol = {"schema_version": "serveflow-task-protocol-v2", "batch": BATCH, "candidate": CANDIDATE,
            "created_at": base._now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": _artifacts(),
            "cases_sha256": bench.sha256(engine.CASES_PATH), "dependencies": engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(engine.LIMITS), "http_limits": deepcopy(engine.HTTP_LIMITS), "policy": deepcopy(engine.POLICY),
            "phase_parameters": deepcopy(engine.PHASE_PARAMETERS), "agent_parameters": deepcopy(engine.AGENT_PARAMETERS),
            "plan": engine._plan(values), "initial_intent_requests": initial, "offline_data_proof": engine.offline_data_proof(),
            "prior_proof": archived_proof(), "effective_environment": effective_environment(),
            "authorized_model_requests": 0, "proposed_model_requests": 168}
        protocol = json.loads(json.dumps(protocol))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", protocol)
        TaskBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), protocol["plan"], initialize=True)
        return protocol

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        bench._exact(p, PROTOCOL_FIELDS, "task_recheck_protocol_fields")
        config = base._configuration()
        if (p["schema_version"] != "serveflow-task-protocol-v2" or p["batch"] != BATCH or p["candidate"] != CANDIDATE
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["messages_url"] != messages_url(config["base_url"])
                or p["limits"] != engine.LIMITS or p["http_limits"] != engine.HTTP_LIMITS or p["policy"] != engine.POLICY
                or p["phase_parameters"] != json.loads(json.dumps(engine.PHASE_PARAMETERS))
                or p["agent_parameters"] != json.loads(json.dumps(engine.AGENT_PARAMETERS))
                or type(p["authorized_model_requests"]) is not int or p["authorized_model_requests"] != 0
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != 168
                or p["dependencies"] != engine._dependencies() or p["plan"] != engine._plan(engine.cases())
                or p["effective_environment"] != effective_environment()):
            base._fail("task_recheck_protocol_changed")
        if (p["artifacts"] != _artifacts() or p["cases_sha256"] != bench.sha256(engine.CASES_PATH)
                or p["offline_data_proof"] != engine.offline_data_proof() or p["prior_proof"] != archived_proof()):
            base._fail("task_recheck_frozen_evidence_changed")
        if type(p["initial_intent_requests"]) is not list or len(p["initial_intent_requests"]) != 6:
            base._fail("task_initial_requests_changed")
        for op, case in zip(p["initial_intent_requests"], engine.cases()):
            if (type(op) is not dict or set(op) != {"case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or op["case_id"] != case["id"]
                    or op["input_sha256"] != canonical_payload_hash({"message": case["turns"][0], "history": []})
                    or op["payload_sha256"] != canonical_payload_hash(op["payload"])):
                base._fail("task_initial_requests_changed")
        self.ledger(p).status()
        self._sealed(p)
        return p

    def report(self):
        return {**super().report(), "batch": BATCH, "candidate": CANDIDATE,
                "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "status", "report", "preflight", "authorize", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-new-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(argv)
    trial = TaskRecheck()
    try:
        if args.operation == "prepare":
            value = asyncio.run(trial.prepare())
        elif args.operation == "verify":
            trial.verify()
            value = {"verified": True, "model_requests": trial.ledger(trial.verify()).status()["attempts"]}
        elif args.operation == "status":
            value = trial.ledger(trial.verify()).status()
        elif args.operation == "report":
            value = trial.report()
        elif args.operation == "preflight":
            value = asyncio.run(trial.preflight())
        elif args.operation == "authorize":
            value = trial.authorize(args.reference, confirmed=args.confirm_new_authorization)
        elif args.operation == "review":
            if not args.review_file:
                base._fail("task_review_file_required")
            value = trial.review(bench.read_json(args.review_file))
        elif args.operation == "_worker":
            value = asyncio.run(trial.execute())
        else:
            trial._approval(trial.verify())
            from evaluation.task_supervisor import supervise
            os.chdir(bench.BACKEND_ROOT)
            value = supervise([sys.executable, "-B", "-u", "-m", "evaluation.task_recheck", "_worker"], root=ROOT)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        print("STOP: task_recheck_failed; inspect private receipts, do not retry or reinitialize", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
