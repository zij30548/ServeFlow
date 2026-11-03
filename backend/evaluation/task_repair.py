"""Fixed post-repair task trial; preparation grants zero model requests.

This is a new candidate, not a continuation or refund of the paused v2 trial.
Execution, physical-send accounting and report gates use the existing runner.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation import task_trial as engine
from evaluation.intent_transport import canonical_payload_hash, messages_url
from evaluation.task_budget import TaskBudget
from evaluation.task_recheck import effective_environment


ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-e2e-v3-evidence-fix"
BATCH = "serveflow-task-e2e-v3-evidence-fix"
CANDIDATE = "mysql-collections-capture-stop-policy-sources"
DATA_PREFLIGHT = bench.BACKEND_ROOT / "data/benchmarks/task-e2e-v3-data/preflight-v1.json"
REVIEW_POLICY = {
    "dimensions": list(engine.REVIEW_FIELDS),
    "judge_role": "auxiliary only; never overrides deterministic or semantic blockers",
    "source_projection": "unchanged source text still requires direct relevance and coverage of every legitimate subquestion",
    "record_time": "returned as_of or related record_updated_at; query wall time alone is not record time",
    "uncertainty": "unknown amounts, transaction gaps and source applicability stay explicit",
    "reading_burden": "record unrelated context separately; full-source display is not automatically a complete answer",
    "sample_scope": "known authored regression cases; not blind testing or generalization proof",
    "old_results": "previous scores, unknown sends and unused allowance remain separate and unchanged",
}
PROTOCOL_FIELDS = {
    "schema_version", "batch", "candidate", "created_at", "git_head", "root_binding_sha256",
    "artifacts", "cases_sha256", "dependencies", "model", "base_url", "messages_url",
    "limits", "http_limits", "policy", "review_policy", "phase_parameters", "agent_parameters",
    "plan", "initial_intent_requests", "offline_data_proof", "prior_proof",
    "authorized_model_requests", "proposed_model_requests", "effective_environment", "data_proof_sha256",
}


def prior_proof():
    # Importing/preparing this module performs no historical writes or sends.
    from evaluation.task_repair_archive import paused_predecessor_proof
    return paused_predecessor_proof()


def data_binding():
    """Bind the independently prepared data identity, never credentials."""
    value = bench.read_json(base._path(DATA_PREFLIGHT.parent, DATA_PREFLIGHT.name))
    if (type(value) is not dict or set(value) != {"model_requests", "data_proof", "order_checks"}
            or type(value["model_requests"]) is not int or value["model_requests"] != 0
            or type(value["order_checks"]) is not dict
            or set(value["order_checks"]) != {"own_order_amount", "cross_customer_denied"}
            or any(item is not True for item in value["order_checks"].values())
            or type(value["data_proof"]) is not dict
            or value["data_proof"].get("schema_version") != "serveflow-task-live-data-v1"
            or type(value["data_proof"].get("mysql")) is not dict
            or value["data_proof"]["mysql"].get("literal_schema_grants") is not True):
        base._fail("task_repair_data_preflight_required")
    return canonical_payload_hash(value["data_proof"])


def _artifacts():
    extra = (
        "evaluation/task_repair.py", "tests/test_task_repair.py",
        "evaluation/task_repair_archive.py", "tests/test_task_repair_archive.py",
        "evaluation/task_recheck.py", "tests/test_task_recheck.py", "tests/test_routing_evidence.py",
    )
    return {**engine._artifacts(), **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in extra}}


class TaskRepair(engine.TaskTrial):
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        historical = bench.BACKEND_ROOT / "data/benchmarks"
        if self.root != ROOT and self.root.is_relative_to(historical):
            base._fail("fixed_task_repair_root_required")

    async def prepare(self):
        if self.root.exists():
            base._fail("task_root_already_exists")
        values, config = engine.cases(), base._configuration()
        initial = []
        for case in values:
            item = await base._capture({"id": case["id"], "input": {"message": case["turns"][0], "history": []}}, config["base_url"])
            item.pop("operation_id")
            initial.append(item)
        protocol = {
            "schema_version": "serveflow-task-protocol-v3", "batch": BATCH, "candidate": CANDIDATE,
            "created_at": base._now(), "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": _artifacts(),
            "cases_sha256": bench.sha256(engine.CASES_PATH), "dependencies": engine._dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(engine.LIMITS), "http_limits": deepcopy(engine.HTTP_LIMITS),
            "policy": deepcopy(engine.POLICY), "review_policy": deepcopy(REVIEW_POLICY),
            "phase_parameters": deepcopy(engine.PHASE_PARAMETERS), "agent_parameters": deepcopy(engine.AGENT_PARAMETERS),
            "plan": engine._plan(values), "initial_intent_requests": initial,
            "offline_data_proof": engine.offline_data_proof(), "prior_proof": prior_proof(),
            "effective_environment": effective_environment(), "data_proof_sha256": data_binding(),
            "authorized_model_requests": 0, "proposed_model_requests": 168,
        }
        protocol = json.loads(json.dumps(protocol))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", protocol)
        TaskBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), protocol["plan"], initialize=True)
        return protocol

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        bench._exact(p, PROTOCOL_FIELDS, "task_repair_protocol_fields")
        config = base._configuration()
        if (p["schema_version"] != "serveflow-task-protocol-v3" or p["batch"] != BATCH or p["candidate"] != CANDIDATE
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["messages_url"] != messages_url(config["base_url"])
                or p["limits"] != engine.LIMITS or p["http_limits"] != engine.HTTP_LIMITS
                or p["policy"] != engine.POLICY or p["review_policy"] != REVIEW_POLICY
                or p["phase_parameters"] != json.loads(json.dumps(engine.PHASE_PARAMETERS))
                or p["agent_parameters"] != json.loads(json.dumps(engine.AGENT_PARAMETERS))
                or type(p["authorized_model_requests"]) is not int or p["authorized_model_requests"] != 0
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != 168
                or p["dependencies"] != engine._dependencies() or p["plan"] != engine._plan(engine.cases())
                or p["effective_environment"] != effective_environment() or p["data_proof_sha256"] != data_binding()):
            base._fail("task_repair_protocol_changed")
        if (p["artifacts"] != _artifacts() or p["cases_sha256"] != bench.sha256(engine.CASES_PATH)
                or p["offline_data_proof"] != engine.offline_data_proof() or p["prior_proof"] != prior_proof()):
            base._fail("task_repair_frozen_evidence_changed")
        if type(p["initial_intent_requests"]) is not list or len(p["initial_intent_requests"]) != 6:
            base._fail("task_initial_requests_changed")
        for op, case in zip(p["initial_intent_requests"], engine.cases()):
            if (type(op) is not dict or set(op) != {"case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or op["case_id"] != case["id"]
                    or op["input_sha256"] != canonical_payload_hash({"message": case["turns"][0], "history": []})
                    or op["payload_sha256"] != canonical_payload_hash(op["payload"])
                    or type(op["request_bytes_sha256"]) is not str
                    or re.fullmatch(r"[a-f0-9]{64}", op["request_bytes_sha256"]) is None):
                base._fail("task_initial_requests_changed")
        self.ledger(p).status()
        self._sealed(p)
        actual = base._path(self.root, "data-proof.json")
        if actual.exists() and canonical_payload_hash(bench.read_json(actual)) != p["data_proof_sha256"]:
            base._fail("task_repair_data_identity_changed")
        return p

    async def preflight(self):
        p = self.verify()
        result = await super().preflight()
        if result["ready"] and canonical_payload_hash(result["data_proof"]) != p["data_proof_sha256"]:
            base._fail("task_repair_data_identity_changed")
        return result

    def report(self):
        return {**super().report(), "batch": BATCH, "candidate": CANDIDATE,
                "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")),
                "review_policy": deepcopy(REVIEW_POLICY)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "status", "report", "preflight", "authorize", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-new-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(argv)
    trial = TaskRepair()
    try:
        if args.operation == "prepare":
            value = asyncio.run(trial.prepare())
        elif args.operation == "verify":
            p = trial.verify()
            value = {"verified": True, "model_requests": trial.ledger(p).status()["attempts"]}
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
            value = supervise([sys.executable, "-B", "-u", "-m", "evaluation.task_repair", "_worker"], root=ROOT)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        print("STOP: task_repair_failed; inspect private receipts, do not retry or reinitialize", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
