"""Fixed six-case / seven-turn customer-service evaluation; preparation grants 0 sends.

No application endpoint, production quota, business data or old score is changed.
Only a separately confirmed, protocol-bound authorization admits model work.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation.benchmark_metrics import REVIEW_DIMENSIONS, REVIEW_METHODS, _resources
from evaluation.intent_transport import canonical_payload_hash, messages_url
from evaluation.task_budget import TaskBudget, LIMITS
from evaluation.task_data import offline_data_proof, build_live_data, parse_mysql_url
from evaluation.task_runtime import execute_case
from evaluation.task_transport import TaskHTTPTransport, PHASE_PARAMETERS, AGENT_PARAMETERS, HTTP_LIMITS

ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-e2e-v1"
CASES_PATH = bench.DATASET_ROOT / "task-e2e-v1.json"
BATCH = "serveflow-task-e2e-v1"
MODULES = ("task_trial", "task_budget", "task_data", "task_evidence", "task_transport", "task_runtime", "task_supervisor")
REVIEW_FIELDS = REVIEW_DIMENSIONS
POLICY = {"scope": "Orchestrator.run-not-HTTP-or-memory", "max_tool_calls_per_turn": 12,
          "max_rag_calls_per_turn": 3, "outer_process_deadline_s": 900,
          "judge_attempts_per_turn": 1, "judge_screening_threshold": 0.75,
          "remote_embedding": False, "background_memory": False, "external_mcp": False,
          "requires_manual_review": True,
          "dynamic_payloads": "freeze initial intent; hash each actual SDK body before reserve/send",
          "stop": "execution/version/data/ledger failure; no retry or automatic recovery",
          "ordinary_failure": "retain failed actor/structure/judge/semantic outcome; continue fixed plan"}


def cases():
    data = bench.read_json(CASES_PATH)
    bench._exact(data, {"schema_version", "dataset_id", "provenance", "scope", "cases"}, "task_cases_fields")
    if (data["schema_version"] != "serveflow-task-e2e-cases-v1" or data["dataset_id"] != "task-e2e-v1"
            or data["provenance"] != "synthetic-authored-not-independent-blind-test"
            or data["scope"] != "orchestrator-not-http-memory" or type(data["cases"]) is not list
            or len(data["cases"]) != 6):
        base._fail("task_cases_invalid")
    corpus, tables = bench._sources(bench.BACKEND_ROOT)
    original = {c["id"]: c for c in bench.load_suites()["tasks"]["cases"]}
    for number, case in enumerate(data["cases"], 1):
        bench._exact(case, {"id", "source_case_id", "variant", "customer_id", "turns", "expected", "references", "tags"}, "task_case_fields")
        if (case["id"] != f"e{number:02d}" or type(case["variant"]) is not bool
                or case["customer_id"] not in {"demo-customer-a", "demo-customer-b"}
                or type(case["turns"]) is not list or not 1 <= len(case["turns"]) <= 2):
            base._fail("task_case_invalid")
        for question in case["turns"]:
            bench._text(question, 2000)
        bench._exact(case["expected"], {"required_agents", "required_tools", "required_facts", "forbidden_claims", "rationale"}, "task_expected_fields")
        for name in ("required_agents", "required_tools", "required_facts", "forbidden_claims"):
            bench._strings(case["expected"][name])
        bench._text(case["expected"]["rationale"])
        bench._strings(case["tags"], allow_empty=False)
        bench._references(case["references"], corpus, tables)
        if case["source_case_id"] is not None:
            old = original.get(case["source_case_id"])
            if old is None:
                base._fail("task_original_missing")
            if not case["variant"] and (case["turns"] != [old["input"]["message"]]
                    or case["expected"] != old["expected"] or case["references"] != old["references"]):
                base._fail("task_original_changed")
    if sum(len(c["turns"]) for c in data["cases"]) != 7:
        base._fail("task_turn_count")
    return data["cases"]


def _plan(values):
    return [{"case_id": c["id"], "turn_ids": [f"turn-{n}" for n in range(1, len(c["turns"]) + 1)]} for c in values]


def _artifacts():
    paths = bench.artifact_paths() + [f"evaluation/{name}.py" for name in MODULES]
    paths += [f"tests/test_{name}.py" for name in MODULES]
    paths += ["evaluation/benchmark/task-e2e-v1.json", "evaluation/intent_trial.py", "evaluation/intent_transport.py",
              "evaluation/intent_validation.py", "evaluation/intent_recheck.py", "evaluation/trial_ledger.py", "evaluation/validation_ledger.py",
              "demo_business/initialize.py", "core/redaction.py"]
    return {p: bench.sha256(bench.BACKEND_ROOT / p) for p in sorted(set(paths))}


def _dependencies():
    from importlib import metadata
    return {**base._dependencies(), **{name: metadata.version(name) for name in (
        "chromadb", "PyMySQL", "numpy", "onnxruntime", "tokenizers", "jsonschema", "pydantic", "python-dotenv")},
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "os": platform.system(), "machine": platform.machine()}


def prior_proof():
    from evaluation.intent_recheck import RecheckTrial, ROOT as previous
    from evaluation.intent_validation import _ledger_snapshot
    before = _ledger_snapshot(previous)
    trial = RecheckTrial(previous)
    protocol, report = trial.verify(), trial.report()
    if report["ledger"]["attempts"] != 24 or report["ledger"]["paused"] or report["ledger"]["pending"]:
        base._fail("prior_trial_unsealed")
    report_name = "report-" + report["ledger"]["chain_hash"] + ".json"
    if bench.read_json(base._path(previous, report_name)) != report:
        base._fail("prior_report_changed")
    hashes = {name: bench.sha256(base._path(previous, name))
              for name in ("protocol.json", "authorization.json", report_name)}
    if before != _ledger_snapshot(previous):
        base._fail("prior_trial_changed")
    return {"root_binding_sha256": base._root_hash(previous), "hashes": hashes,
            "ledger_sha256": before, "chain_hash": report["ledger"]["chain_hash"],
            "archived_evidence": protocol["archived_evidence_proof"]}


def _mysql_url():
    # A dedicated variable is mandatory; never fall back to the normal demo DB.
    from dotenv import dotenv_values
    return os.environ.get("SERVEFLOW_EVAL_MYSQL_URL", dotenv_values(bench.BACKEND_ROOT / ".env").get("SERVEFLOW_EVAL_MYSQL_URL"))


class TaskTrial:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        base._path(self.root, "protocol.json")
        historical = bench.BACKEND_ROOT / "data/benchmarks"
        if self.root != ROOT and self.root.is_relative_to(historical):
            base._fail("fixed_task_root_required")

    def ledger(self, protocol):
        return TaskBudget(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")), protocol["plan"])

    async def prepare(self):
        if self.root.exists():
            base._fail("task_root_already_exists")
        values, config = cases(), base._configuration()
        initial = []
        for case in values:
            item = await base._capture({"id": case["id"], "input": {"message": case["turns"][0], "history": []}}, config["base_url"])
            item.pop("operation_id")
            initial.append(item)
        protocol = {"schema_version": "serveflow-task-protocol-v1", "batch": BATCH,
            "created_at": base._now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": _artifacts(),
            "cases_sha256": bench.sha256(CASES_PATH), "dependencies": _dependencies(),
            "model": base.MODEL, "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(LIMITS), "http_limits": deepcopy(HTTP_LIMITS), "policy": deepcopy(POLICY),
            "phase_parameters": deepcopy(PHASE_PARAMETERS), "agent_parameters": deepcopy(AGENT_PARAMETERS),
            "plan": _plan(values), "initial_intent_requests": initial, "offline_data_proof": offline_data_proof(),
            "prior_proof": prior_proof(), "authorized_model_requests": 0, "proposed_model_requests": 168}
        # Normalize tuples to their persisted JSON representation before return/comparison.
        protocol = json.loads(json.dumps(protocol))
        self.root.mkdir(mode=0o700, parents=True)
        base._write(self.root, "protocol.json", protocol)
        TaskBudget(self.root / "ledger", bench.sha256(self.root / "protocol.json"), protocol["plan"], initialize=True)
        return protocol

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        expected = {"schema_version", "batch", "created_at", "git_head", "root_binding_sha256", "artifacts", "cases_sha256",
                    "dependencies", "model", "base_url", "messages_url", "limits", "http_limits", "policy", "phase_parameters",
                    "agent_parameters", "plan", "initial_intent_requests", "offline_data_proof", "prior_proof",
                    "authorized_model_requests", "proposed_model_requests"}
        bench._exact(p, expected, "task_protocol_fields")
        config = base._configuration()
        if (p["schema_version"] != "serveflow-task-protocol-v1" or p["batch"] != BATCH
                or p["root_binding_sha256"] != base._root_hash(self.root) or p["model"] != base.MODEL
                or p["base_url"] != config["base_url"] or p["messages_url"] != messages_url(config["base_url"])
                or p["limits"] != LIMITS or p["http_limits"] != HTTP_LIMITS or p["policy"] != POLICY
                or p["phase_parameters"] != json.loads(json.dumps(PHASE_PARAMETERS))
                or p["agent_parameters"] != json.loads(json.dumps(AGENT_PARAMETERS))
                or type(p["authorized_model_requests"]) is not int or p["authorized_model_requests"] != 0
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != 168
                or p["dependencies"] != _dependencies() or p["plan"] != _plan(cases())):
            base._fail("task_protocol_changed")
        if (p["artifacts"] != _artifacts() or p["cases_sha256"] != bench.sha256(CASES_PATH)
                or p["offline_data_proof"] != offline_data_proof() or p["prior_proof"] != prior_proof()):
            base._fail("task_frozen_evidence_changed")
        if len(p["initial_intent_requests"]) != 6:
            base._fail("task_initial_requests_changed")
        for op, case in zip(p["initial_intent_requests"], cases()):
            if (set(op) != {"case_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or op["case_id"] != case["id"] or op["input_sha256"] != canonical_payload_hash({"message": case["turns"][0], "history": []})
                    or op["payload_sha256"] != canonical_payload_hash(op["payload"])):
                base._fail("task_initial_requests_changed")
        self.ledger(p).status()
        self._sealed(p)
        return p

    def _sealed(self, p):
        result = self.ledger(p).results()
        for operation in result["operations"].values():
            if operation["output_sha256"] is not None:
                path = base._path(self.root, f"physical-{operation['attempt']:03d}.json")
                if bench.sha256(path) != operation["output_sha256"]:
                    base._fail("task_physical_changed")
        for case_id, item in result["cases"].items():
            if bench.sha256(base._path(self.root, f"case-{case_id}.json")) != item["artifact_sha256"]:
                base._fail("task_case_changed")
        return result

    def authorize(self, reference, *, confirmed=False):
        p = self.verify()
        if confirmed is not True or type(reference) is not str or not reference.strip() or len(reference) > 500:
            base._fail("task_explicit_authorization_required")
        ledger = self.ledger(p)
        if ledger.status()["attempts"] or ledger.status()["active_case"]:
            base._fail("task_already_started")
        value = {"schema_version": "serveflow-task-authorization-v1", "reference": reference,
            "recorded_at": base._now(), "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "root_binding_sha256": base._root_hash(self.root), "initial_chain_hash": ledger.initial_chain_hash,
            "max_requests": 168, "plan": p["plan"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, p):
        value = bench.read_json(base._path(self.root, "authorization.json"))
        bench._exact(value, {"schema_version", "reference", "recorded_at", "protocol_sha256", "root_binding_sha256", "initial_chain_hash", "max_requests", "plan"}, "task_authorization_fields")
        if (value["schema_version"] != "serveflow-task-authorization-v1"
                or value["protocol_sha256"] != bench.sha256(self.root / "protocol.json")
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or value["initial_chain_hash"] != self.ledger(p).initial_chain_hash
                or type(value["max_requests"]) is not int or value["max_requests"] != 168 or value["plan"] != p["plan"]):
            base._fail("task_authorization_changed")
        return value

    async def preflight(self):
        self.verify()
        url = _mysql_url()
        if not url:
            return {"ready": False, "reason": "dedicated_eval_mysql_not_configured", "model_requests": 0}
        parse_mysql_url(url)
        data = await build_live_data(url)
        try:
            data.assert_healthy()
            return {"ready": True, "data_proof": data.proof, "model_requests": 0}
        finally:
            await data.close()

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        """Library mocks require BOTH controlled data and MockTransport factory."""
        from anthropic import AsyncAnthropic
        import httpx
        p = self.verify()
        self._approval(p)  # Before secrets, data connections or any model client.
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("task_incomplete_controlled_boundary")
        for name in ("stopped.json", "supervisor-stop.json", "worker-stop.json"):
            if (self.root / name).exists():
                base._fail("task_paused")
        config = base._configuration(secret=delegate_factory is None)
        ledger = self.ledger(p)
        data, transport, sdk = controlled_data, None, None
        with ledger.exclusive_run() as lease:
            try:
                if data is None:
                    url = _mysql_url()
                    if not url:
                        base._fail("dedicated_eval_mysql_not_configured")
                    data = await build_live_data(url)
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("task_controlled_data_required")
                data.assert_healthy()
                proof_name = "data-proof.json"
                if not (self.root / proof_name).exists():
                    base._write(self.root, proof_name, data.proof)
                elif bench.read_json(base._path(self.root, proof_name)) != data.proof:
                    base._fail("task_data_proof_changed")
                proof_hash = bench.sha256(self.root / proof_name)
                def admission():
                    self.verify()
                    self._approval(p)
                    if bench.sha256(base._path(self.root, proof_name)) != proof_hash:
                        base._fail("task_data_proof_changed")
                    data.assert_healthy()
                transport = TaskHTTPTransport(config["base_url"], self.root, ledger, lease, admission,
                                               delegate_factory=delegate_factory)
                http = httpx.AsyncClient(transport=transport, timeout=8, follow_redirects=False, trust_env=False,
                    headers={"Accept-Encoding": "identity"})
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
                        result = await execute_case(case, sdk, data, ensure_healthy=healthy)
                    result.update({"protocol_sha256": bench.sha256(self.root / "protocol.json"),
                        "evidence_kind": transport.kind, "data_proof_sha256": proof_hash})
                    name = f"case-{case['id']}.json"
                    base._write(self.root, name, result)
                    ledger.finish_case(case["id"], bench.sha256(self.root / name),
                        outcome="success" if result["execution_status"] == "success" else "failed", lease=lease)
            except BaseException as exc:
                path = self.root / "worker-stop.json"
                if not path.exists():
                    from evaluation.task_evidence import EvidenceError, FATAL_EVIDENCE_CODES
                    stopped = {"schema_version": "serveflow-task-worker-stop-v1",
                        "reason": "cancelled" if isinstance(exc, asyncio.CancelledError) else "execution_stopped",
                        "recorded_at": base._now()}
                    if isinstance(exc, EvidenceError) and exc.reason in FATAL_EVIDENCE_CODES:
                        stopped["evidence_error"] = exc.reason
                    base._write(self.root, path.name, stopped)
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
        bench._exact(value, {"case_id", "artifact_sha256", "reviewer", "method", "dimensions", "reason", "blocking_errors", "reading_burden"}, "task_review_fields")
        if value["case_id"] not in {c["id"] for c in cases()}:
            base._fail("task_review_case")
        for key in ("reviewer", "method", "reason"):
            bench._text(value[key], 4000)
        bench._strings(value["blocking_errors"])
        if value["method"] not in REVIEW_METHODS or value["reading_burden"] not in {"none", "some", "heavy"}:
            base._fail("task_review_method_or_burden")
        bench._exact(value["dimensions"], set(REVIEW_FIELDS), "task_review_dimensions")
        if not all(type(v) is bool for v in value["dimensions"].values()):
            base._fail("task_review_dimensions")
        if value["artifact_sha256"] != bench.sha256(base._path(self.root, f"case-{value['case_id']}.json")):
            base._fail("task_review_binding")

    def review(self, value):
        self.verify()
        self._review(value)
        base._write(self.root, f"review-{value['case_id']}.json", value)
        return value

    def report(self):
        p = self.verify()
        ledger = self.ledger(p)
        state, sealed = ledger.status(), self._sealed(p)
        rows, case_resources = [], []
        for case in cases():
            case_id = case["id"]
            status = "unknown" if state["active_case"] == case_id else "not_executed"
            row = {"case_id": case_id, "status": status, "planned_turns": len(case["turns"])}
            if case_id in sealed["cases"]:
                artifact = bench.read_json(base._path(self.root, f"case-{case_id}.json"))
                hard = artifact["execution_status"] == "success" and not artifact["deterministic"]["blockers"] and all(artifact["deterministic"]["checks"].values())
                judges = [turn["judge_overall"] for turn in artifact["turns"]]
                row.update({"deterministic_pass": hard, "judge_scores": judges,
                            "judge_ready": artifact["judge_ready"], "data_kind": artifact["data_kind"],
                            "evidence_kind": artifact["evidence_kind"]})
                case_resources.append(artifact)
                status = "failed" if not hard else "evaluation_failed" if not artifact["judge_ready"] else "pending_review"
                path = base._path(self.root, f"review-{case_id}.json")
                if path.exists():
                    review = bench.read_json(path)
                    self._review(review)
                    row["semantic_review"] = {**review, "review_sha256": bench.sha256(path)}
                    status = "passed" if hard and artifact["judge_ready"] and all(s is not None and s >= .75 for s in judges) and all(review["dimensions"].values()) and not review["blocking_errors"] else "failed"
                row["status"] = status
            rows.append(row)
        physical = [bench.read_json(base._path(self.root, f"physical-{op['attempt']:03d}.json"))
                    for op in sealed["operations"].values() if op["output_sha256"] is not None]
        token_totals = {}
        for key in ("input_tokens", "output_tokens"):
            known = [item["response"]["usage"][key] for item in physical if item["response"]["usage"][key] is not None]
            token_totals[key] = {"known_sum": sum(known) if known else None,
                                 "known_requests": len(known), "unknown_requests": state["attempts"] - len(known)}
        physical_resources = _resources([{"latency_ms": item["latency_ms"], "usage": {
            **item["response"]["usage"], "physical_requests": 1}} for item in physical])
        latency = {field: _resources([{"latency_ms": item.get(field), "usage": {
            "input_tokens": None, "output_tokens": None, "physical_requests": None}}
            for item in case_resources])["latency_ms"] for field in ("wall_latency_ms", "actor_path_latency_ms")}
        return {"schema_version": "serveflow-task-report-v1", "planned_cases": 6, "planned_turns": 7,
            "authorized_requests": 168 if (self.root / "authorization.json").exists() and self._approval(p) else 0,
            "ledger": state, "cases": rows, "counts": {status: sum(r["status"] == status for r in rows)
                for status in ("passed", "failed", "evaluation_failed", "pending_review", "unknown", "not_executed")},
            "usage": token_totals, "physical_latency_ms": [p["latency_ms"] for p in physical],
            "physical_resources": physical_resources, "case_latency": latency,
            "review_dimensions": {name: {"true_count": sum(row.get("semantic_review", {}).get("dimensions", {}).get(name) is True for row in rows),
                "reviewed_count": sum("semantic_review" in row for row in rows), "planned_denominator": 6} for name in REVIEW_FIELDS},
            "limitations": ["authored small known scenarios; no blind or generalization claim", "Judge scores are not fact accuracy", "manual review and deterministic blockers are separate gates", "does not test HTTP, browser, Redis or background memory"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "status", "report", "preflight", "authorize", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-new-authorization", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args(argv)
    trial = TaskTrial()
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
            value = supervise([sys.executable, "-B", "-u", "-m", "evaluation.task_trial", "_worker"], root=ROOT)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        # Never print arbitrary provider/DB URLs, prompt bodies or exception inputs.
        print("STOP: task_trial_failed; inspect private receipts, do not retry or reinitialize", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
