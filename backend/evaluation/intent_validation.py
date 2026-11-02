"""Fixed 36-case validation batch; preparation grants no model calls.

The development run remains sealed and is read as evidence, never resumed.
Only a new protocol-bound authorization can admit this batch's one-send attempts.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation.intent_transport import (
    GuardedIntentTransport, TrialTransportError, canonical_payload_hash,
    messages_url, validated_endpoint,
)
from evaluation.validation_ledger import ValidationLedger
from evaluation.trial_ledger import TrialLedgerError


BATCH = "serveflow-intent-validation-observation-v1"
TRIAL_ROOT = bench.BACKEND_ROOT / "data" / "benchmarks" / "intent-validation-v1"
MAX_REQUESTS = 36
LIMITS = {**base.LIMITS, "physical_requests": MAX_REQUESTS}
EXTENSION_FILES = ["evaluation/intent_validation.py", "evaluation/validation_ledger.py",
                   "tests/test_intent_validation.py", "tests/test_validation_ledger.py"]
PROTOCOL_FIELDS = {
    "schema_version", "batch", "split", "phase", "created_at", "git_head", "root_binding_sha256",
    "offline_manifest_path", "offline_manifest_sha256", "extension_artifacts", "dependencies",
    "model", "base_url", "messages_url", "limits", "decision_policy", "stop_policy", "operations",
    "intent_suite_sha256", "development_proof", "authorized_model_requests", "proposed_model_requests",
}
APPROVAL_FIELDS = base.APPROVAL_FIELDS
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ValidationError(base.TrialError):
    """Public failures carry fixed codes only."""


def _fail(code):
    raise ValidationError(code)


def _configuration(*, secret=False):
    return base._configuration(secret=secret)


def validation_cases():
    """Retain dataset order, without selecting or tuning on observed answers."""
    return [case for case in bench.load_suites()["intent"]["cases"] if case["split"] == "validation"]


def _ledger_snapshot(root):
    ledger = Path(root) / "ledger"
    # Hash every immutable claim as well as identity and journal. Merely hashing
    # the last chain value would miss truncation or a removed claim.
    files = [ledger / "identity.json", ledger / "journal.jsonl"]
    claims = ledger / "claims"
    if not claims.is_dir() or claims.is_symlink():
        _fail("development_ledger_invalid")
    files.extend(sorted(claims.iterdir()))
    hashes = {}
    for path in files:
        if (not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
                or path.stat().st_mode & 0o077):
            _fail("development_ledger_invalid")
        hashes[str(path.relative_to(ledger))] = bench.sha256(path)
    return canonical_payload_hash(hashes)


def development_proof(root=base.TRIAL_ROOT):
    """Read-only verification of the completed, sealed 24-request live batch.

    Calls existing verification/reporting only; does not write a report, rescore
    into old files, repair a ledger, or grant/reuse its consumed authorization.
    """
    root = Path(root).absolute()
    before = _ledger_snapshot(root)
    protocol = base.verify(root)
    result = base.report(root)
    state = result["ledger"]
    expected = [op["operation_id"] for op in protocol["operations"]]
    if (state["attempts"] != 24 or len(expected) != 24 or state["completed"] != expected
            or state["pending"] or state["paused"] or (root / "stopped.json").exists()
            or result["evidence_kind"] != "live-shared-observation"
            or len(result["observation_sha256"]) != 24
            or result["paired_counts"]["execution_unknown"] or result["paired_counts"]["not_executed"]):
        _fail("development_not_sealed")
    after = _ledger_snapshot(root)
    if before != after:
        _fail("development_changed_during_read")
    return {"root_binding_sha256": base._root_hash(root),
            "protocol_sha256": bench.sha256(base._path(root, "protocol.json")),
            "ledger_sha256": after, "ledger_chain_hash": state["chain_hash"],
            "observation_sha256": result["observation_sha256"]}


async def _capture(case, base_url):
    operation = await base._capture(case, base_url)
    # This is runner metadata, outside the SDK payload and model input.
    operation["operation_id"] = BATCH + ":" + case["id"] + ":observation-1"
    return operation


def _observation_name(operation):
    return operation["case_id"] + "-observation.json"


def _record(case_id, status, *, predicted=None, latency=None, usage=None, arm=False):
    value = {"case_id": case_id, "status": status, "latency_ms": latency,
             "usage": usage or {"input_tokens": None, "output_tokens": None, "physical_requests": None}}
    if arm:
        value["predicted_intent"] = predicted
    return value


class ValidationTrial:
    # Narrow extension points for another fixed, independently authorized batch.
    # The original validation defaults and prepare/verify contract stay fixed.
    batch = BATCH
    max_requests = MAX_REQUESTS
    split = "validation"
    authorization_schema = "serveflow-intent-validation-authorization-v1"
    observation_schema = "serveflow-intent-validation-observation-v1"
    report_schema = "serveflow-intent-validation-report-v1"
    stop_schema = "serveflow-intent-validation-stop-v1"
    report_limitations = [
        "36 authored validation cases; visible to dataset authors, not independent human blind evaluation",
        "shared observation ablation, not independent A/B", "production parser retained",
        "shared usage counted once; per-arm resources unknown", "single measurement; no stability claim"]

    def _cases(self):
        return validation_cases()

    def _configuration(self, *, secret=False):
        return _configuration(secret=secret)

    def __init__(self, root=TRIAL_ROOT, development_root=base.TRIAL_ROOT):
        self.root = Path(root).absolute()
        self.development_root = Path(development_root).absolute()
        current, previous = self.root.resolve(), self.development_root.resolve()
        if current.is_relative_to(previous) or previous.is_relative_to(current):
            _fail("development_root_forbidden")

    def _ledger(self, protocol):
        return ValidationLedger(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")),
                                [op["operation_id"] for op in protocol["operations"]], MAX_REQUESTS)

    async def prepare(self, offline_manifest):
        if base._path(self.root, "protocol.json").exists() or (self.root / "ledger").exists():
            _fail("trial_already_prepared")
        offline_manifest = Path(offline_manifest).resolve(strict=True)
        bench.verify_manifest(bench.read_json(offline_manifest))
        proof = development_proof(self.development_root)
        config = _configuration()
        cases = validation_cases()
        if len(cases) != MAX_REQUESTS:
            _fail("operation_count")
        operations = [await _capture(case, config["base_url"]) for case in cases]
        protocol = {
            "schema_version": "serveflow-intent-validation-v1", "batch": BATCH, "split": "validation",
            "phase": "awaiting-explicit-authorization", "created_at": base._now(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root),
            "offline_manifest_path": str(offline_manifest), "offline_manifest_sha256": bench.sha256(offline_manifest),
            "extension_artifacts": {name: bench.sha256(bench.BACKEND_ROOT / name) for name in EXTENSION_FILES},
            "dependencies": base._dependencies(), "model": base.MODEL,
            "base_url": validated_endpoint(config["base_url"]), "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(LIMITS), "decision_policy": deepcopy(base.DECISIONS),
            "stop_policy": deepcopy(base.STOP_POLICY), "operations": operations,
            "intent_suite_sha256": bench.sha256(bench.DATASET_ROOT / "intent-v1.json"),
            "development_proof": proof, "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
        }
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        base._write(self.root, "protocol.json", protocol)
        if self._ledger(protocol).status()["attempts"]:
            _fail("unexpected_existing_attempts")
        return protocol

    def verify(self):
        protocol = bench.read_json(base._path(self.root, "protocol.json"))
        if type(protocol) is not dict or set(protocol) != PROTOCOL_FIELDS:
            _fail("invalid_protocol")
        if (protocol["schema_version"] != "serveflow-intent-validation-v1" or protocol["batch"] != BATCH
                or protocol["split"] != "validation" or protocol["phase"] != "awaiting-explicit-authorization"
                or protocol["model"] != base.MODEL or protocol["limits"] != LIMITS
                or protocol["decision_policy"] != base.DECISIONS or protocol["stop_policy"] != base.STOP_POLICY
                or type(protocol["authorized_model_requests"]) is not int or protocol["authorized_model_requests"] != 0
                or type(protocol["proposed_model_requests"]) is not int or protocol["proposed_model_requests"] != MAX_REQUESTS
                or protocol["root_binding_sha256"] != base._root_hash(self.root)
                or protocol["dependencies"] != base._dependencies()):
            _fail("protocol_version")
        config = _configuration()
        if (protocol["base_url"] != validated_endpoint(config["base_url"])
                or protocol["messages_url"] != messages_url(config["base_url"])):
            _fail("configuration")
        manifest = Path(protocol["offline_manifest_path"])
        if bench.sha256(manifest) != protocol["offline_manifest_sha256"]:
            _fail("offline_manifest_changed")
        bench.verify_manifest(bench.read_json(manifest))
        if (type(protocol["extension_artifacts"]) is not dict
                or set(protocol["extension_artifacts"]) != set(EXTENSION_FILES)
                or any(bench.sha256(bench.BACKEND_ROOT / name) != digest
                       for name, digest in protocol["extension_artifacts"].items())):
            _fail("extension_artifacts_changed")
        if protocol["intent_suite_sha256"] != bench.sha256(bench.DATASET_ROOT / "intent-v1.json"):
            _fail("intent_suite_changed")
        if protocol["development_proof"] != development_proof(self.development_root):
            _fail("development_evidence_changed")
        cases, operations = validation_cases(), protocol["operations"]
        if type(operations) is not list or len(operations) != MAX_REQUESTS or len(cases) != MAX_REQUESTS:
            _fail("operation_count")
        for operation, case in zip(operations, cases):
            if (type(operation) is not dict or set(operation) != {
                    "case_id", "operation_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or operation["case_id"] != case["id"]
                    or operation["operation_id"] != BATCH + ":" + case["id"] + ":observation-1"
                    or operation["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                    or operation["payload_sha256"] != canonical_payload_hash(operation["payload"])
                    or type(operation["request_bytes_sha256"]) is not str
                    or not SHA256.fullmatch(operation["request_bytes_sha256"])):
                _fail("operation_binding")
        if not (self.root / "ledger" / "identity.json").is_file() or not (self.root / "ledger" / "journal.jsonl").is_file():
            _fail("ledger_missing")
        self._ledger(protocol).status()
        return protocol

    def status(self):
        return self._ledger(self.verify()).status()

    def authorize(self, reference, *, confirmed=False):
        if confirmed is not True or type(reference) is not str or not 1 <= len(reference.strip()) <= 500:
            _fail("explicit_authorization_required")
        protocol = self.verify()
        state = self._ledger(protocol).status()
        if state["attempts"] or state["pending"] or state["paused"] or (self.root / "stopped.json").exists():
            _fail("cannot_authorize_started_trial")
        value = {"schema_version": self.authorization_schema, "batch": self.batch,
                 "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")),
                 "root_binding_sha256": base._root_hash(self.root), "max_requests": self.max_requests,
                 "user_authorization_reference": reference.strip(), "created_at": base._now(),
                 "initial_ledger_chain_hash": state["chain_hash"]}
        base._write(self.root, "authorization.json", value)
        return value

    def _approval(self, protocol):
        path = base._path(self.root, "authorization.json")
        if not path.is_file():
            _fail("explicit_authorization_required")
        value = bench.read_json(path)
        if (type(value) is not dict or set(value) != APPROVAL_FIELDS
                or value["schema_version"] != self.authorization_schema or value["batch"] != self.batch
                or type(value["max_requests"]) is not int or value["max_requests"] != self.max_requests
                or value["protocol_sha256"] != bench.sha256(base._path(self.root, "protocol.json"))
                or value["root_binding_sha256"] != base._root_hash(self.root)
                or type(value["user_authorization_reference"]) is not str
                or not 1 <= len(value["user_authorization_reference"].strip()) <= 500
                or value["initial_ledger_chain_hash"] != self._ledger(protocol).initial_chain_hash):
            _fail("authorization_binding")
        return value

    def _sealed(self, protocol, ledger, expected_kind=None):
        sealed = ledger.results()
        operations = {op["operation_id"]: op for op in protocol["operations"]}
        protocol_hash = bench.sha256(base._path(self.root, "protocol.json"))
        for operation_id, result in sealed.items():
            operation = operations.get(operation_id)
            if operation is None:
                _fail("observation_binding")
            path = base._path(self.root, _observation_name(operation))
            if not path.is_file():
                _fail("completed_observation_missing")
            if result["output_sha256"] != bench.sha256(path):
                _fail("observation_hash_changed")
            value = bench.read_json(path)
            if (type(value) is not dict or value.get("schema_version") != self.observation_schema
                    or value.get("batch") != self.batch or value.get("protocol_sha256") != protocol_hash
                    or value.get("operation_id") != operation_id or value.get("case_id") != operation["case_id"]
                    or value.get("payload_sha256") != operation["payload_sha256"]
                    or value.get("request_bytes_sha256") != operation["request_bytes_sha256"]
                    or value.get("evidence_kind") not in {"controlled-transport", "live-model"}):
                _fail("observation_binding")
            if expected_kind is not None and value["evidence_kind"] != expected_kind:
                _fail("mixed_or_unknown_evidence")
        return sealed

    async def _execute_operation(self, protocol, operation, case, *, config, delegate=None):
        import httpx
        from anthropic import AsyncAnthropic
        if delegate is not None and not isinstance(delegate, httpx.MockTransport):
            _fail("invalid_controlled_transport")
        ledger, reserved, holder = self._ledger(protocol), False, {}
        kind = "controlled-transport" if delegate is not None else "live-model"

        def admit():
            nonlocal reserved
            self.verify()
            self._approval(protocol)
            if (self.root / "stopped.json").exists():
                _fail("trial_paused")
            self._sealed(protocol, ledger, kind)
            if guard.request_bytes_sha256 != operation["request_bytes_sha256"]:
                _fail("serialized_request_changed")
            ledger.reserve(operation["operation_id"], operation["payload_sha256"])
            reserved = True

        guard = GuardedIntentTransport(protocol["base_url"], operation["payload_sha256"], admit, transport=delegate)
        http = httpx.AsyncClient(transport=guard, follow_redirects=False, trust_env=False,
                                 timeout=8, headers={"Accept-Encoding": "identity"})
        sdk = AsyncAnthropic(api_key=config["api_key"], base_url=protocol["base_url"],
                             max_retries=0, timeout=8, http_client=http)

        async def create(**kwargs):
            response = await sdk.messages.create(**kwargs)
            holder["response"] = response
            return response

        recognizer = base._recognizer(SimpleNamespace(base_url=sdk.base_url, messages=SimpleNamespace(create=create)))
        started, llm = time.perf_counter(), None
        with base._quiet_clients():
            try:
                async with asyncio.timeout(8):
                    llm = await recognizer._llm_recognize(**bench.model_input(case))
            except TimeoutError:
                guard.failure_code = guard.failure_code or "timeout"
            except asyncio.CancelledError:
                guard.failure_code = guard.failure_code or "cancelled"
            finally:
                await sdk.close()
        elapsed = (time.perf_counter() - started) * 1000
        if not reserved or not guard.sent:
            _fail("send_admission_failed")
        error_code = base._error(guard.failure_code) if guard.failure_code else None
        if llm is None:
            from core.intent_recognizer import IntentCategory
            llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
        fusion = None if error_code else await base._local_fusion(recognizer, case, llm)
        output = {"schema_version": self.observation_schema, "batch": self.batch,
                  "evidence_kind": kind, "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")),
                  "operation_id": operation["operation_id"], "case_id": operation["case_id"],
                  "payload_sha256": operation["payload_sha256"], "request_bytes_sha256": guard.request_bytes_sha256,
                  "request_model": base.MODEL,
                  "transport": {"sent": guard.sent, "http_status": guard.http_status,
                                "request_id": guard.request_id, "error_code": error_code},
                  "response": base._response_metadata(holder.get("response")),
                  "llm": {"intent": llm["intent"].value, "confidence": llm.get("confidence", 0.0),
                          "parser_failed": bool(llm.get("failed"))},
                  "fusion": fusion, "shared_latency_ms": elapsed}
        name = _observation_name(operation)
        base._write(self.root, name, output)
        outcome = ("cancelled" if error_code == "cancelled" else "execution_error" if error_code
                   else "structure_failed" if llm.get("failed") else "received")
        ledger.finish(operation["operation_id"], outcome, bench.sha256(base._path(self.root, name)), error_code)
        return output

    def _stop(self):
        try:
            base._write(self.root, "stopped.json", {
                "schema_version": self.stop_schema, "error_code": "execution_stopped",
                "recorded_at": base._now()})
        except (OSError, ValueError):
            pass

    async def execute(self, *, delegate_factory=None):
        protocol = self.verify()
        self._approval(protocol)
        ledger = self._ledger(protocol)
        state = ledger.status()
        if state["paused"] or state["pending"] or (self.root / "stopped.json").exists():
            _fail("trial_paused")
        self._sealed(protocol, ledger, "controlled-transport" if delegate_factory is not None else "live-model")
        config = self._configuration(secret=True)
        for operation, case in zip(protocol["operations"], self._cases()):
            if (self.root / "stopped.json").exists():
                _fail("trial_paused")
            if operation["operation_id"] in state["completed"]:
                continue
            try:
                delegate = delegate_factory(operation) if delegate_factory is not None else None
                if delegate_factory is not None:
                    import httpx
                    # A faulty test factory must never fall through to the live
                    # transport merely because it returned None.
                    if not isinstance(delegate, httpx.MockTransport):
                        _fail("invalid_controlled_transport")
                output = await self._execute_operation(protocol, operation, case, config=config,
                                                       delegate=delegate)
                state = ledger.status()
            except (Exception, asyncio.CancelledError):
                self._stop()
                _fail("execution_stopped")
            print(json.dumps({"case_id": operation["case_id"], "attempts": state["attempts"], "paused": state["paused"]}))
            if output["transport"]["error_code"]:
                _fail("trial_paused")
        return self.report()

    def report(self):
        from evaluation.benchmark_metrics import aggregate_resources, score_intent
        protocol = self.verify()
        ledger = self._ledger(protocol)
        state = ledger.status()
        sealed = self._sealed(protocol, ledger)
        if state["attempts"]:
            self._approval(protocol)
        records, resources, hashes, kinds = {"llm_only": [], "fusion": []}, [], [], set()
        for operation in protocol["operations"]:
            operation_id, case_id = operation["operation_id"], operation["case_id"]
            path = base._path(self.root, _observation_name(operation))
            if operation_id not in sealed:
                if path.exists() and operation_id not in state["pending"]:
                    _fail("unsealed_observation")
                if operation_id in state["pending"]:
                    # A crash after writing an observation but before finish()
                    # leaves no sealed result. Do not read, score or retry it.
                    for arm in records:
                        records[arm].append(_record(case_id, "unknown", arm=True))
                    resources.append(_record(case_id, "unknown"))
                continue
            value = bench.read_json(path)
            digest = bench.sha256(path)
            if digest != sealed[operation_id]["output_sha256"]:
                _fail("observation_hash_changed")
            hashes.append(digest)
            kinds.add(value["evidence_kind"])
            valid = value["transport"]["error_code"] is None
            for arm, decision in (("llm_only", value["llm"]), ("fusion", value["fusion"])):
                success = valid and decision is not None and (arm != "llm_only" or not decision["parser_failed"])
                records[arm].append(_record(case_id, "success" if success else "failed", arm=True,
                                             predicted=decision["intent"] if decision else None))
            resources.append(_record(case_id, "success" if valid else "failed", latency=value["shared_latency_ms"],
                                     usage={**value["response"]["usage"], "physical_requests": 1}))
        if len(kinds) > 1:
            _fail("mixed_or_unknown_evidence")
        scores = {arm: score_intent(self._cases(), rows) for arm, rows in records.items()}
        matrix = dict.fromkeys(("both_correct", "llm_only_correct", "fusion_only_correct", "both_incorrect",
                                "execution_unknown", "not_executed"), 0)
        paired = []
        for first, second in zip(scores["llm_only"]["per_case"], scores["fusion"]["per_case"]):
            key = ("not_executed" if first["status"] == second["status"] == "not_executed"
                   else "execution_unknown" if "unknown" in (first["status"], second["status"])
                   else "both_correct" if first["correct"] and second["correct"]
                   else "llm_only_correct" if first["correct"] else "fusion_only_correct" if second["correct"]
                   else "both_incorrect")
            matrix[key] += 1
            paired.append({"case_id": first["case_id"], "outcome": key})
        if ledger.status()["chain_hash"] != state["chain_hash"]:
            _fail("ledger_changed_during_report")
        return {"schema_version": self.report_schema, "batch": self.batch, "split": self.split,
                "evidence_kind": ("controlled-shared-observation" if kinds == {"controlled-transport"}
                                  else "live-shared-observation" if kinds == {"live-model"}
                                  else "pending-send" if state["attempts"] else "not-run"),
                "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")), "ledger": state,
                "scores": scores, "paired_counts": matrix, "paired_cases": paired,
                "shared_resources": aggregate_resources(resources), "observation_sha256": hashes,
                "limits": deepcopy(self.report_limitations)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--offline-manifest", type=Path, required=True)
    for name in ("verify", "status", "report", "run"):
        commands.add_parser(name)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--reference", required=True)
    authorize.add_argument("--confirm-user-authorization", action="store_true")
    args = parser.parse_args(argv)
    trial = ValidationTrial()
    try:
        if args.command == "prepare":
            asyncio.run(trial.prepare(args.offline_manifest))
            value = {"prepared": True, "proposed_physical_requests": MAX_REQUESTS, "authorized_model_requests": 0}
        elif args.command == "authorize":
            trial.authorize(args.reference, confirmed=args.confirm_user_authorization)
            value = {"authorization_recorded": True, "max_physical_requests": MAX_REQUESTS}
        elif args.command == "verify":
            trial.verify()
            value = {"verified": True, "physical_requests": trial.status()["attempts"]}
        elif args.command == "status":
            value = trial.status()
        else:
            report = asyncio.run(trial.execute()) if args.command == "run" else trial.report()
            name = "report-" + report["ledger"]["chain_hash"] + ".json"
            path = base._path(trial.root, name)
            if not path.exists():
                base._write(trial.root, name, report)
            elif bench.read_json(path) != report:
                _fail("saved_report_changed")
            value = {"attempts": report["ledger"]["attempts"], "paused": report["ledger"]["paused"],
                     "evidence_kind": report["evidence_kind"], "paired_counts": report["paired_counts"]}
        print(json.dumps(value, ensure_ascii=False))
        return 0
    except Exception as error:
        code = (error.code if isinstance(error, (TrialTransportError, TrialLedgerError))
                else str(error) if isinstance(error, (ValidationError, base.TrialError)) else "validation_input_or_storage_failure")
        print(json.dumps({"error": code, "no_automatic_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
