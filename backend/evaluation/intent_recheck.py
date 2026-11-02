"""Prepare a fixed post-fix intent recheck; preparation authorizes zero sends.

Two seen regressions and 22 new authored challenges share the existing bounded
transport/ledger path. Old snapshots are audited against archived Git sources,
never resumed, rewritten or made to match the current recognizer.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path
import json
import subprocess

from evaluation import benchmark as bench
from evaluation import intent_trial as base
from evaluation import intent_validation as engine
from evaluation.intent_transport import canonical_payload_hash, messages_url, validated_endpoint
from evaluation.trial_ledger import TrialLedger, TrialLedgerError
from evaluation.validation_ledger import ValidationLedger


BATCH = "serveflow-intent-recheck-v2"
MAX_REQUESTS = 24
ROOT = bench.BACKEND_ROOT / "data" / "benchmarks" / "intent-recheck-v2"
CASES_PATH = bench.DATASET_ROOT / "intent-recheck-v2.json"
LIMITS = {**base.LIMITS, "physical_requests": MAX_REQUESTS}
EXTRA_ARTIFACTS = (
    "evaluation/intent_recheck.py", "evaluation/intent_validation.py",
    "evaluation/intent_trial.py", "evaluation/intent_transport.py",
    "evaluation/trial_ledger.py", "evaluation/validation_ledger.py",
    "tests/test_intent_recheck.py", "evaluation/benchmark/intent-recheck-v2.json",
)
PROTOCOL_FIELDS = {
    "schema_version", "batch", "split", "phase", "created_at", "git_head", "root_binding_sha256",
    "artifacts", "dependencies", "model", "base_url", "messages_url", "limits", "decision_policy",
    "stop_policy", "operations", "cases_sha256", "archived_evidence_proof",
    "authorized_model_requests", "proposed_model_requests",
}
OLD_MANIFEST = bench.BACKEND_ROOT / "data/benchmarks/preparation-20261101-v1/manifest.json"
OLD_MANIFEST_HASH = "807b503bdee8ce00925ff2627e2570d5442b241b7b7eceaebb322fc57558ef3f"


class _ArchivedReadOnly:
    """Keep historical constructors on the non-creating lock path, even in a race."""

    def _locked(self, create=False):
        return super()._locked(create=False)

    def reserve(self, *_args, **_kwargs):
        raise TrialLedgerError("protocol")

    def finish(self, *_args, **_kwargs):
        raise TrialLedgerError("protocol")


class _ArchivedDevelopmentLedger(_ArchivedReadOnly, TrialLedger):
    pass


class _ArchivedValidationLedger(_ArchivedReadOnly, ValidationLedger):
    pass


OLD_BATCHES = (
    ("intent-development-v1", 24, _ArchivedDevelopmentLedger,
     "4e09d597d8ec5c9480108af5ce8c37e7504eb5f293778464c3f55d97feead14f",
     "49375536c1748f4c4841b3d176501dd780d848e785c2b2a08308bd9bfe39656d"),
    ("intent-validation-v1", 36, _ArchivedValidationLedger,
     "858e262c28e406bf17e40782af1c0098220252683ecefe4e8b9ddd5d8083c5bd",
     "1c3f29e8b7570f7c08843677bf3ba769972b7797ed414b0c728284544df75e2d"),
)
OLD_AUXILIARY_HASHES = {
    "intent-development-v1": {
        "authorization": "863d2770c9c18d13d8140379418fe4e1e68e922ecbf4217f5b1cd9b4c49f12d3",
        "report": "88bd3f53b4d7449e7e74f2f5211c3befaac0d91cdb3207e16cb0c8df8c77bbe2"},
    "intent-validation-v1": {
        "authorization": "a7d273c74f0e08bc92a0e0f4893e4075edca864f38184c32512a5b05c6530530",
        "report": "cd0f84b2069ee92f40ae206a2f90999dd0062c835caa50f9e352494a1e5cfbf2"},
}


def _configuration(*, secret=False):
    return base._configuration(secret=secret)


def cases():
    value = bench.read_json(CASES_PATH)
    if (type(value) is not dict or set(value) != {"schema_version", "dataset_id", "provenance", "cases"}
            or value["schema_version"] != "serveflow-intent-recheck-cases-v2"
            or value["dataset_id"] != "intent-recheck-v2"
            or value["provenance"] != "synthetic-authored-not-independent-blind-test"
            or type(value["cases"]) is not list or len(value["cases"]) != MAX_REQUESTS):
        engine._fail("invalid_recheck_cases")
    old = {case["id"]: case for case in bench.load_suites()["intent"]["cases"]}
    previous_inputs = {canonical_payload_hash(bench.model_input(case)) for case in old.values()}
    seen_inputs = set()
    for index, case in enumerate(value["cases"]):
        expected_id = ("i24", "i43")[index] if index < 2 else f"n{index - 1:02d}"
        partition = "seen_regression" if index < 2 else "new_challenge"
        if (type(case) is not dict or set(case) != {"id", "partition", "input", "expected", "tags"}
                or case["id"] != expected_id or case["partition"] != partition
                or type(case["input"]) is not dict or set(case["input"]) != {"message", "history"}
                or type(case["expected"]) is not dict or set(case["expected"]) != {"intent", "rationale"}
                or case["expected"]["intent"] not in bench.BENCHMARK_INTENTS):
            engine._fail("invalid_recheck_case")
        bench._text(case["input"]["message"], 2000)
        history = case["input"]["history"]
        if type(history) is not list or len(history) > 6:
            engine._fail("invalid_recheck_history")
        for item in history:
            if type(item) is not dict or set(item) != {"role", "content"} or item["role"] not in {"user", "assistant"}:
                engine._fail("invalid_recheck_history")
            bench._text(item["content"], 2000)
        bench._text(case["expected"]["rationale"], 2000)
        bench._strings(case["tags"], allow_empty=False)
        identity = canonical_payload_hash(bench.model_input(case))
        if identity in seen_inputs or (index >= 2 and identity in previous_inputs):
            engine._fail("duplicate_recheck_input")
        if index < 2 and (case["input"] != old[expected_id]["input"] or case["expected"] != old[expected_id]["expected"]):
            engine._fail("seen_regression_changed")
        seen_inputs.add(identity)
    return value["cases"]


def _archived_sources(record, field):
    hashes = record[field]
    if type(hashes) is not dict or not hashes:
        engine._fail("archived_sources_invalid")
    for name, expected in hashes.items():
        if type(name) is not str or Path(name).is_absolute() or ".." in Path(name).parts:
            engine._fail("archived_sources_invalid")
        blob = subprocess.run(["git", "show", record["git_head"] + ":backend/" + name],
                              cwd=bench.BACKEND_ROOT, capture_output=True, check=True, timeout=5).stdout
        import hashlib
        if hashlib.sha256(blob).hexdigest() != expected:
            engine._fail("archived_sources_changed")


def archived_evidence_proof():
    """Read only the two known, fully consumed batches and original manifest.

    This intentionally does not call their current-code verify/report methods:
    the current recognizer has changed. No historical file is initialized,
    repaired, rescored or overwritten, and no old allowance is imported.
    """
    manifest_path = base._path(OLD_MANIFEST.parent, OLD_MANIFEST.name)
    if bench.sha256(manifest_path) != OLD_MANIFEST_HASH:
        engine._fail("archived_manifest_changed")
    manifest = bench.read_json(manifest_path)
    _archived_sources(manifest, "artifacts")
    result = {"offline_manifest_sha256": OLD_MANIFEST_HASH, "batches": []}
    for name, count, ledger_type, protocol_hash, journal_hash in OLD_BATCHES:
        root = bench.BACKEND_ROOT / "data/benchmarks" / name
        # Refuse missing state BEFORE invoking a constructor that can initialize.
        for filename in ("ledger/identity.json", "ledger/journal.jsonl", "authorization.json", "protocol.json"):
            if not base._path(root, filename).is_file():
                engine._fail("archived_evidence_missing")
        if not (root / "ledger/claims").is_dir() or (root / "stopped.json").exists():
            engine._fail("archived_evidence_not_sealed")
        snapshot = engine._ledger_snapshot(root)
        if bench.sha256(root / "protocol.json") != protocol_hash or bench.sha256(root / "ledger/journal.jsonl") != journal_hash:
            engine._fail("archived_evidence_changed")
        approval_hash = bench.sha256(root / "authorization.json")
        if approval_hash != OLD_AUXILIARY_HASHES[name]["authorization"]:
            engine._fail("archived_authorization_changed")
        protocol = bench.read_json(root / "protocol.json")
        _archived_sources(protocol, "extension_artifacts")
        ledger = ledger_type(root / "ledger", protocol_hash,
                             [op["operation_id"] for op in protocol["operations"]], count)
        state, sealed = ledger.status(), ledger.results()
        if (state["attempts"] != count or state["completed"] != [op["operation_id"] for op in protocol["operations"]]
                or state["pending"] or state["paused"] or len(sealed) != count):
            engine._fail("archived_evidence_not_sealed")
        approval = bench.read_json(root / "authorization.json")
        if (approval["protocol_sha256"] != protocol_hash or approval["max_requests"] != count
                or approval["root_binding_sha256"] != base._root_hash(root)
                or approval["initial_ledger_chain_hash"] != ledger.initial_chain_hash):
            engine._fail("archived_authorization_changed")
        observations = {}
        for op in protocol["operations"]:
            filename = op["case_id"] + "-observation.json"
            digest = bench.sha256(base._path(root, filename))
            if sealed[op["operation_id"]]["output_sha256"] != digest:
                engine._fail("archived_observation_changed")
            observations[op["case_id"]] = digest
        report = base._path(root, "report-" + state["chain_hash"] + ".json")
        if not report.is_file():
            engine._fail("archived_report_missing")
        report_hash = bench.sha256(report)
        if report_hash != OLD_AUXILIARY_HASHES[name]["report"]:
            engine._fail("archived_report_changed")
        if (snapshot != engine._ledger_snapshot(root)
                or bench.sha256(root / "protocol.json") != protocol_hash
                or bench.sha256(root / "authorization.json") != approval_hash
                or bench.sha256(report) != report_hash
                or any(bench.sha256(base._path(root, key + "-observation.json")) != digest
                       for key, digest in observations.items())):
            engine._fail("archived_evidence_changed_during_read")
        result["batches"].append({"batch": name, "protocol_sha256": protocol_hash,
            "authorization_sha256": approval_hash,
            "ledger_sha256": snapshot, "ledger_chain_hash": state["chain_hash"],
            "observation_sha256": observations, "report_sha256": report_hash})
    if bench.sha256(manifest_path) != OLD_MANIFEST_HASH:
        engine._fail("archived_evidence_changed_during_read")
    return result


def _artifacts():
    return {**bench.freeze()["artifacts"],
            **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in EXTRA_ARTIFACTS}}


class RecheckTrial(engine.ValidationTrial):
    batch = BATCH
    max_requests = MAX_REQUESTS
    split = "post-fix-recheck"
    authorization_schema = "serveflow-intent-recheck-authorization-v2"
    observation_schema = "serveflow-intent-recheck-observation-v2"
    report_schema = "serveflow-intent-recheck-report-v2"
    stop_schema = "serveflow-intent-recheck-stop-v2"
    report_limitations = [
        "2 seen regressions and 22 authored challenges are reported separately",
        "new wording is not independent human blind evaluation or guaranteed semantic independence",
        "shared LLM observation ablation; no new remote embedding/RAG/judge/memory calls",
        "single measurement; changed prompt prevents cross-version causal improvement claims",
        "all planned cases remain in denominators; shared resources counted once"]

    def __init__(self, root=ROOT):
        super().__init__(root)
        for name, *_ in OLD_BATCHES:
            previous = (bench.BACKEND_ROOT / "data/benchmarks" / name).resolve()
            current = self.root.resolve()
            if current.is_relative_to(previous) or previous.is_relative_to(current):
                engine._fail("archived_root_forbidden")

    def _cases(self):
        return cases()

    def _configuration(self, *, secret=False):
        return _configuration(secret=secret)

    def _ledger(self, protocol):
        return TrialLedger(self.root / "ledger", bench.sha256(base._path(self.root, "protocol.json")),
                           [op["operation_id"] for op in protocol["operations"]], MAX_REQUESTS)

    async def prepare(self):
        if base._path(self.root, "protocol.json").exists() or (self.root / "ledger").exists():
            engine._fail("trial_already_prepared")
        proof, config = archived_evidence_proof(), self._configuration()
        operations = []
        for case in cases():
            operation = await base._capture(case, config["base_url"])
            operation["operation_id"] = BATCH + ":" + case["id"] + ":observation-1"
            operations.append(operation)
        protocol = {"schema_version": "serveflow-intent-recheck-v2", "batch": BATCH, "split": self.split,
            "phase": "awaiting-explicit-authorization", "created_at": base._now(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip(),
            "root_binding_sha256": base._root_hash(self.root), "artifacts": _artifacts(),
            "dependencies": base._dependencies(), "model": base.MODEL,
            "base_url": validated_endpoint(config["base_url"]), "messages_url": messages_url(config["base_url"]),
            "limits": deepcopy(LIMITS), "decision_policy": deepcopy(base.DECISIONS),
            "stop_policy": deepcopy(base.STOP_POLICY), "operations": operations,
            "cases_sha256": bench.sha256(CASES_PATH), "archived_evidence_proof": proof,
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        base._write(self.root, "protocol.json", protocol)
        if self._ledger(protocol).status()["attempts"]:
            engine._fail("unexpected_existing_attempts")
        return protocol

    def verify(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        if type(p) is not dict or set(p) != PROTOCOL_FIELDS:
            engine._fail("invalid_protocol")
        if (p["schema_version"] != "serveflow-intent-recheck-v2" or p["batch"] != BATCH or p["split"] != self.split
                or p["phase"] != "awaiting-explicit-authorization" or p["model"] != base.MODEL
                or p["limits"] != LIMITS or p["decision_policy"] != base.DECISIONS or p["stop_policy"] != base.STOP_POLICY
                or type(p["authorized_model_requests"]) is not int or p["authorized_model_requests"] != 0
                or type(p["proposed_model_requests"]) is not int or p["proposed_model_requests"] != MAX_REQUESTS
                or p["root_binding_sha256"] != base._root_hash(self.root)
                or p["dependencies"] != base._dependencies()):
            engine._fail("protocol_version")
        config = self._configuration()
        if p["base_url"] != validated_endpoint(config["base_url"]) or p["messages_url"] != messages_url(config["base_url"]):
            engine._fail("configuration")
        if p["artifacts"] != _artifacts() or p["cases_sha256"] != bench.sha256(CASES_PATH):
            engine._fail("frozen_artifacts_changed")
        if p["archived_evidence_proof"] != archived_evidence_proof():
            engine._fail("archived_evidence_changed")
        if type(p["operations"]) is not list or len(p["operations"]) != MAX_REQUESTS:
            engine._fail("operation_count")
        for op, case in zip(p["operations"], cases()):
            if (type(op) is not dict or set(op) != {"case_id", "operation_id", "input_sha256", "payload", "payload_sha256", "request_bytes_sha256"}
                    or op["case_id"] != case["id"] or op["operation_id"] != BATCH + ":" + case["id"] + ":observation-1"
                    or op["input_sha256"] != canonical_payload_hash(bench.model_input(case))
                    or op["payload_sha256"] != canonical_payload_hash(op["payload"])
                    or type(op["request_bytes_sha256"]) is not str or not engine.SHA256.fullmatch(op["request_bytes_sha256"])):
                engine._fail("operation_binding")
        for name in ("ledger/identity.json", "ledger/journal.jsonl"):
            if not base._path(self.root, name).is_file():
                engine._fail("ledger_missing")
        self._ledger(p).status()
        return p

    def report(self):
        from evaluation.benchmark_metrics import score_intent
        result = super().report()
        grouped = {}
        for partition in ("seen_regression", "new_challenge"):
            subset = [case for case in self._cases() if case["partition"] == partition]
            ids = {case["id"] for case in subset}
            grouped[partition] = {"classification_scores": {},
                "paired_cases": [row for row in result["paired_cases"] if row["case_id"] in ids]}
            for arm in ("llm_only", "fusion"):
                rows = [engine._record(row["case_id"], row["status"], predicted=row["predicted_intent"], arm=True)
                        for row in result["scores"][arm]["per_case"]
                        if row["case_id"] in ids and row["status"] != "not_executed"]
                grouped[partition]["classification_scores"][arm] = score_intent(subset, rows)
        result["partitions"] = grouped
        if self._ledger(self.verify()).status()["chain_hash"] != result["ledger"]["chain_hash"]:
            engine._fail("ledger_changed_during_report")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "status", "report", "run"):
        commands.add_parser(name)
    approval = commands.add_parser("authorize")
    approval.add_argument("--reference", required=True)
    approval.add_argument("--confirm-user-authorization", action="store_true")
    args = parser.parse_args(argv)
    trial = RecheckTrial()
    try:
        if args.command == "prepare":
            asyncio.run(trial.prepare())
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
            if not base._path(trial.root, name).exists():
                base._write(trial.root, name, report)
            elif bench.read_json(base._path(trial.root, name)) != report:
                engine._fail("saved_report_changed")
            value = {"attempts": report["ledger"]["attempts"], "paused": report["ledger"]["paused"],
                     "evidence_kind": report["evidence_kind"], "paired_counts": report["paired_counts"]}
        print(json.dumps(value, ensure_ascii=False))
        return 0
    except Exception as error:
        from evaluation.intent_transport import TrialTransportError
        from evaluation.trial_ledger import TrialLedgerError
        code = error.code if isinstance(error, (TrialTransportError, TrialLedgerError)) else str(error) if isinstance(error, base.TrialError) else "recheck_input_or_storage_failure"
        print(json.dumps({"error": code, "no_automatic_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
