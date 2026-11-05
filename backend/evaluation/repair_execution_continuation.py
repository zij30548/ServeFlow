"""One explicit continuation of the original zero-send repair batch.

The old protocol, authorization, stop and ledger identity stay immutable. This
entry point admits only the recorded pre-case-start pause, on the same ledger
and original 96-request allowance. It never resets or resumes an attempted job.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, benchmark_metrics as metrics
from evaluation import repair_execution_trial as old
from evaluation import intent_trial as base, task_data, task_trial as engine, task_recheck as history
from evaluation import diagnostic_repair_proof as ancestry
from evaluation.intent_transport import canonical_payload_hash, messages_url, TrialTransportError
from evaluation.phase_deadline_transport import SDK_TIMEOUT_S, PHASE_HTTP_LIMITS
from evaluation.prospective_trial import STOP_FILES, _binding
from evaluation.prospective_repaired_proof import _git_blobs
from evaluation.repair_execution_budget import ROOT, MAX_REQUESTS, LIMITS, fixed_plan, job_specs
from evaluation.repair_execution_runtime import rows, execute_task, run_retrieval
from evaluation.task_transport import PHASE_PARAMETERS, AGENT_PARAMETERS
from evaluation.trial_ledger import TrialLedgerError, ERROR_CODES, LEDGER_ERROR_CODES


ORIGINAL_PROTOCOL_SHA256 = "1e11e64c66b006c402b1a46c6e6bfaa4fed703f3f61a5119225fee8bb31224d8"
ORIGINAL_SOURCE_COMMIT = "683c33d2d6f876a0a8a0ee4f1620cbcfdeb3d11b"
SUPERVISOR_HOOK_SHA256 = "e7f8797464fbf4877cd83f4d7271b50bf89ab8587e096a93f2de24dc5d4ee066"
SUPERVISOR_PATH = "evaluation/task_supervisor.py"
ADDITIONS = ("evaluation/repair_execution_continuation.py", "tests/test_repair_execution_continuation.py")
CONTINUATION_FILE = "continuation.json"
STARTED_FILE = "continuation-started.json"
STOP_FILE = "continuation-stop.json"
CONTINUATION_SCHEMA = "serveflow-repair-execution-continuation-v1"
ZERO_ROOT_FILES = {"protocol.json", "authorization.json", "worker-stop.json"}
DATA_ERROR_CODES = frozenset({
    "evaluation_fixture_changed", "evaluation_fixture_invalid", "evaluation_corpus_changed",
    "evaluation_corpus_invalid", "evaluation_mysql_url_required", "evaluation_database_name_required",
    "evaluation_literal_schema_grants_required", "evaluation_select_only_account_required",
    "evaluation_mysql_preflight_failed", "evaluation_database_fixture_mismatch",
    "evaluation_embedding_cache_unavailable", "evaluation_embedding_cache_changed",
    "evaluation_chroma_version_unsupported", "evaluation_embedding_dependency_unavailable",
    "evaluation_corpus_contract_changed", "evaluation_embedding_download_forbidden",
    "evaluation_corpus_index_mismatch", "evaluation_knowledge_preflight_failed",
    "evaluation_data_closed", "evaluation_data_read_failed", "evaluation_data_cleanup_failed",
    "evaluation_fixture_identity_invalid",
})


def _safe_error(error):
    """Allowlisted diagnostics only; never exception text, SQL or credentials."""
    if isinstance(error, task_data.TaskDataError):
        return "task_data", error.code if type(error.code) is str and error.code in DATA_ERROR_CODES else "task_data_error"
    if isinstance(error, TrialTransportError):
        return "transport", error.code if type(error.code) is str and error.code in ERROR_CODES else "transport_error"
    if isinstance(error, TrialLedgerError):
        return "ledger", error.code if type(error.code) is str and error.code in LEDGER_ERROR_CODES else "ledger_error"
    if isinstance(error, base.TrialError):
        code = error.args[0] if len(error.args) == 1 else None
        return "trial", code if type(code) is str and code in {"configuration", "unsafe_trial_path", "unsafe_trial_permissions",
            "unsafe_trial_file", "repair_continuation_context_previously_failed",
            "repair_continuation_run_identity_changed", "repair_continuation_runtime_context_changed",
            "repair_continuation_controlled_data_required", "repair_continuation_case_status"} else "trial_error"
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
        return "cancelled", "cancelled"
    if isinstance(error, SystemExit):
        return "process_exit", "process_exit"
    return "unexpected_error", "unknown"


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.BACKEND_ROOT,
                                   text=True, timeout=15).strip()


def source_proof(p):
    """Verify all 233 original blobs and a closed two-file/one-hook extension.

    The original runner remains frozen and continues rejecting changed source.
    This independent proof has no ignore-version mode or alternate source root.
    """
    original, current = p["artifacts"], ancestry._current_sources()
    if (type(original) is not dict or len(original) != 233
            or set(current) != set(original) | set(ADDITIONS)
            or set(original) & set(ADDITIONS) or SUPERVISOR_PATH not in original):
        base._fail("repair_continuation_source_closure_changed")
    names, blobs = ["backend/" + name for name in sorted(original)], {}
    for start in range(0, len(names), 256):
        blobs.update(_git_blobs(ORIGINAL_SOURCE_COMMIT, names[start:start+256]))
    historical = {name.removeprefix("backend/"): hashlib.sha256(raw).hexdigest()
                  for name, raw in blobs.items()}
    if historical != original:
        base._fail("repair_continuation_original_source_changed")
    approved = {**original, SUPERVISOR_PATH: SUPERVISOR_HOOK_SHA256}
    if any(current[name] != digest for name, digest in approved.items()):
        base._fail("repair_continuation_business_or_hook_changed")
    return {"original_source_commit": ORIGINAL_SOURCE_COMMIT,
            "original_artifacts_sha256": canonical_payload_hash(original),
            "current_artifacts": current, "exact_supervisor_hook_sha256": SUPERVISOR_HOOK_SHA256,
            "closed_additions": list(ADDITIONS)}


def _documentation_head_changed_only(c):
    """Exact Git evidence for later README-only publication, never execution."""
    frozen_head = c["execution_git_head"]
    if type(frozen_head) is not str or re.fullmatch(r"[a-f0-9]{40}", frozen_head) is None:
        return False
    result = subprocess.run(["git", "diff", "--name-only", "--no-renames", frozen_head, _git_head(), "--"],
        cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15)
    if len(result.stdout) > 4096 or set(result.stdout.decode().splitlines()) != {"README.md"}:
        return False
    names = ["backend/" + name for name in sorted(c["current_artifacts"])]
    blobs = {}
    for start in range(0, len(names), 256):
        blobs.update(_git_blobs(frozen_head, names[start:start+256]))
    return {name.removeprefix("backend/"): hashlib.sha256(raw).hexdigest()
            for name, raw in blobs.items()} == c["current_artifacts"]


def _storage_identity(path, *, directory=False):
    info = path.lstat()
    if (path.resolve() != path or path.is_symlink() or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)
            or (not stat.S_ISDIR(info.st_mode) if directory else
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
        base._fail("repair_continuation_unsafe_storage")
    return {"device": info.st_dev, "inode": info.st_ino}


class ContinuationTrial(old.RepairExecutionTrial):
    """Same ROOT, grant, native runtime and accounting; one new startup only."""

    def prepare(self):
        base._fail("repair_continuation_original_preparation_required")

    def authorize(self, *args, **kwargs):
        base._fail("repair_continuation_original_authorization_only")

    def _protocol(self):
        p = bench.read_json(base._path(self.root, "protocol.json"))
        if self.root == ROOT and bench.sha256(self.root / "protocol.json") != ORIGINAL_PROTOCOL_SHA256:
            base._fail("repair_continuation_original_protocol_changed")
        preparation = old.frozen_preparation()
        config = base._configuration()  # Public settings only.
        expected = {"schema_version": self.protocol_schema, "root_binding_sha256": base._root_hash(self.root),
            "preparation_sha256": old.PREPARATION_SHA256, "dataset_sha256": old.DATASET_SHA256,
            "source_proof": p["source_proof"], "artifacts": p["artifacts"], "prior_proof": old.prior_proof(),
            "cases": rows(), "plan": fixed_plan(), "jobs": job_specs(), "model": base.MODEL,
            "base_url": config["base_url"], "messages_url": messages_url(config["base_url"]),
            "dependencies": engine._dependencies(), "effective_environment": history.effective_environment(),
            "offline_data_proof": task_data.offline_data_proof(), "limits": LIMITS,
            "http_limits": PHASE_HTTP_LIMITS, "phase_parameters": {k: v for k, v in PHASE_PARAMETERS.items() if k != "judge"},
            "agent_parameters": AGENT_PARAMETERS, "policy": old.POLICY, "rubric": preparation["rubric"],
            "authorized_model_requests": 0, "proposed_model_requests": MAX_REQUESTS,
            "judge_requests": 0, "external_business_requests": 0, **old.initial_requests(preparation)}
        expected = json.loads(json.dumps(expected))
        bench._exact(p, set(expected) | {"created_at", "git_head"}, "repair_continuation_original_protocol_fields")
        if (any(p[k] != value for k, value in expected.items())
                or config != {"base_url": preparation["base_url"], "model": preparation["model"]}
                or p["dependencies"] != preparation["dependencies"]
                or p["effective_environment"] != preparation["effective_environment"]
                or p["offline_data_proof"] != preparation["offline_data_proof"]):
            base._fail("repair_continuation_original_context_changed")
        for first in p["initial_intent_requests"] + p["initial_rewrite_requests"]:
            row = next(v for v in p["cases"] if v["id"] == first["case_id"])
            raw = json.dumps(first["payload"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if (first["source_case_id"] != row["case"]["id"]
                    or first["input_sha256"] != canonical_payload_hash(bench.model_input(row["case"]))
                    or first["payload_sha256"] != canonical_payload_hash(first["payload"])
                    or first["request_bytes_sha256"] != hashlib.sha256(raw).hexdigest()):
                base._fail("repair_continuation_first_request_changed")
        return p

    def _zero_start(self, p, *, preparing=False):
        state = self.ledger(p).status()
        if (state["attempts"] or state["active_case"] or state["completed_cases"] or state["pending"]
                or state["paused"] or state["chain_hash"] != self.ledger(p).initial_chain_hash
                or (self.root / "ledger/journal.jsonl").read_bytes()
                or any((self.root / "ledger/claims").iterdir())
                or {path.name for path in (self.root / "ledger").iterdir()}
                    != {"identity.json", "journal.jsonl", "run.lock", "claims"}):
            base._fail("repair_continuation_requires_zero_start")
        allowed = ZERO_ROOT_FILES | (set() if preparing else {CONTINUATION_FILE})
        if {path.name for path in self.root.iterdir() if path.name != "ledger"} != allowed:
            base._fail("repair_continuation_previously_started_or_stopped")
        stop = bench.read_json(base._path(self.root, "worker-stop.json"))
        bench._exact(stop, {"reason", "recorded_at"}, "repair_continuation_old_stop_fields")
        if stop["reason"] != "execution_stopped_no_auto_resume":
            base._fail("repair_continuation_old_stop_not_zero_start")
        return state

    def _origin(self, p):
        ledger = self.root / "ledger"
        journal = base._path(ledger, "journal.jsonl")
        return {"protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")),
            "authorization_sha256": bench.sha256(base._path(self.root, "authorization.json")),
            "historical_worker_stop_sha256": bench.sha256(base._path(self.root, "worker-stop.json")),
            "ledger_identity_sha256": bench.sha256(base._path(ledger, "identity.json")),
            "ledger_run_lock_sha256": bench.sha256(base._path(ledger, "run.lock")),
            "journal_prefix_bytes": journal.stat().st_size,
            "journal_prefix_sha256": bench.sha256(journal),
            "initial_chain_hash": self.ledger(p).initial_chain_hash,
            "initial_claims": [], "storage": {
                "ledger": _storage_identity(ledger, directory=True),
                "identity": _storage_identity(ledger / "identity.json"),
                "journal": _storage_identity(journal),
                "run_lock": _storage_identity(ledger / "run.lock"),
                "claims": _storage_identity(ledger / "claims", directory=True)}}

    async def prepare_continuation(self, reference, *, confirmed=False):
        if (confirmed is not True or type(reference) is not str or not reference.strip() or len(reference) > 500):
            base._fail("repair_continuation_explicit_confirmation_required")
        if (self.root / CONTINUATION_FILE).exists():
            base._fail("repair_continuation_already_prepared")
        p = self._protocol()
        approval = self._approval(p)
        if approval["max_requests"] != MAX_REQUESTS:
            base._fail("repair_continuation_original_96_required")
        proof = source_proof(p)
        # Re-capture locally with the existing fake capture credential. This
        # never obtains a secret or constructs a real HTTP transport.
        preparation = old.frozen_preparation()
        if await old.preparation.capture_requests(preparation) != preparation["sdk_first_captures"]:
            base._fail("repair_continuation_initial_capture_changed")
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        with ledger.exclusive_run():
            self._zero_start(p, preparing=True)
            if self._protocol() != p or source_proof(p) != proof or self._approval(p) != approval:
                base._fail("repair_continuation_changed_during_preparation")
            origin = self._origin(p)
            value = {"schema_version": CONTINUATION_SCHEMA, "recorded_at": base._now(), "reference": reference,
                "root_binding_sha256": base._root_hash(self.root), "execution_git_head": _git_head(),
                "original_source_commit": ORIGINAL_SOURCE_COMMIT, "original_authorized_requests": MAX_REQUESTS,
                "additional_authorized_requests": 0, "maximum_requests": MAX_REQUESTS,
                "source_proof": proof, "current_artifacts": proof["current_artifacts"], "zero_start_binding": origin,
                "policy": "one startup on the original empty ledger and 96 grant; no retry, restart, reset or allowance extension"}
            base._write(self.root, CONTINUATION_FILE, value)
        self.verify()
        return value

    def _continuation(self, p, *, strict_head=False):
        c = bench.read_json(base._path(self.root, CONTINUATION_FILE))
        bench._exact(c, {"schema_version", "recorded_at", "reference", "root_binding_sha256", "execution_git_head",
            "original_source_commit", "original_authorized_requests", "additional_authorized_requests", "maximum_requests",
            "source_proof", "current_artifacts", "zero_start_binding", "policy"}, "repair_continuation_fields")
        proof, approval = source_proof(p), self._approval(p)
        if (c["schema_version"] != CONTINUATION_SCHEMA or c["root_binding_sha256"] != base._root_hash(self.root)
                or c["original_source_commit"] != ORIGINAL_SOURCE_COMMIT
                or c["original_authorized_requests"] != MAX_REQUESTS or c["additional_authorized_requests"] != 0
                or c["maximum_requests"] != MAX_REQUESTS or approval["max_requests"] != MAX_REQUESTS
                or c["source_proof"] != proof or c["current_artifacts"] != proof["current_artifacts"]):
            base._fail("repair_continuation_binding_changed")
        if c["execution_git_head"] != _git_head():
            if strict_head or not (self.root / STARTED_FILE).exists() or not (self.root / "cleanup-complete.json").exists():
                base._fail("repair_continuation_execution_head_changed")
            state = self.ledger(p).status()
            if state["active_case"] or state["pending"]:
                base._fail("repair_continuation_execution_head_changed")
            # A running worker owns this exact lease, even in a gap between
            # positions. This compatibility applies only after resource close.
            with self.ledger(p).exclusive_run():
                if not _documentation_head_changed_only(c):
                    base._fail("repair_continuation_execution_head_changed")
        origin, current = c["zero_start_binding"], self._origin(p)
        # Only the journal grows. Replay below checks all later claims and
        # events; the original identity, run-lock and directory inodes stay.
        expected = {**current, "journal_prefix_bytes": 0,
                    "journal_prefix_sha256": hashlib.sha256(b"").hexdigest()}
        if origin != expected or current["journal_prefix_bytes"] < origin["journal_prefix_bytes"]:
            base._fail("repair_continuation_original_ledger_or_stop_changed")
        started_path = self.root / STARTED_FILE
        if started_path.exists():
            started = bench.read_json(base._path(self.root, STARTED_FILE))
            bench._exact(started, {"schema_version", "recorded_at", "continuation_sha256", "protocol_sha256",
                "authorization_sha256", "initial_chain_hash"}, "repair_continuation_started_fields")
            if any(started[key] != value for key, value in self._started_binding(p).items()):
                base._fail("repair_continuation_started_changed")
        else:
            self._zero_start(p)
        return c

    def _started_binding(self, p):
        return {"schema_version": "serveflow-repair-execution-continuation-started-v1",
            "continuation_sha256": bench.sha256(base._path(self.root, CONTINUATION_FILE)),
            "protocol_sha256": bench.sha256(base._path(self.root, "protocol.json")),
            "authorization_sha256": bench.sha256(base._path(self.root, "authorization.json")),
            "initial_chain_hash": self.ledger(p).initial_chain_hash}

    def _stopped(self):
        # Only worker-stop.json is historical, and _continuation checks its
        # exact bound hash. Every other old or new stop remains effective.
        return any((self.root / name).exists() for name in (*STOP_FILES, STOP_FILE) if name != "worker-stop.json")

    def _can_start(self, p):
        if (self.root / STARTED_FILE).exists():
            base._fail("repair_continuation_already_started_no_retry")
        if self._stopped():
            base._fail("repair_continuation_paused")
        self._zero_start(p)

    def verify(self, *, strict_head=False):
        p = self._protocol()
        self._continuation(p, strict_head=strict_head)
        self.ledger(p).status()
        sealed = self._sealed(p)
        for key in sealed["cases"]:
            artifact = bench.read_json(self.root / f"case-{key}.json")
            row = next(v for v in p["cases"] if v["id"] == key)
            if artifact.get("trial_kind") != old.KIND or artifact.get("shared_checks") != old.artifact_checks(row, artifact):
                base._fail("repair_continuation_artifact_changed")
        return p

    def validate(self):
        """Read-only admission audit, including before a continuation exists."""
        prepared = (self.root / CONTINUATION_FILE).exists()
        if prepared:
            p = self.verify()
            proof = self._continuation(p)["source_proof"]
            state = self.ledger(p).status()
        else:
            p = self._protocol()
            approval = self._approval(p)
            if approval["max_requests"] != MAX_REQUESTS:
                base._fail("repair_continuation_original_96_required")
            with self.ledger(p).exclusive_run():
                state = self._zero_start(p, preparing=True)
                proof = source_proof(p)
                self._origin(p)
        return {"schema_version": "serveflow-repair-execution-continuation-validation-v1",
            "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "authorization_sha256": bench.sha256(self.root / "authorization.json"),
            "historical_worker_stop_sha256": bench.sha256(self.root / "worker-stop.json"),
            "continuation_prepared": prepared, "continuation_started": (self.root / STARTED_FILE).exists(),
            "original_authorized_requests": MAX_REQUESTS, "additional_authorized_requests": 0,
            "reserved_requests": state["attempts"], "current_source_files": len(proof["current_artifacts"]),
            "zero_start": not state["attempts"] and not state["active_case"] and not state["completed_cases"],
            "paused": state["paused"] or self._stopped(), "quality_pass_claim": False}

    def inspect_zero_start(self):
        if (self.root / CONTINUATION_FILE).exists():
            base._fail("repair_continuation_already_prepared")
        return self.validate()

    def _write_stop(self, reason, error):
        if not (self.root / STOP_FILE).exists():
            error_kind, error_code = _safe_error(error)
            base._write(self.root, STOP_FILE, {"schema_version": "serveflow-repair-execution-continuation-stop-v1",
                "reason": reason, "recorded_at": base._now(),
                "error_kind": error_kind, "error_code": error_code,
                "continuation_sha256": bench.sha256(base._path(self.root, CONTINUATION_FILE))})

    async def execute(self, *, delegate_factory=None, controlled_data=None):
        p = self.verify(strict_head=True)
        approval = self._approval(p)
        if (delegate_factory is None) != (controlled_data is None):
            base._fail("repair_continuation_controlled_pair_required")
        if delegate_factory is None and self.root != ROOT or delegate_factory is not None and self.root == ROOT:
            base._fail("repair_continuation_live_controlled_root_mismatch")
        self._can_start(p)
        ledger = self.ledger(p, send_limit=approval["max_requests"])
        data, sdk, http, transport = controlled_data, None, None, None
        with ledger.exclusive_run() as lease:
            # Re-check after the process lease, before the one-use marker and
            # before any secret extraction, SDK, data connection or resource.
            if self.verify(strict_head=True) != p:
                base._fail("repair_continuation_context_changed_before_start")
            self._can_start(p)
            try:
                base._write(self.root, STARTED_FILE, {**self._started_binding(p), "recorded_at": base._now()})
                config = base._configuration(secret=delegate_factory is None)
                if data is None:
                    data = await task_data.build_live_data(engine._mysql_url())
                if controlled_data is not None and data.proof.get("kind") != "controlled-fixture":
                    base._fail("repair_continuation_controlled_data_required")
                if controlled_data is None:
                    _binding(data.proof, p["offline_data_proof"])
                data.assert_healthy()
                base._write(self.root, "data-proof.json", data.proof)
                data_hash = bench.sha256(self.root / "data-proof.json")
                protocol_hash = bench.sha256(self.root / "protocol.json")
                continuation_hash = bench.sha256(self.root / CONTINUATION_FILE)
                authorization_hash = bench.sha256(self.root / "authorization.json")
                started_hash = bench.sha256(self.root / STARTED_FILE)
                data_canonical = canonical_payload_hash(data.proof)
                continuation = self._continuation(p, strict_head=True)
                current_artifacts = continuation["current_artifacts"]
                historical_stop_hash = continuation["zero_start_binding"]["historical_worker_stop_sha256"]
                execution_git_head = continuation["execution_git_head"]
                base._write(self.root, "data-proof-seal.json", {"protocol_sha256": protocol_hash,
                                                             "data_proof_sha256": data_hash})
                faults = []

                def healthy():
                    try:
                        if faults or transport is not None and transport.fault or self._stopped():
                            base._fail("repair_continuation_context_previously_failed")
                        self.verify(strict_head=True)
                        if (bench.sha256(self.root / "protocol.json") != protocol_hash
                                or bench.sha256(self.root / "authorization.json") != authorization_hash
                                or bench.sha256(self.root / CONTINUATION_FILE) != continuation_hash
                                or bench.sha256(self.root / STARTED_FILE) != started_hash
                                or bench.sha256(self.root / "data-proof.json") != data_hash
                                or canonical_payload_hash(data.proof) != data_canonical):
                            base._fail("repair_continuation_run_identity_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise

                def runtime_healthy():
                    try:
                        if faults or transport is not None and transport.fault or self._stopped():
                            base._fail("repair_continuation_context_previously_failed")
                        if (bench.sha256(base._path(self.root, "protocol.json")) != protocol_hash
                                or bench.sha256(base._path(self.root, "authorization.json")) != authorization_hash
                                or bench.sha256(base._path(self.root, CONTINUATION_FILE)) != continuation_hash
                                or bench.sha256(base._path(self.root, STARTED_FILE)) != started_hash
                                or bench.sha256(base._path(self.root, "worker-stop.json")) != historical_stop_hash
                                or _git_head() != execution_git_head
                                or bench.sha256(base._path(self.root, "data-proof.json")) != data_hash
                                or canonical_payload_hash(data.proof) != data_canonical
                                or ancestry._current_sources() != current_artifacts
                                or history.effective_environment() != p["effective_environment"]
                                or base._configuration() != {"base_url": p["base_url"], "model": p["model"]}):
                            base._fail("repair_continuation_runtime_context_changed")
                        data.assert_healthy()
                    except BaseException:
                        faults.append(True)
                        raise

                transport = old.RepairTransport(config["base_url"], self.root, ledger, lease, healthy,
                                                delegate_factory=delegate_factory)
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
                    artifact.update(logical_case_id=key, trial_kind=old.KIND, protocol_sha256=protocol_hash,
                                    data_proof_sha256=data_hash, evidence_kind=transport.kind)
                    artifact["shared_checks"] = old.artifact_checks(row, artifact)
                    base._write(self.root, f"case-{key}.json", artifact)
                    outcome = artifact.get("execution_status", artifact.get("status"))
                    if outcome not in {"success", "partial", "failed"}:
                        base._fail("repair_continuation_case_status")
                    ledger.finish_case(key, bench.sha256(self.root / f"case-{key}.json"), outcome=outcome, lease=lease)
            except BaseException as error:
                self._write_stop("execution_stopped_no_auto_resume", error)
                raise
            finally:
                try:
                    try:
                        if sdk is not None:
                            await task_data._owned(sdk.close())
                        elif http is not None:
                            await task_data._owned(http.aclose())
                    finally:
                        if data is not None:
                            await task_data._owned(data.close())
                    if data is not None:
                        base._write(self.root, "cleanup-complete.json", {"owned_resources_closed": True,
                            "protocol_sha256": bench.sha256(self.root / "protocol.json")})
                except BaseException as error:
                    self._write_stop("cleanup_failed_no_auto_resume", error)
                    raise
        return self.report()

    def run(self):
        if self.root != ROOT or Path.cwd() != bench.BACKEND_ROOT:
            base._fail("repair_continuation_fixed_worker_required")
        p = self.verify(strict_head=True)
        self._approval(p)
        self._can_start(p)
        from evaluation.task_supervisor import supervise
        return supervise([sys.executable, "-B", "-u", "-m", "evaluation.repair_execution_continuation", "_worker"],
                         root=ROOT, deadline_s=900)

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
                observed = old.artifact_checks(row, artifact)
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
                observed = old.artifact_checks(row, artifact)
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
                    and not self._stopped() and (self.root / "cleanup-complete.json").exists())
        reviewed = sum("review" in r for r in task_rows)
        evidence = dict(Counter(v["evidence_kind"] for v in artifacts.values()))
        retrieval_valid = all(r["status"] == "recorded" for r in retrieval_rows)
        authorized = self._approval(p)["max_requests"] if (self.root / "authorization.json").exists() else 0
        result = {"schema_version": "serveflow-repair-execution-progress-v1", "protocol_sha256": bench.sha256(self.root / "protocol.json"),
            "authorization_present": (self.root / "authorization.json").exists(), "reserved_requests": state["attempts"],
            "authorized_requests": authorized, "maximum_requests": MAX_REQUESTS, "planned_jobs": 16, "completed_jobs": len(artifacts),
            "not_executed_jobs": [r["id"] for r in p["cases"] if r["id"] not in artifacts and state["active_case"] != r["id"]],
            "unknown_jobs": [state["active_case"]] if state["active_case"] else [],
            "sealed_physical": sum(op["output_sha256"] is not None for op in sealed["operations"].values()),
            "pending_physical": len(state["pending"]), "paused": state["paused"] or self._stopped(),
            "task_rows": task_rows, "task_groups": grouped, "retrieval": retrieval, "retrieval_rows": retrieval_rows,
            "semantic_task_reviews": reviewed, "execution_complete": bool(complete),
            "review_ready": bool(complete and retrieval_valid), "comparison_ready": bool(complete and retrieval_valid and reviewed == 8),
            "real_comparison_ready": bool(complete and retrieval_valid and reviewed == 8 and evidence == {"live-model": 16}),
            "artifact_evidence_kinds": evidence, "physical_resources": metrics.aggregate_resources(resources),
            "resource_scope": "all reserved attempts once, including failed/unknown; no inferred tokens",
            "quality_pass_claim": False, "scope": old.POLICY["scope"]}

        result.update(continuation_sha256=bench.sha256(self.root / CONTINUATION_FILE),
                      continuation_started=(self.root / STARTED_FILE).exists(),
                      historical_worker_stop_sha256=bench.sha256(self.root / "worker-stop.json"),
                      additional_authorized_requests=0,
                      original_zero_start_continued=True)
        return result


RepairExecutionContinuation = ContinuationTrial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare-continuation", "verify", "report", "run", "_worker", "review"))
    parser.add_argument("--reference")
    parser.add_argument("--confirm-explicit-continuation", action="store_true")
    parser.add_argument("--review-file", type=Path)
    args, trial = parser.parse_args(argv), ContinuationTrial()
    try:
        if args.command == "validate":
            value = trial.validate()
        elif args.command == "prepare-continuation":
            value = asyncio.run(trial.prepare_continuation(args.reference, confirmed=args.confirm_explicit_continuation))
        elif args.command == "verify":
            value = trial.verify()
        elif args.command == "report":
            value = trial.report()
        elif args.command == "review":
            value = trial.review(bench.read_json(args.review_file))
        elif args.command == "run":
            value = trial.run()
            if value.get("exit_code") != 0 or value.get("timed_out") or value.get("cancelled"):
                base._fail("repair_continuation_supervised_worker_failed")
        else:
            value = asyncio.run(trial.execute())
        print(json.dumps(value if args.command in {"report", "validate"} else {"result": args.command,
            "reserved_requests": value.get("reserved_requests", 0),
            "additional_authorized_requests": 0}, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"error": "repair_continuation_stopped_inspect_private_evidence", "no_auto_retry": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
