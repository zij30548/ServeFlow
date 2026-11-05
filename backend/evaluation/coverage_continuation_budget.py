"""One explicitly bound continuation of the existing 20/480 coverage ledger.

This class never initializes, refunds, replaces or deletes a ledger.  Its two
additional journal events retain the original identity/genesis/hash domain and
only close an already-finished timeout/network case before the next fixed case.
All old stop receipts remain immutable.  The original CoverageBudget still
rejects the new event vocabulary instead of acquiring a generic resume switch.
"""
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from evaluation import task_budget as storage
from evaluation.trial_ledger import _pairs
from evaluation.coverage_budget import (
    CoverageBudget, ROOT, MAX_REQUESTS, CASE_IDS, fixed_plan, job_specs,
)


TaskBudgetError = storage.TaskBudgetError
SCHEMA = "serveflow-coverage-continuation-v1"
MANIFEST_NAME = "continuation-v1.json"
RUN_CLAIM_NAME = "continuation.run"
ORIGINAL_ATTEMPTS = 20
ORIGINAL_EVENTS = 53
ORIGINAL_CASE = "D-CT-D04"
RESUMABLE_ERRORS = frozenset({"timeout", "network"})
CONSECUTIVE_FAILURE_LIMIT = 3
_MANIFEST_LIMIT = 8 * 1024 * 1024
_NULL_FIELDS = ("turn_id", "phase", "agent", "ordinal", "output_sha256")


class ContinuationBudget(CoverageBudget):
    """Existing ledger, fixed suffix, immutable manifest and one run lease.

    Temporary roots outside the application's benchmark tree are usable only
    for offline tests.  The live runner separately requires exact ``ROOT``.
    """

    def __init__(self, root, protocol_sha256, case_plan, *, continuation_sha256,
                 prefix_size, prefix_sha256, prefix_chain_hash,
                 send_limit=MAX_REQUESTS):
        self.continuation_sha256 = storage._hash(continuation_sha256)
        self.prefix_sha256 = storage._hash(prefix_sha256)
        self.prefix_chain_hash = storage._hash(prefix_chain_hash)
        if (type(prefix_size) is not int or not 1 <= prefix_size <= storage._JOURNAL_LIMIT
                or case_plan != fixed_plan() or type(send_limit) is not int
                or not ORIGINAL_ATTEMPTS <= send_limit <= MAX_REQUESTS):
            raise TaskBudgetError("configuration")
        self.prefix_size = prefix_size
        self._original_protocol_sha256 = storage._hash(protocol_sha256)
        root = Path(root).absolute()
        benchmark_root = ROOT.parent
        if root.is_relative_to(benchmark_root) and root != ROOT / "ledger":
            raise TaskBudgetError("configuration")
        self.manifest_path = root.parent / MANIFEST_NAME
        super().__init__(root, protocol_sha256, case_plan,
                         initialize=False, send_limit=send_limit)

    def _manifest(self):
        fd = None
        try:
            path = self.manifest_path
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise TaskBudgetError("protocol")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            storage._private_file(fd)
            if os.fstat(fd).st_uid != os.getuid():
                raise TaskBudgetError("storage_failure")
            raw = storage._read(fd, _MANIFEST_LIMIT)
            if hashlib.sha256(raw).hexdigest() != self.continuation_sha256:
                raise TaskBudgetError("protocol")
            def finite(number):
                parsed = float(number)
                if not math.isfinite(parsed):
                    raise TaskBudgetError("protocol")
                return parsed

            def reject(_value):
                raise TaskBudgetError("protocol")

            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                               parse_float=finite, parse_constant=reject)
            expected = {
                "original_protocol_sha256": self._original_protocol_sha256,
                "prefix_size": self.prefix_size,
                "prefix_sha256": self.prefix_sha256,
                "prefix_chain_hash": self.prefix_chain_hash,
                "already_spent": ORIGINAL_ATTEMPTS,
                "max_requests": MAX_REQUESTS,
            }
            budget = value.get("budget") if type(value) is dict else None
            if (type(value) is not dict or value.get("schema_version") != SCHEMA
                    or type(budget) is not dict or set(budget) != set(expected)
                    or any(type(budget[k]) is not type(v) for k, v in expected.items())
                    or budget != expected
                    or value.get("previous_active_case") != ORIGINAL_CASE
                    or value.get("remaining_jobs") != job_specs()[7:]):
                raise TaskBudgetError("protocol")
            return value
        except TaskBudgetError:
            raise
        except (OSError, ValueError, TypeError):
            raise TaskBudgetError("storage_failure") from None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def _locked(self, create=False):
        if create:
            raise TaskBudgetError("configuration")
        with super()._locked(create=False) as handles:
            self._manifest()
            yield handles

    def _empty(self):
        return {**super()._empty(), "continuation_started": False,
                "consecutive_execution_failures": 0}

    def _replay(self, journal, claims):
        raw = storage._read(journal, storage._JOURNAL_LIMIT)
        prefix = raw[:self.prefix_size]
        if (len(prefix) != self.prefix_size or not prefix.endswith(b"\n")
                or len(prefix.splitlines()) != ORIGINAL_EVENTS
                or hashlib.sha256(prefix).hexdigest() != self.prefix_sha256):
            raise TaskBudgetError("protocol")
        os.lseek(journal, 0, os.SEEK_SET)
        state = super()._replay(journal, claims)
        # A claim without a journal event (crash during initial publication) is
        # retained and blocks another run. A resumed journal without its durable
        # one-shot claim must not authorize sends.
        claim = self._run_claim(read_only=True)
        if state["continuation_started"] and claim is None:
            raise TaskBudgetError("protocol")
        return state

    @staticmethod
    def _original_state(state):
        return (state["attempts"] == ORIGINAL_ATTEMPTS
                and state["seq"] == ORIGINAL_EVENTS
                and state["chain_hash"] is not None
                and state["completed_cases"] == list(CASE_IDS[:6])
                and state["starts"] == list(CASE_IDS[:7])
                and state["active_case"] == ORIGINAL_CASE
                and state["paused"] is True and not state["pending"]
                and len(state["operations"]) == ORIGINAL_ATTEMPTS
                and state["operations"].get(ORIGINAL_CASE + ":turn-1:intent:none:1", {}).get("error_code") == "timeout")

    def _apply(self, state, event):
        kind = event["event"]
        if event["seq"] <= ORIGINAL_EVENTS:
            if kind in {"resume_after_timeout", "case_execution_failure"}:
                raise TaskBudgetError("protocol")
            super()._apply(state, event)
            if event["seq"] == ORIGINAL_EVENTS:
                # seq/chain are published by the parent replay after _apply.
                observed = {**state, "seq": event["seq"], "chain_hash": event["chain_hash"]}
                if (not self._original_state(observed)
                        or event["chain_hash"] != self.prefix_chain_hash
                        or event["event"] != "finish"
                        or event["outcome"] != "execution_error"
                        or event["error_code"] != "timeout"
                        or event["output_sha256"] is None):
                    raise TaskBudgetError("protocol")
            return
        if kind == "resume_after_timeout":
            if (state["continuation_started"] or not self._original_state(state)
                    or state["chain_hash"] != self.prefix_chain_hash
                    or event["operation_id"] != "continuation:" + self.continuation_sha256
                    or event["case_id"] != ORIGINAL_CASE
                    or event["attempt"] != ORIGINAL_ATTEMPTS
                    or event["outcome"] != "failed" or event["error_code"] != "timeout"
                    or event["payload_sha256"] != self.continuation_sha256
                    or event["request_bytes_sha256"] != self.prefix_sha256
                    or storage._hash(event["artifact_sha256"], "protocol") != event["artifact_sha256"]
                    or any(event[k] is not None for k in _NULL_FIELDS)):
                raise TaskBudgetError("protocol")
            state["continuation_started"] = True
            state["consecutive_execution_failures"] = 1
            self._close_failed_case(state, ORIGINAL_CASE, event["artifact_sha256"])
            state["paused"] = False
            return
        if not state["continuation_started"]:
            raise TaskBudgetError("protocol")
        if kind == "case_execution_failure":
            case_id = event["case_id"]
            faults = self._recoverable_faults(state, case_id)
            if (event["operation_id"] != "failed-case:" + case_id
                    or event["attempt"] != state["attempts"]
                    or event["outcome"] != "failed"
                    or event["error_code"] not in RESUMABLE_ERRORS
                    or event["error_code"] not in {op["error_code"] for op in faults}
                    or event["payload_sha256"] != self.continuation_sha256
                    or event["request_bytes_sha256"] is not None
                    or storage._hash(event["artifact_sha256"], "protocol") != event["artifact_sha256"]
                    or any(event[k] is not None for k in _NULL_FIELDS)):
                raise TaskBudgetError("protocol")
            self._close_failed_case(state, case_id, event["artifact_sha256"])
            state["consecutive_execution_failures"] += 1
            state["paused"] = state["consecutive_execution_failures"] >= CONSECUTIVE_FAILURE_LIMIT
            return
        super()._apply(state, event)
        if kind == "case_finish":
            state["consecutive_execution_failures"] = 0

    @staticmethod
    def _close_failed_case(state, case_id, digest):
        state["completed_cases"].append(case_id)
        state["cases"][case_id] = {"artifact_sha256": digest, "outcome": "failed"}
        state["active_case"] = None

    @staticmethod
    def _recoverable_faults(state, case_id):
        if (state["active_case"] != case_id or state["pending"] or not state["paused"]
                or case_id in state["completed_cases"]):
            raise TaskBudgetError("pending" if state["pending"] else "protocol")
        ops = [op for op in state["operations"].values() if op["case_id"] == case_id]
        faults = [op for op in ops if op["outcome"] not in {"received", "structure_failed"}]
        if (not faults or any(op["outcome"] != "execution_error"
                or op["error_code"] not in RESUMABLE_ERRORS or op["output_sha256"] is None for op in faults)):
            raise TaskBudgetError("protocol")
        return faults

    def _claim_bytes(self):
        return storage._canonical({"continuation_sha256": self.continuation_sha256,
            "protocol_sha256": self.protocol_sha256, "root_sha256": hashlib.sha256(str(self.root).encode()).hexdigest(),
            "prefix_sha256": self.prefix_sha256, "attempts_at_start": ORIGINAL_ATTEMPTS}) + b"\n"

    def _run_claim(self, *, read_only):
        fd = None
        try:
            flags = os.O_RDONLY if read_only else os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(self.root / RUN_CLAIM_NAME, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            except FileNotFoundError:
                if read_only:
                    return None
                raise
            storage._private_file(fd)
            if os.fstat(fd).st_uid != os.getuid():
                raise TaskBudgetError("storage_failure")
            if read_only:
                if storage._read(fd, 4096) != self._claim_bytes():
                    raise TaskBudgetError("protocol")
            else:
                storage._write(fd, self._claim_bytes())
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return True
        except TaskBudgetError:
            raise
        except (OSError, ValueError, TypeError):
            raise TaskBudgetError("storage_failure") from None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def exclusive_run(self):
        if self._lease is not None:
            raise TaskBudgetError("ledger_busy")
        fd = None
        entered = False
        try:
            with self._locked() as (_, _, journal, claims):
                state = self._replay(journal, claims)
                if state["pending"]:
                    raise TaskBudgetError("pending")
                if not self._original_state(state) or self._run_claim(read_only=True) is not None:
                    raise TaskBudgetError("paused")
            fd = os.open(self.root / "run.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            storage._private_file(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EWOULDBLOCK, errno.EAGAIN}:
                    raise TaskBudgetError("ledger_busy") from None
                raise
            with self._locked() as (directory, _, journal, claims):
                storage._same_file(directory, "run.lock", fd)
                state = self._replay(journal, claims)
                if not self._original_state(state) or self._run_claim(read_only=True) is not None:
                    raise TaskBudgetError("paused")
                self._run_claim(read_only=False)
            lease = storage._RunLease(os.getpid(), uuid.uuid4().hex)
            self._lease, self._lease_fd = lease, fd
            entered = True
            yield lease
        except TaskBudgetError:
            raise
        except (OSError, ValueError, TypeError):
            if entered:
                raise
            raise TaskBudgetError("storage_failure") from None
        finally:
            self._lease, self._lease_fd = None, None
            if fd is not None:
                os.close(fd)

    def resume_after_timeout(self, artifact_sha256, *, lease):
        self._require_lease(lease)
        digest = storage._hash(artifact_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            return self._append_event(directory, identity, journal, claims, state,
                event="resume_after_timeout", operation_id="continuation:" + self.continuation_sha256,
                case_id=ORIGINAL_CASE, outcome="failed", artifact_sha256=digest,
                payload_sha256=self.continuation_sha256, request_bytes_sha256=self.prefix_sha256,
                error_code="timeout")

    def recover_case_failure(self, case_id, artifact_sha256, *, lease):
        self._require_lease(lease)
        digest = storage._hash(artifact_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            faults = self._recoverable_faults(state, case_id)
            code = min(faults, key=lambda op: op["attempt"])["error_code"]
            return self._append_event(directory, identity, journal, claims, state,
                event="case_execution_failure", operation_id="failed-case:" + case_id,
                case_id=case_id, outcome="failed", artifact_sha256=digest,
                payload_sha256=self.continuation_sha256, error_code=code)

    def status(self):
        with self._locked() as (_, _, journal, claims):
            s = self._replay(journal, claims)
        return {"attempts": s["attempts"], "completed_cases": list(s["completed_cases"]),
            "active_case": s["active_case"], "pending": list(s["pending"]), "paused": s["paused"],
            "chain_hash": s["chain_hash"], "turn_attempts": {f"{c}:{t}": n for (c, t), n in s["turn_attempts"].items()},
            "continuation_started": s["continuation_started"],
            "consecutive_execution_failures": s["consecutive_execution_failures"]}
