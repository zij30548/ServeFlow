"""Narrow second continuation, retaining both original ledger prefixes.

Only the independently frozen 201-attempt boundary can enter this run. The
original 480 maximum, first continuation manifest/claim and event vocabulary
remain unchanged; another directory or process never creates more allowance.
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
from evaluation.coverage_budget import ROOT, MAX_REQUESTS, CASE_IDS, fixed_plan, job_specs
from evaluation.coverage_continuation_budget import ContinuationBudget


TaskBudgetError = storage.TaskBudgetError
SCHEMA = "serveflow-coverage-continuation-v2"
MANIFEST = "continuation-v2.json"
OLD_MANIFEST = "continuation-v1.json"
RUN_CLAIM = "continuation-v2.run"
PREFIX_ATTEMPTS = 201
PREFIX_EVENTS = 541
FAILED_CASE = "A-CT-V14"
COMPLETED_CASES = 69


def _private_json(path):
    fd = None
    try:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise TaskBudgetError("protocol")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        storage._private_file(fd)
        if os.fstat(fd).st_uid != os.getuid():
            raise TaskBudgetError("storage_failure")
        raw = storage._read(fd, 8 * 1024 * 1024)
        def finite(value):
            number = float(value)
            if not math.isfinite(number):
                raise TaskBudgetError("protocol")
            return number
        def reject(_value):
            raise TaskBudgetError("protocol")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_float=finite, parse_constant=reject)
        return raw, value
    except TaskBudgetError:
        raise
    except (OSError, TypeError, ValueError, KeyError):
        raise TaskBudgetError("storage_failure") from None
    finally:
        if fd is not None:
            os.close(fd)


class Continuation2Budget(ContinuationBudget):
    """Read the original vocabulary, authorize only one explicit new lease."""

    def __init__(self, root, protocol_sha256, case_plan, *, continuation2_sha256,
                 send_limit=MAX_REQUESTS):
        root = Path(root).absolute()
        if (root.resolve() != root or any(p.is_symlink() for p in root.parents)
                or root.is_relative_to(ROOT.parent) and root != ROOT / "ledger"
                or case_plan != fixed_plan() or type(send_limit) is not int
                or not PREFIX_ATTEMPTS <= send_limit <= MAX_REQUESTS):
            raise TaskBudgetError("configuration")
        self._v2_manifest_path = root.parent / MANIFEST
        self.continuation2_sha256 = storage._hash(continuation2_sha256)
        self._v2_protocol_sha256 = storage._hash(protocol_sha256)
        raw, manifest = _private_json(self._v2_manifest_path)
        self._validate_v2_header(raw, manifest)
        self._v2_budget = dict(manifest["budget"])
        self._v1_sha256 = manifest["original_continuation_sha256"]
        raw1, first = _private_json(root.parent / OLD_MANIFEST)
        if hashlib.sha256(raw1).hexdigest() != self._v1_sha256:
            raise TaskBudgetError("protocol")
        try:
            budget1 = first["budget"]
            super().__init__(root, protocol_sha256, case_plan,
                continuation_sha256=self._v1_sha256,
                prefix_size=budget1["prefix_size"], prefix_sha256=budget1["prefix_sha256"],
                prefix_chain_hash=budget1["prefix_chain_hash"], send_limit=send_limit)
        except (KeyError, TypeError):
            raise TaskBudgetError("protocol") from None

    def _validate_v2_header(self, raw, value):
        keys = {"original_protocol_sha256", "prefix_size", "prefix_sha256",
                "prefix_chain_hash", "prefix_events", "already_spent", "max_requests"}
        budget = value.get("budget") if type(value) is dict else None
        if (hashlib.sha256(raw).hexdigest() != self.continuation2_sha256
                or type(value) is not dict or value.get("schema_version") != SCHEMA
                or type(budget) is not dict or set(budget) != keys
                or budget.get("original_protocol_sha256") != self._v2_protocol_sha256
                or type(budget.get("prefix_size")) is not int
                or not 1 <= budget["prefix_size"] <= storage._JOURNAL_LIMIT
                or type(budget.get("already_spent")) is not int or budget["already_spent"] != PREFIX_ATTEMPTS
                or type(budget.get("prefix_events")) is not int or budget["prefix_events"] != PREFIX_EVENTS
                or type(budget.get("max_requests")) is not int or budget["max_requests"] != MAX_REQUESTS
                or value.get("previous_active_case") != FAILED_CASE
                or value.get("remaining_jobs") != job_specs()[70:]):
            raise TaskBudgetError("protocol")
        for field in ("prefix_sha256", "prefix_chain_hash"):
            storage._hash(budget[field], "protocol")
        storage._hash(value.get("original_continuation_sha256"), "protocol")

    def _manifest2(self):
        raw, value = _private_json(self._v2_manifest_path)
        self._validate_v2_header(raw, value)
        if (value["budget"] != self._v2_budget
                or value["original_continuation_sha256"] != self._v1_sha256):
            raise TaskBudgetError("protocol")

    @contextmanager
    def _locked(self, create=False):
        with super()._locked(create=create) as handles:
            self._manifest2()
            yield handles

    def _replay(self, journal, claims):
        raw = storage._read(journal, storage._JOURNAL_LIMIT)
        size = self._v2_budget["prefix_size"]
        prefix = raw[:size]
        if (len(prefix) != size or not prefix.endswith(b"\n")
                or len(prefix.splitlines()) != PREFIX_EVENTS
                or hashlib.sha256(prefix).hexdigest() != self._v2_budget["prefix_sha256"]):
            raise TaskBudgetError("protocol")
        last = storage._decode(prefix.splitlines()[-1])
        if last["chain_hash"] != self._v2_budget["prefix_chain_hash"]:
            raise TaskBudgetError("protocol")
        os.lseek(journal, 0, os.SEEK_SET)
        state = super()._replay(journal, claims)
        claim = self._claim2(read_only=True)
        if len(raw) > size and claim is None:
            raise TaskBudgetError("protocol")
        if len(raw) == size and not self._boundary(state):
            raise TaskBudgetError("protocol")
        return state

    def _boundary(self, state):
        op = state["operations"].get(FAILED_CASE + ":turn-1:rag_rewrite:billing:2", {})
        return (state["attempts"] == PREFIX_ATTEMPTS and state["seq"] == PREFIX_EVENTS
                and state["chain_hash"] == self._v2_budget["prefix_chain_hash"]
                and state["continuation_started"] is True and state["paused"] is True
                and not state["pending"] and state["active_case"] == FAILED_CASE
                and state["completed_cases"] == list(CASE_IDS[:COMPLETED_CASES])
                and state["starts"] == list(CASE_IDS[:COMPLETED_CASES + 1])
                and op.get("attempt") == PREFIX_ATTEMPTS
                and op.get("outcome") == "execution_error" and op.get("error_code") == "timeout"
                and op.get("output_sha256") is not None)

    def _claim2_bytes(self):
        return storage._canonical({"continuation2_sha256": self.continuation2_sha256,
            "original_continuation_sha256": self._v1_sha256,
            "original_protocol_sha256": self.protocol_sha256,
            "prefix201_sha256": self._v2_budget["prefix_sha256"],
            "root_sha256": hashlib.sha256(str(self.root).encode()).hexdigest(),
            "attempts_at_start": PREFIX_ATTEMPTS}) + b"\n"

    def _claim2(self, *, read_only):
        fd = None
        try:
            flags = os.O_RDONLY if read_only else os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(self.root / RUN_CLAIM, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            except FileNotFoundError:
                if read_only:
                    return None
                raise
            storage._private_file(fd)
            if os.fstat(fd).st_uid != os.getuid():
                raise TaskBudgetError("storage_failure")
            if read_only:
                if storage._read(fd, 4096) != self._claim2_bytes():
                    raise TaskBudgetError("protocol")
            else:
                storage._write(fd, self._claim2_bytes())
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return True
        except TaskBudgetError:
            raise
        except (OSError, TypeError, ValueError):
            raise TaskBudgetError("storage_failure") from None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def exclusive_run(self):
        if self._lease is not None:
            raise TaskBudgetError("ledger_busy")
        fd, entered = None, False
        try:
            with self._locked() as (_, _, journal, claims):
                state = self._replay(journal, claims)
                if not self._boundary(state) or self._claim2(read_only=True) is not None:
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
                if not self._boundary(state) or self._claim2(read_only=True) is not None:
                    raise TaskBudgetError("paused")
                self._claim2(read_only=False)
            lease = storage._RunLease(os.getpid(), uuid.uuid4().hex)
            self._lease, self._lease_fd = lease, fd
            entered = True
            yield lease
        except TaskBudgetError:
            raise
        except (OSError, TypeError, ValueError):
            if entered:
                raise
            raise TaskBudgetError("storage_failure") from None
        finally:
            self._lease, self._lease_fd = None, None
            if fd is not None:
                os.close(fd)
