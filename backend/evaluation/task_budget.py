"""Local, fail-closed physical-send accounting for one fixed E2E trial.

This is not authorization or a product quota. The runner must separately bind an
explicit allowance to this protocol, canonical directory and chain genesis.
Reservations precede sends; failed/uncertain sends are never refunded. Only one
process owns a run, while up to three in-flight requests may finish out of order.
An unfinished case cannot be resumed automatically after the run lease ends.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import uuid

from evaluation.trial_ledger import (
    TrialLedger, TrialLedgerError, OUTCOMES, PAUSING_OUTCOMES, ERROR_CODES,
    _canonical, _decode, _hash, _read, _write, _private_file, _same_file,
    _same_directory, _claim_name, _JOURNAL_LIMIT, _IDENTITY_LIMIT,
)


TaskBudgetError = TrialLedgerError
MAX_REQUESTS = 168
PER_TURN = 24
MAX_IN_FLIGHT = 3
PHASE_CAPS = {"intent": 1, "agent": 15, "composer": 1,
              "rag_rewrite": 3, "rag_rerank": 3, "judge": 1}
LIMITS = {"cases": 6, "turns": 7, "physical_requests": MAX_REQUESTS,
          "per_turn": PER_TURN, "in_flight": MAX_IN_FLIGHT,
          "phase_caps": dict(PHASE_CAPS)}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_AGENTS = {"general", "technical", "billing"}
_CASE_OUTCOMES = {"success", "failed", "partial"}
_FIELDS = {"version", "seq", "attempt", "protocol_sha256", "event",
           "operation_id", "case_id", "turn_id", "phase", "agent", "ordinal",
           "payload_sha256", "request_bytes_sha256", "outcome", "output_sha256",
           "artifact_sha256", "error_code", "previous_hash", "chain_hash"}


@dataclass(frozen=True)
class _RunLease:
    pid: int
    token: str


class TaskBudget(TrialLedger):
    """A fixed 6-case / 7-turn / 168-send ledger with explicit initialization.

    ``case_plan`` is six ordered ``{case_id, turn_ids}`` dictionaries containing
    seven turns in total. It is identity material, not model input. A new ledger
    requires ``initialize=True`` and a nonexistent root; an existing directory is
    never reset or repaired. Old trial classes and their limits are unchanged.
    """

    def __init__(self, root, protocol_sha256, case_plan, *, initialize=False):
        if type(initialize) is not bool or type(case_plan) is not list or len(case_plan) != 6:
            raise TaskBudgetError("configuration")
        plan = []
        for item in case_plan:
            if (type(item) is not dict or set(item) != {"case_id", "turn_ids"}
                    or type(item["case_id"]) is not str or not _ID.fullmatch(item["case_id"])
                    or type(item["turn_ids"]) is not list or not 1 <= len(item["turn_ids"]) <= 2
                    or not all(type(t) is str and _ID.fullmatch(t) for t in item["turn_ids"])
                    or len(set(item["turn_ids"])) != len(item["turn_ids"])):
                raise TaskBudgetError("configuration")
            plan.append({"case_id": item["case_id"], "turn_ids": list(item["turn_ids"])})
        if len({p["case_id"] for p in plan}) != 6 or sum(len(p["turn_ids"]) for p in plan) != 7:
            raise TaskBudgetError("configuration")
        try:
            self.root = Path(root).absolute()
            if self.root.resolve() != self.root or any(p.is_symlink() for p in self.root.parents):
                raise TaskBudgetError("configuration")
        except (TypeError, ValueError, OSError):
            raise TaskBudgetError("configuration") from None
        if initialize and (self.root.exists() or self.root.is_symlink()):
            raise TaskBudgetError("protocol")
        self.case_plan = tuple(plan)
        self._case_ids = tuple(p["case_id"] for p in plan)
        self._turns = {p["case_id"]: tuple(p["turn_ids"]) for p in plan}
        self.protocol_sha256 = _hash(protocol_sha256)
        self.max_requests = MAX_REQUESTS
        self.operation_ids = tuple(f'{p["case_id"]}:{t}' for p in plan for t in p["turn_ids"])
        # Reuse the established private storage/locking contract. The fixed
        # larger maximum, plan identity and distinct genesis prevent old-ledger
        # interpretation; this does not loosen either original trial class.
        self._identity = {"version": 1, "protocol_sha256": self.protocol_sha256,
                          "operation_ids": list(self.operation_ids), "max_requests": MAX_REQUESTS}
        self._genesis = hashlib.sha256(b"serveflow-task-budget:identity:v1\n" + _canonical(self._identity)).hexdigest()
        self._lease = None
        self._lease_fd = None
        self._initializing = initialize
        try:
            with self._locked(create=initialize) as (directory, _identity, journal, claims):
                self._replay(journal, claims)
                if initialize:
                    fd = os.open("run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
                    try:
                        _private_file(fd)
                        _write(fd, self._run_identity())
                    finally:
                        os.close(fd)
                    os.fsync(directory)
        finally:
            self._initializing = False

    def _run_identity(self):
        return _canonical({"protocol_sha256": self.protocol_sha256,
                           "root_sha256": hashlib.sha256(str(self.root).encode()).hexdigest()}) + b"\n"

    @contextmanager
    def _locked(self, create=False):
        with super()._locked(create=create) as handles:
            for fd in handles:
                if os.fstat(fd).st_uid != os.getuid():
                    raise TaskBudgetError("storage_failure")
            directory = handles[0]
            if not self._initializing:
                fd = os.open("run.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                try:
                    _private_file(fd)
                    if os.fstat(fd).st_uid != os.getuid() or _read(fd, 1024) != self._run_identity():
                        raise TaskBudgetError("storage_failure")
                    _same_file(directory, "run.lock", fd)
                    if self._lease_fd is not None:
                        _same_file(directory, "run.lock", self._lease_fd)
                finally:
                    os.close(fd)
            yield handles

    @contextmanager
    def exclusive_run(self):
        """Hold across the run; never resume a pending or unsealed prior case."""
        if self._lease is not None:
            raise TaskBudgetError("ledger_busy")
        fd = directory = None
        entered = False
        try:
            with self._locked() as (_directory, _identity, journal, claims):
                state = self._replay(journal, claims)
                if state["paused"]:
                    raise TaskBudgetError("paused")
                if state["pending"] or state["active_case"] is not None:
                    raise TaskBudgetError("pending")
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            fd = os.open("run.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            _private_file(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EWOULDBLOCK, errno.EAGAIN}:
                    raise TaskBudgetError("ledger_busy") from None
                raise
            _same_file(directory, "run.lock", fd)
            # Check again after obtaining the exclusive run lock.
            with self._locked() as (_directory, _identity, journal, claims):
                state = self._replay(journal, claims)
                if state["paused"]:
                    raise TaskBudgetError("paused")
                if state["pending"] or state["active_case"] is not None:
                    raise TaskBudgetError("pending")
            lease = _RunLease(os.getpid(), uuid.uuid4().hex)
            self._lease, self._lease_fd = lease, fd
            entered = True
            yield lease
        except TaskBudgetError:
            raise
        except (OSError, ValueError, TypeError):
            if entered:
                # A caller's data/configuration failure is not a ledger fault.
                # Preserve its type/code while still releasing the run lease.
                raise
            raise TaskBudgetError("storage_failure") from None
        finally:
            self._lease, self._lease_fd = None, None
            if fd is not None:
                os.close(fd)
            if directory is not None:
                os.close(directory)

    def _require_lease(self, lease):
        if lease is not self._lease or self._lease is None or lease.pid != os.getpid() or self._lease_fd is None:
            raise TaskBudgetError("protocol")

    @staticmethod
    def _event_hash(unsigned):
        return hashlib.sha256(b"serveflow-task-budget:event:v1\n" + _canonical(unsigned)).hexdigest()

    def _empty(self):
        return {"attempts": 0, "completed_cases": [], "active_case": None,
                "pending": {}, "paused": False, "chain_hash": self._genesis, "seq": 0,
                "operations": {}, "cases": {}, "starts": [], "turn_attempts": {},
                "active_turn": {}, "phase_counts": {}, "actor_counts": {}}

    def _metadata(self, case_id, turn_id, phase, agent):
        if (type(case_id) is not str or type(turn_id) is not str or type(phase) is not str
                or agent is not None and type(agent) is not str
                or case_id not in self._turns or turn_id not in self._turns[case_id]
                or phase not in PHASE_CAPS or agent not in _AGENTS | {None}
                or (phase == "agent" and agent is None)
                or (phase in {"intent", "composer", "judge"} and agent is not None)):
            raise TaskBudgetError("protocol")

    def _reserve_check(self, state, case_id, turn_id, phase, agent):
        self._metadata(case_id, turn_id, phase, agent)
        if state["paused"]:
            raise TaskBudgetError("paused")
        if state["active_case"] != case_id:
            raise TaskBudgetError("protocol")
        if len(state["pending"]) >= MAX_IN_FLIGHT:
            raise TaskBudgetError("ledger_busy")
        turn = (case_id, turn_id)
        phase_key = (case_id, turn_id, phase)
        if (state["attempts"] >= MAX_REQUESTS or state["turn_attempts"].get(turn, 0) >= PER_TURN
                or state["phase_counts"].get(phase_key, 0) >= PHASE_CAPS[phase]):
            raise TaskBudgetError("quota")
        active = state["active_turn"].get(case_id)
        planned = self._turns[case_id]
        if active is None:
            if turn_id != planned[0]:
                raise TaskBudgetError("protocol")
        elif active != turn_id:
            if (planned.index(turn_id) != planned.index(active) + 1 or state["pending"]):
                raise TaskBudgetError("protocol")

    def _apply(self, state, event):
        common = (event["case_id"] in self._turns and type(event["attempt"]) is int
                  and type(event["operation_id"]) is str)
        if not common:
            raise TaskBudgetError("protocol")
        case_id, kind = event["case_id"], event["event"]
        if kind == "case_start":
            if (state["active_case"] is not None or state["paused"] or state["pending"]
                    or len(state["completed_cases"]) >= 6
                    or case_id != self._case_ids[len(state["completed_cases"])]
                    or case_id in state["starts"] or event["operation_id"] != "case:" + case_id
                    or event["attempt"] != state["attempts"]
                    or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256",
                            "request_bytes_sha256", "outcome", "output_sha256", "artifact_sha256", "error_code"))):
                raise TaskBudgetError("protocol")
            state["active_case"] = case_id
            state["starts"].append(case_id)
        elif kind == "reserve":
            self._reserve_check(state, case_id, event["turn_id"], event["phase"], event["agent"])
            actor = (case_id, event["turn_id"], event["phase"], event["agent"])
            ordinal = state["actor_counts"].get(actor, 0) + 1
            expected_id = ":".join((case_id, event["turn_id"], event["phase"], event["agent"] or "none", str(ordinal)))
            if (type(event["ordinal"]) is not int or event["ordinal"] != ordinal
                    or event["operation_id"] != expected_id or event["attempt"] != state["attempts"] + 1
                    or event["operation_id"] in state["operations"] or event["operation_id"] in state["pending"]
                    or _hash(event["payload_sha256"], "protocol") != event["payload_sha256"]
                    or _hash(event["request_bytes_sha256"], "protocol") != event["request_bytes_sha256"]
                    or any(event[k] is not None for k in ("outcome", "output_sha256", "artifact_sha256", "error_code"))):
                raise TaskBudgetError("protocol")
            state["attempts"] += 1
            state["pending"][expected_id] = event
            state["actor_counts"][actor] = ordinal
            state["active_turn"][case_id] = event["turn_id"]
            turn = (case_id, event["turn_id"])
            phase = (*turn, event["phase"])
            state["turn_attempts"][turn] = state["turn_attempts"].get(turn, 0) + 1
            state["phase_counts"][phase] = state["phase_counts"].get(phase, 0) + 1
        elif kind == "finish":
            previous = state["pending"].get(event["operation_id"])
            if (previous is None or event["outcome"] not in OUTCOMES
                    or event["error_code"] is not None and event["error_code"] not in ERROR_CODES
                    or event["artifact_sha256"] is not None
                    or any(event[k] != previous[k] for k in ("attempt", "case_id", "turn_id", "phase", "agent", "ordinal", "payload_sha256", "request_bytes_sha256"))):
                raise TaskBudgetError("protocol")
            if event["output_sha256"] is not None and _hash(event["output_sha256"], "protocol") != event["output_sha256"]:
                raise TaskBudgetError("protocol")
            if event["outcome"] in {"received", "structure_failed"} and event["output_sha256"] is None:
                raise TaskBudgetError("protocol")
            if event["outcome"] in {"received", "structure_failed"} and event["error_code"] is not None:
                raise TaskBudgetError("protocol")
            state["operations"][event["operation_id"]] = {k: event[k] for k in (
                "attempt", "case_id", "turn_id", "phase", "agent", "ordinal", "payload_sha256",
                "request_bytes_sha256", "outcome", "output_sha256", "error_code")}
            del state["pending"][event["operation_id"]]
            state["paused"] = state["paused"] or event["outcome"] in PAUSING_OUTCOMES
        elif kind == "case_finish":
            if (state["active_case"] != case_id or state["paused"] or state["pending"]
                    or event["operation_id"] != "case:" + case_id or event["attempt"] != state["attempts"]
                    or event["outcome"] not in _CASE_OUTCOMES
                    or _hash(event["artifact_sha256"], "protocol") != event["artifact_sha256"]
                    or any(event[k] is not None for k in ("turn_id", "phase", "agent", "ordinal", "payload_sha256", "request_bytes_sha256", "output_sha256", "error_code"))
                    or event["outcome"] == "success" and any((case_id, t) not in state["turn_attempts"] for t in self._turns[case_id])):
                raise TaskBudgetError("protocol")
            state["completed_cases"].append(case_id)
            state["cases"][case_id] = {"artifact_sha256": event["artifact_sha256"], "outcome": event["outcome"]}
            state["active_case"] = None
        else:
            raise TaskBudgetError("protocol")

    def _replay(self, journal, claims):
        data = _read(journal, _JOURNAL_LIMIT)
        if data and not data.endswith(b"\n"):
            raise TaskBudgetError("protocol")
        state, receipts = self._empty(), {}
        for line in data.splitlines():
            event = _decode(line)
            if (type(event) is not dict or set(event) != _FIELDS or type(event["version"]) is not int
                    or event["version"] != 1 or type(event["seq"]) is not int
                    or event["seq"] != state["seq"] + 1 or event["protocol_sha256"] != self.protocol_sha256
                    or event["previous_hash"] != state["chain_hash"]
                    or event["chain_hash"] != self._event_hash({k: v for k, v in event.items() if k != "chain_hash"})):
                raise TaskBudgetError("protocol")
            self._apply(state, event)
            if event["event"] in {"case_start", "reserve"}:
                receipts[_claim_name(event["operation_id"])] = event
            state["seq"], state["chain_hash"] = event["seq"], event["chain_hash"]
        if set(os.listdir(claims)) != set(receipts):
            raise TaskBudgetError("protocol")
        for name, expected in receipts.items():
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=claims)
            try:
                _private_file(fd)
                if os.fstat(fd).st_uid != os.getuid() or _decode(_read(fd, _IDENTITY_LIMIT)) != expected:
                    raise TaskBudgetError("protocol")
                _same_file(claims, name, fd)
            finally:
                os.close(fd)
        return state

    def _append_event(self, directory, identity, journal, claims, state, **values):
        event = {key: None for key in _FIELDS}
        event.update(version=1, seq=state["seq"] + 1, attempt=state["attempts"],
                     protocol_sha256=self.protocol_sha256, previous_hash=state["chain_hash"])
        event.update(values)
        event["chain_hash"] = self._event_hash({k: v for k, v in event.items() if k != "chain_hash"})
        # Validate before publishing a durable marker, then replay persisted bytes.
        self._apply(deepcopy(state), event)
        _same_file(directory, "identity.json", identity)
        _same_file(directory, "journal.jsonl", journal)
        _same_directory(directory, "claims", claims)
        if event["event"] in {"case_start", "reserve"}:
            fd = os.open(_claim_name(event["operation_id"]), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=claims)
            try:
                _private_file(fd)
                _write(fd, _canonical(event) + b"\n")
            finally:
                os.close(fd)
            os.fsync(claims)
        _write(journal, _canonical(event) + b"\n")
        _same_file(directory, "identity.json", identity)
        _same_file(directory, "journal.jsonl", journal)
        os.lseek(journal, 0, os.SEEK_SET)
        if self._replay(journal, claims)["chain_hash"] != event["chain_hash"]:
            raise TaskBudgetError("protocol")
        return event

    def status(self):
        with self._locked() as (_directory, _identity, journal, claims):
            state = self._replay(journal, claims)
        return {"attempts": state["attempts"], "completed_cases": list(state["completed_cases"]),
                "active_case": state["active_case"], "pending": list(state["pending"]),
                "paused": state["paused"], "chain_hash": state["chain_hash"],
                "turn_attempts": {f"{c}:{t}": n for (c, t), n in state["turn_attempts"].items()}}

    def results(self):
        with self._locked() as (_directory, _identity, journal, claims):
            state = self._replay(journal, claims)
        return deepcopy({"operations": state["operations"], "cases": state["cases"]})

    def start_case(self, case_id, *, lease):
        self._require_lease(lease)
        if type(case_id) is not str or case_id not in self._turns:
            raise TaskBudgetError("configuration")
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            return self._append_event(directory, identity, journal, claims, state,
                                      event="case_start", operation_id="case:" + case_id, case_id=case_id)

    def reserve(self, case_id, turn_id, phase, agent, payload_sha256, request_bytes_sha256, *, lease):
        self._require_lease(lease)
        payload_sha256, request_bytes_sha256 = _hash(payload_sha256), _hash(request_bytes_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            self._reserve_check(state, case_id, turn_id, phase, agent)
            ordinal = state["actor_counts"].get((case_id, turn_id, phase, agent), 0) + 1
            operation_id = ":".join((case_id, turn_id, phase, agent or "none", str(ordinal)))
            return self._append_event(directory, identity, journal, claims, state,
                event="reserve", attempt=state["attempts"] + 1, operation_id=operation_id,
                case_id=case_id, turn_id=turn_id, phase=phase, agent=agent, ordinal=ordinal,
                payload_sha256=payload_sha256, request_bytes_sha256=request_bytes_sha256)

    def finish(self, operation_id, outcome, output_sha256=None, error_code=None, *, lease):
        self._require_lease(lease)
        if (type(operation_id) is not str or type(outcome) is not str or outcome not in OUTCOMES
                or error_code is not None and (type(error_code) is not str or error_code not in ERROR_CODES)):
            raise TaskBudgetError("configuration")
        if output_sha256 is not None:
            output_sha256 = _hash(output_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            previous = state["pending"].get(operation_id)
            if previous is None:
                raise TaskBudgetError("protocol")
            metadata = {key: previous[key] for key in ("attempt", "operation_id", "case_id", "turn_id", "phase", "agent", "ordinal", "payload_sha256", "request_bytes_sha256")}
            return self._append_event(directory, identity, journal, claims, state, **metadata,
                event="finish", outcome=outcome, output_sha256=output_sha256, error_code=error_code)

    def finish_case(self, case_id, artifact_sha256, outcome="success", *, lease):
        self._require_lease(lease)
        if type(case_id) is not str or case_id not in self._turns or type(outcome) is not str or outcome not in _CASE_OUTCOMES:
            raise TaskBudgetError("configuration")
        artifact_sha256 = _hash(artifact_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            return self._append_event(directory, identity, journal, claims, state,
                event="case_finish", operation_id="case:" + case_id, case_id=case_id,
                artifact_sha256=artifact_sha256, outcome=outcome)
