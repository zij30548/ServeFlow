"""Fail-closed, local send reservations for explicitly authorized evaluations.

This ledger records attempts, not authorization. The caller must bind the canonical
batch directory, protocol and allowance to its authorization chain before using it.
It must send only after ``reserve`` returns, never retry a reservation, and finish
an interrupted/uncertain send as ``cancelled``/``unknown`` when possible. A pending
reservation remains blocking across restarts. Nothing here makes network calls.

The journal and per-operation claims are append-only. Claims also detect removal
of a whole valid journal suffix. This cannot protect against an administrator
rolling back the entire directory, including claims. Keep authorization evidence
separate and bind its canonical directory; a new directory is not a new allowance.
"""
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


VERSION = 1
OUTCOMES = frozenset({
    "received", "structure_failed", "execution_error", "cancelled", "unknown",
})
PAUSING_OUTCOMES = frozenset({"execution_error", "cancelled", "unknown"})
ERROR_CODES = frozenset({
    "configuration", "version", "auth", "quota", "rate_limit", "network",
    "timeout", "cancelled", "response_limit", "protocol", "storage_failure",
})
LEDGER_ERROR_CODES = ERROR_CODES | {
    "pending", "paused", "ledger_busy",
}
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_IDENTITY_FIELDS = {"version", "protocol_sha256", "operation_ids", "max_requests"}
_EVENT_FIELDS = {
    "version", "seq", "attempt", "protocol_sha256", "operation_id", "event",
    "payload_sha256", "outcome", "output_sha256", "error_code", "previous_hash",
    "chain_hash",
}
_CLAIM_FIELDS = {
    "version", "seq", "attempt", "protocol_sha256", "operation_id", "payload_sha256",
}
_IDENTITY_LIMIT = 1024 * 1024
_JOURNAL_LIMIT = 8 * 1024 * 1024


class TrialLedgerError(Exception):
    """A fixed safe code: never include paths, payloads or raw storage errors."""

    def __init__(self, code: str):
        self.code = code if code in LEDGER_ERROR_CODES else "protocol"
        super().__init__(self.code)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash(value, code="configuration") -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise TrialLedgerError(code)
    return value.lower()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TrialLedgerError("protocol")
        result[key] = value
    return result


def _non_integer(_value):
    # Neither file has any floating point fields; this also rejects 1e999.
    raise TrialLedgerError("protocol")


def _decode(data: bytes):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_float=_non_integer, parse_constant=_non_integer)
    except (ValueError, UnicodeError, RecursionError):
        raise TrialLedgerError("protocol") from None


def _read(fd: int, limit: int) -> bytes:
    result = bytearray()
    while True:
        chunk = os.read(fd, min(65536, limit + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > limit:
            raise TrialLedgerError("protocol")


def _write(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise TrialLedgerError("storage_failure")
        remaining = remaining[written:]
    # Return from reserve is the send boundary. Failure here must never permit it.
    os.fsync(fd)


def _private_file(fd: int) -> None:
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise TrialLedgerError("storage_failure")


def _same_file(directory_fd: int, filename: str, fd: int) -> None:
    current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    opened = os.fstat(fd)
    if ((current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
            or stat.S_IMODE(current.st_mode) != 0o600):
        raise TrialLedgerError("storage_failure")


def _claim_name(operation_id: str) -> str:
    return hashlib.sha256(operation_id.encode("utf-8")).hexdigest() + ".json"


def _same_directory(parent_fd: int, name: str, fd: int) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(fd)
    if (not stat.S_ISDIR(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o700
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
        raise TrialLedgerError("storage_failure")


class TrialLedger:
    """A private, nonblocking-flock JSONL ledger for at most 24 send attempts.

    Constructing is the only operation that can create the root, identity, journal
    and claims directory. Reserve creates only a new immutable operation claim.
    The root's parent must already exist. Existing files are never reset, repaired,
    truncated, chmod'ed or replaced, including a missing journal after identity.
    ``completed`` includes every finished attempt, including pausing outcomes.
    """

    def __init__(self, root: Path, protocol_sha256: str,
                 operation_ids: list[str], max_requests: int = 24):
        if (type(max_requests) is not int or not 1 <= max_requests <= 24
                or not isinstance(operation_ids, list)
                or not all(isinstance(op, str) and op for op in operation_ids)
                or len(set(operation_ids)) != len(operation_ids)):
            raise TrialLedgerError("configuration")
        try:
            self.root = Path(root)
        except (TypeError, ValueError):
            raise TrialLedgerError("configuration") from None
        self.protocol_sha256 = _hash(protocol_sha256)
        try:
            for operation_id in operation_ids:
                operation_id.encode("utf-8")
        except UnicodeError:
            raise TrialLedgerError("configuration") from None
        self.operation_ids = tuple(operation_ids)
        self.max_requests = max_requests
        self._identity = {
            "version": VERSION, "protocol_sha256": self.protocol_sha256,
            "operation_ids": list(self.operation_ids), "max_requests": max_requests,
        }
        if len(_canonical(self._identity)) > _IDENTITY_LIMIT:
            raise TrialLedgerError("configuration")
        self._genesis = hashlib.sha256(
            b"serveflow-trial-ledger:identity:v1\n" + _canonical(self._identity)
        ).hexdigest()
        with self._locked(create=True) as (_directory, _identity, journal, claims):
            self._replay(journal, claims)

    @contextmanager
    def _locked(self, create=False):
        descriptors = []
        identity_fd = None
        try:
            if create:
                try:
                    self.root.mkdir(mode=0o700)
                    # Persist the directory entry before publishing reservations.
                    parent_fd = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
                except FileExistsError:
                    pass
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.append(directory_fd)
            info = os.fstat(directory_fd)
            if stat.S_IMODE(info.st_mode) != 0o700:
                raise TrialLedgerError("storage_failure")
            present = {}
            for filename in ("identity.json", "journal.jsonl", "claims"):
                try:
                    os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                    present[filename] = True
                except FileNotFoundError:
                    present[filename] = False
            new_identity = False
            if not present["identity.json"]:
                if not create or present["journal.jsonl"] or present["claims"]:
                    raise TrialLedgerError("protocol")
                try:
                    identity_fd = os.open(
                        "identity.json", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=directory_fd,
                    )
                    new_identity = True
                except FileExistsError:
                    # A concurrent initializer won; lock it, never reset it.
                    identity_fd = os.open("identity.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                          dir_fd=directory_fd)
            else:
                identity_fd = os.open("identity.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                      dir_fd=directory_fd)
            descriptors.append(identity_fd)
            _private_file(identity_fd)
            try:
                fcntl.flock(identity_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EWOULDBLOCK, errno.EAGAIN}:
                    raise TrialLedgerError("ledger_busy") from None
                raise
            _same_file(directory_fd, "identity.json", identity_fd)
            if new_identity:
                _write(identity_fd, _canonical(self._identity) + b"\n")
                journal_fd = os.open(
                    "journal.jsonl", os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
                    | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd,
                )
                descriptors.append(journal_fd)
                _private_file(journal_fd)
                os.fsync(journal_fd)
                os.mkdir("claims", mode=0o700, dir_fd=directory_fd)
                claims_fd = os.open("claims", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory_fd)
                descriptors.append(claims_fd)
                os.fsync(claims_fd)
                os.fsync(directory_fd)
            else:
                identity = _decode(_read(identity_fd, _IDENTITY_LIMIT))
                if (not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS
                        or type(identity.get("version")) is not int):
                    raise TrialLedgerError("protocol")
                if identity["version"] != VERSION:
                    raise TrialLedgerError("version")
                if (type(identity["max_requests"]) is not int
                        or identity != self._identity):
                    raise TrialLedgerError("protocol")
                # O_CREAT is intentionally absent: a missing file is not recovery.
                journal_fd = os.open("journal.jsonl", os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     dir_fd=directory_fd)
                descriptors.append(journal_fd)
                _private_file(journal_fd)
                claims_fd = os.open("claims", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory_fd)
                descriptors.append(claims_fd)
            if stat.S_IMODE(os.fstat(claims_fd).st_mode) != 0o700:
                raise TrialLedgerError("storage_failure")
            _same_file(directory_fd, "journal.jsonl", journal_fd)
            _same_directory(directory_fd, "claims", claims_fd)
            yield directory_fd, identity_fd, journal_fd, claims_fd
            _same_directory(directory_fd, "claims", claims_fd)
        except TrialLedgerError:
            raise
        except (OSError, ValueError, OverflowError, RecursionError, TypeError):
            raise TrialLedgerError("storage_failure") from None
        finally:
            # Closing identity releases flock. Always attempt every close, and
            # keep raw storage exceptions out of errors delivered to the caller.
            cleanup_failed = False
            for fd in reversed(descriptors):
                try:
                    os.close(fd)
                except OSError:
                    cleanup_failed = True
            if cleanup_failed:
                raise TrialLedgerError("storage_failure") from None

    def _replay(self, fd, claims_fd):
        data = _read(fd, _JOURNAL_LIMIT)
        if data and not data.endswith(b"\n"):
            raise TrialLedgerError("protocol")
        state = {"attempts": 0, "completed": [], "pending": [], "paused": False,
                 "chain_hash": self._genesis, "seq": 0, "reservation": None,
                 "results": {}}
        reservations = []
        for line in data.splitlines():
            event = _decode(line)
            if (not isinstance(event, dict) or set(event) != _EVENT_FIELDS
                    or type(event["version"]) is not int):
                raise TrialLedgerError("protocol")
            if event["version"] != VERSION:
                raise TrialLedgerError("version")
            unsigned = {key: value for key, value in event.items() if key != "chain_hash"}
            expected_hash = self._event_hash(unsigned)
            if (type(event["seq"]) is not int or event["seq"] != state["seq"] + 1
                    or type(event["attempt"]) is not int
                    or event["protocol_sha256"] != self.protocol_sha256
                    or event["previous_hash"] != state["chain_hash"]
                    or event["chain_hash"] != expected_hash
                    or not isinstance(event["operation_id"], str)
                    or _hash(event["payload_sha256"], "protocol") != event["payload_sha256"]):
                raise TrialLedgerError("protocol")
            if event["event"] == "reserve":
                if (state["pending"] or state["paused"]
                        or state["attempts"] >= self.max_requests
                        or state["attempts"] >= len(self.operation_ids)
                        or event["operation_id"] != self.operation_ids[state["attempts"]]
                        or event["attempt"] != state["attempts"] + 1
                        or any(event[field] is not None for field in
                               ("outcome", "output_sha256", "error_code"))):
                    raise TrialLedgerError("protocol")
                state["attempts"] += 1
                state["pending"] = [event["operation_id"]]
                state["reservation"] = event
                reservations.append(event)
            elif event["event"] == "finish":
                reservation = state["reservation"]
                if (state["pending"] != [event["operation_id"]] or reservation is None
                        or event["attempt"] != reservation["attempt"]
                        or event["payload_sha256"] != reservation["payload_sha256"]
                        or not isinstance(event["outcome"], str)
                        or event["outcome"] not in OUTCOMES
                        or (event["error_code"] is not None and
                            (not isinstance(event["error_code"], str)
                             or event["error_code"] not in ERROR_CODES))):
                    raise TrialLedgerError("protocol")
                if event["output_sha256"] is not None:
                    if _hash(event["output_sha256"], "protocol") != event["output_sha256"]:
                        raise TrialLedgerError("protocol")
                state["completed"].append(event["operation_id"])
                state["results"][event["operation_id"]] = {
                    key: event[key] for key in ("output_sha256", "outcome", "error_code")
                }
                state["pending"] = []
                state["reservation"] = None
                state["paused"] = event["outcome"] in PAUSING_OUTCOMES
            else:
                raise TrialLedgerError("protocol")
            state["seq"] = event["seq"]
            state["chain_hash"] = event["chain_hash"]
        expected_names = {_claim_name(event["operation_id"]) for event in reservations}
        if set(os.listdir(claims_fd)) != expected_names:
            raise TrialLedgerError("protocol")
        for event in reservations:
            filename = _claim_name(event["operation_id"])
            claim_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                               dir_fd=claims_fd)
            try:
                _private_file(claim_fd)
                claim = _decode(_read(claim_fd, _IDENTITY_LIMIT))
                expected = {key: event[key] for key in _CLAIM_FIELDS}
                if (not isinstance(claim, dict) or set(claim) != _CLAIM_FIELDS
                        or _canonical(claim) != _canonical(expected)):
                    raise TrialLedgerError("protocol")
                _same_file(claims_fd, filename, claim_fd)
            finally:
                os.close(claim_fd)
        return state

    @staticmethod
    def _event_hash(unsigned):
        return hashlib.sha256(b"serveflow-trial-ledger:event:v1\n"
                              + _canonical(unsigned)).hexdigest()

    def _append(self, directory_fd, identity_fd, journal_fd, claims_fd, state, event):
        event = {"version": VERSION, "seq": state["seq"] + 1,
                 "protocol_sha256": self.protocol_sha256,
                 "previous_hash": state["chain_hash"], **event}
        event["chain_hash"] = self._event_hash(event)
        _same_file(directory_fd, "identity.json", identity_fd)
        _same_file(directory_fd, "journal.jsonl", journal_fd)
        _same_directory(directory_fd, "claims", claims_fd)
        if event["event"] == "reserve":
            # The immutable receipt is published before the journal. An orphan
            # from any failure is blocking evidence, never silently repaired.
            claim_fd = os.open(
                _claim_name(event["operation_id"]),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=claims_fd,
            )
            try:
                _private_file(claim_fd)
                _write(claim_fd, _canonical({key: event[key] for key in _CLAIM_FIELDS}) + b"\n")
            finally:
                os.close(claim_fd)
            os.fsync(claims_fd)
        _write(journal_fd, _canonical(event) + b"\n")
        _same_file(directory_fd, "identity.json", identity_fd)
        _same_file(directory_fd, "journal.jsonl", journal_fd)
        _same_directory(directory_fd, "claims", claims_fd)
        # Recheck the complete persisted pair before granting send permission.
        os.lseek(journal_fd, 0, os.SEEK_SET)
        verified = self._replay(journal_fd, claims_fd)
        if verified["chain_hash"] != event["chain_hash"] or verified["seq"] != event["seq"]:
            raise TrialLedgerError("protocol")
        return event

    def status(self) -> dict:
        with self._locked() as (_directory, _identity, journal, claims):
            state = self._replay(journal, claims)
            return {key: state[key] for key in
                    ("attempts", "completed", "pending", "paused", "chain_hash")}

    @property
    def initial_chain_hash(self) -> str:
        """The immutable protocol-bound chain genesis for authorization binding."""
        return self._genesis

    def results(self) -> dict[str, dict]:
        """Return verified finish metadata for binding observations to their hashes."""
        with self._locked() as (_directory, _identity, journal, claims):
            state = self._replay(journal, claims)
            return {operation_id: dict(result)
                    for operation_id, result in state["results"].items()}

    def reserve(self, operation_id: str, payload_sha256: str) -> dict:
        payload_sha256 = _hash(payload_sha256)
        if not isinstance(operation_id, str):
            raise TrialLedgerError("configuration")
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            if state["pending"]:
                raise TrialLedgerError("pending")
            if state["paused"]:
                raise TrialLedgerError("paused")
            if state["attempts"] >= self.max_requests:
                raise TrialLedgerError("quota")
            if (state["attempts"] >= len(self.operation_ids)
                    or operation_id != self.operation_ids[state["attempts"]]):
                raise TrialLedgerError("protocol")
            return self._append(directory, identity, journal, claims, state, {
                "attempt": state["attempts"] + 1, "operation_id": operation_id,
                "event": "reserve", "payload_sha256": payload_sha256,
                "outcome": None, "output_sha256": None, "error_code": None,
            })

    def finish(self, operation_id: str, outcome: str, output_sha256=None,
               error_code=None) -> dict:
        if (not isinstance(operation_id, str) or not isinstance(outcome, str)
                or outcome not in OUTCOMES
                or (error_code is not None and
                    (not isinstance(error_code, str) or error_code not in ERROR_CODES))):
            raise TrialLedgerError("configuration")
        if output_sha256 is not None:
            output_sha256 = _hash(output_sha256)
        with self._locked() as (directory, identity, journal, claims):
            state = self._replay(journal, claims)
            if state["pending"] != [operation_id]:
                raise TrialLedgerError("protocol")
            reservation = state["reservation"]
            return self._append(directory, identity, journal, claims, state, {
                "attempt": reservation["attempt"], "operation_id": operation_id,
                "event": "finish", "payload_sha256": reservation["payload_sha256"],
                "outcome": outcome, "output_sha256": output_sha256,
                "error_code": error_code,
            })
