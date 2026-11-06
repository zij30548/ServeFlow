"""Single-host, persistent model-send permits for the protected demonstration.

This is a conservative request-count ceiling, not a currency budget. Initialization
is an explicit operator action. Runtime never recreates missing state, refunds an
attempt, resets on restart, or stores prompts, provider bodies or credentials.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # Ordinary SDK callers remain usable on other platforms.
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid


SCHEMA = "serveflow-protected-demo-permits-v1"
MAX_LIMIT = 10000
JOURNAL_LIMIT = 16 * 1024 * 1024
MODEL = "qwen3.7-flash-2026-07-15"
EMBEDDING_MODEL = "qwen3.7-text-embedding"
EMBEDDING_PHASE = "intent_embedding"
PHASES = frozenset({"intent", "agent", "retrieval", "memory", "judge", "provider", EMBEDDING_PHASE})
OUTCOMES = frozenset({"received", "http_error", "network_error", "timeout", "cancelled", "response_limit", "unknown", "not_sent"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_OP = re.compile(r"[0-9a-f]{32}\Z")
_PROCESS = uuid.uuid4().hex
_PROCESS_PID = os.getpid()
_ACTIVE: set[tuple[str, str]] = set()


def _valid_model_phase(model, phase):
    return (type(phase) is str and phase in PHASES and
            (model == EMBEDDING_MODEL if phase == EMBEDDING_PHASE else model == MODEL))


def _process_id():
    global _PROCESS, _PROCESS_PID
    if _PROCESS_PID != os.getpid():
        _PROCESS, _PROCESS_PID = uuid.uuid4().hex, os.getpid()
        _ACTIVE.clear()
    return _PROCESS


class ProtectedDemoQuotaError(RuntimeError):
    """A fixed, safe code suitable for the API; no underlying exception input."""
    _CODES = frozenset({"demo_model_quota_exhausted", "demo_model_quota_unavailable",
                        "demo_model_request_rejected", "demo_model_provider_unavailable"})

    def __init__(self, code="demo_model_quota_unavailable"):
        self.code = code if code in self._CODES else "demo_model_quota_unavailable"
        self.status_code = 429 if self.code == "demo_model_quota_exhausted" else 503
        super().__init__(self.code)


@dataclass
class RequestQuotaState:
    active: bool = True
    code: str | None = None


_REQUEST: ContextVar[RequestQuotaState | None] = ContextVar("protected_demo_model_request", default=None)


def enabled():
    return os.getenv("ECHOMIND_ENV", "development").lower() == "protected-demo"


@contextmanager
def request_scope():
    value = RequestQuotaState()
    token = _REQUEST.set(value)
    try:
        yield value
    finally:
        value.active = False
        _REQUEST.reset(token)


def block_request(error):
    state = _REQUEST.get()
    if state is not None and state.active and state.code is None:
        state.code = error.code if isinstance(error, ProtectedDemoQuotaError) else "demo_model_quota_unavailable"


def raise_if_request_blocked():
    if not enabled():
        return
    state = _REQUEST.get()
    if state is not None and state.active and state.code:
        raise ProtectedDemoQuotaError(state.code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _decode(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def _private(fd, directory=False):
    info = os.fstat(fd)
    if (not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)
            or not directory and info.st_nlink != 1):
        raise ProtectedDemoQuotaError()


def _read(fd, maximum):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks, size = [], 0
    while True:
        part = os.read(fd, min(65536, maximum + 1 - size))
        if not part:
            return b"".join(chunks)
        chunks.append(part)
        size += len(part)
        if size > maximum:
            raise ProtectedDemoQuotaError()


def _write(fd, raw):
    view = memoryview(raw)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise ProtectedDemoQuotaError()
        view = view[n:]
    os.fsync(fd)


class QuotaStore:
    def __init__(self, path, limit):
        try:
            self.path = Path(path)
            if (fcntl is None or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
                    or not self.path.is_absolute() or self.path.resolve() != self.path
                    or any(p.is_symlink() for p in (self.path, *self.path.parents))
                    or type(limit) is not int or not 1 <= limit <= MAX_LIMIT):
                raise ValueError()
            self.limit = limit
            _process_id()
        except Exception:
            raise ProtectedDemoQuotaError() from None

    @classmethod
    def from_env(cls):
        value = os.getenv("SERVEFLOW_DEMO_MODEL_LIMIT", "100")
        if not re.fullmatch(r"[1-9][0-9]{0,4}", value):
            raise ProtectedDemoQuotaError()
        return cls(os.getenv("SERVEFLOW_DEMO_QUOTA_DIR", "/var/lib/serveflow/usage"), int(value))

    def initialize(self):
        """Explicit one-time creation; existing state is verified, never replaced."""
        directory = None
        try:
            if not self.path.exists():
                self.path.mkdir(mode=0o700)
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            _private(directory, True)
            names = os.listdir(directory)
            if names:
                return self.status()
            identity = {"schema_version": SCHEMA, "instance_id": uuid.uuid4().hex, "limit": self.limit}
            # O_EXCL makes two simultaneous initializers stop rather than replace
            # each other's partial state. A partial creation is not auto-repaired.
            for name, raw in (("identity.json", _canonical(identity) + b"\n"),
                              ("journal.jsonl", b""), ("run.lock", b"")):
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                try:
                    _write(fd, raw)
                finally:
                    os.close(fd)
            os.fsync(directory)
            return self.status()
        except ProtectedDemoQuotaError:
            raise
        except Exception:
            raise ProtectedDemoQuotaError() from None
        finally:
            if directory is not None:
                os.close(directory)

    @contextmanager
    def _locked(self):
        handles = []
        try:
            if self.path.resolve() != self.path or any(p.is_symlink() for p in (self.path, *self.path.parents)):
                raise ProtectedDemoQuotaError()
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            handles.append(directory)
            _private(directory, True)
            if set(os.listdir(directory)) != {"identity.json", "journal.jsonl", "run.lock"}:
                raise ProtectedDemoQuotaError()
            lock = os.open("run.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            handles.append(lock)
            _private(lock)
            # Brief file contention is not an upstream retry. Never wait
            # indefinitely for a stalled operator/process holding the lock.
            deadline = time.monotonic() + .25
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ProtectedDemoQuotaError() from None
                    time.sleep(.005)
            identity = os.open("identity.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            handles.append(identity)
            _private(identity)
            record = _decode(_read(identity, 2048))
            if (type(record) is not dict or set(record) != {"schema_version", "instance_id", "limit"}
                    or record["schema_version"] != SCHEMA or record["limit"] != self.limit
                    or type(record["limit"]) is not int or not _OP.fullmatch(record["instance_id"])):
                raise ProtectedDemoQuotaError()
            journal = os.open("journal.jsonl", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            handles.append(journal)
            _private(journal)
            raw = _read(journal, JOURNAL_LIMIT)
            state = self._replay(record, raw)
            yield journal, state
        except ProtectedDemoQuotaError:
            raise
        except Exception:
            raise ProtectedDemoQuotaError() from None
        finally:
            for fd in reversed(handles):
                os.close(fd)

    def _replay(self, identity, raw):
        if raw and not raw.endswith(b"\n"):
            raise ProtectedDemoQuotaError()
        state = {"used": 0, "seq": 0, "chain": _digest(identity), "operations": {}}
        common = {"event", "seq", "operation_id", "attempt", "process_id", "phase", "model", "at",
                  "payload_sha256", "outcome", "http_status", "delegate_entered", "previous_hash", "chain_hash"}
        for line in raw.splitlines():
            event = _decode(line)
            if (type(event) is not dict or set(event) != common or event["seq"] != state["seq"] + 1
                    or type(event["seq"]) is not int or not _OP.fullmatch(event["operation_id"])
                    or not _OP.fullmatch(event["process_id"]) or event["phase"] not in PHASES
                    or not _valid_model_phase(event["model"], event["phase"]) or not _SHA.fullmatch(event["payload_sha256"])
                    or event["previous_hash"] != state["chain"]
                    or event["chain_hash"] != _digest({k: v for k, v in event.items() if k != "chain_hash"})
                    or type(event["attempt"]) is not int or not 1 <= event["attempt"] <= self.limit):
                raise ProtectedDemoQuotaError()
            if type(event["at"]) is not str or len(event["at"]) > 40:
                raise ProtectedDemoQuotaError()
            datetime.fromisoformat(event["at"])
            op = state["operations"].get(event["operation_id"])
            if event["event"] == "reserve":
                if (op is not None or event["attempt"] != state["used"] + 1
                        or event["outcome"] is not None or event["http_status"] is not None or event["delegate_entered"] is not None):
                    raise ProtectedDemoQuotaError()
                state["used"] += 1
                state["operations"][event["operation_id"]] = event
            elif event["event"] == "finish":
                if (op is None or op["outcome"] is not None or event["outcome"] not in OUTCOMES
                        or any(event[k] != op[k] for k in ("attempt", "process_id", "phase", "model", "payload_sha256"))
                        or event["http_status"] is not None and
                           (type(event["http_status"]) is not int or not 100 <= event["http_status"] <= 599)
                        or type(event["delegate_entered"]) is not bool
                        or (event["outcome"] == "not_sent") != (event["delegate_entered"] is False)):
                    raise ProtectedDemoQuotaError()
                state["operations"][event["operation_id"]] = event
            else:
                raise ProtectedDemoQuotaError()
            state["seq"], state["chain"] = event["seq"], event["chain_hash"]
        return state

    @staticmethod
    def _public(state, limit):
        values = list(state["operations"].values())
        return {"schema_version": SCHEMA, "limit": limit, "used": state["used"],
                "remaining": limit - state["used"], "unresolved": sum(v["outcome"] is None for v in values),
                "unit": "conservative model-send permits; not currency or successful responses"}

    def status(self):
        with self._locked() as (_, state):
            return self._public(state, self.limit)

    verify = status

    def _available(self, state):
        if any(v["outcome"] is None and (str(self.path), k) not in _ACTIVE
               for k, v in state["operations"].items()):
            raise ProtectedDemoQuotaError()
        if state["used"] >= self.limit:
            raise ProtectedDemoQuotaError("demo_model_quota_exhausted")

    def ensure_available(self):
        with self._locked() as (_, state):
            self._available(state)
            return self._public(state, self.limit)

    @staticmethod
    def _append(fd, state, value):
        value = {**value, "seq": state["seq"] + 1, "at": datetime.now(timezone.utc).isoformat(),
                 "previous_hash": state["chain"]}
        value["chain_hash"] = _digest(value)
        os.lseek(fd, 0, os.SEEK_END)
        _write(fd, _canonical(value) + b"\n")

    def reserve(self, *, model, phase, payload_sha256):
        if (not _valid_model_phase(model, phase)
                or not isinstance(payload_sha256, str) or not _SHA.fullmatch(payload_sha256)):
            raise ProtectedDemoQuotaError("demo_model_request_rejected")
        with self._locked() as (fd, state):
            self._available(state)
            op = uuid.uuid4().hex
            value = {"event": "reserve", "operation_id": op, "attempt": state["used"] + 1,
                     "process_id": _process_id(), "phase": phase, "model": model, "payload_sha256": payload_sha256,
                     "outcome": None, "http_status": None, "delegate_entered": None}
            self._append(fd, state, value)
            _ACTIVE.add((str(self.path), op))
            return op

    def finish(self, operation_id, outcome, http_status=None, *, delegate_entered=True):
        try:
            if (type(outcome) is not str or outcome not in OUTCOMES
                    or http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599)
                    or type(delegate_entered) is not bool or (outcome == "not_sent") != (delegate_entered is False)):
                raise ProtectedDemoQuotaError()
            with self._locked() as (fd, state):
                prior = state["operations"].get(operation_id)
                if prior is None or prior["outcome"] is not None or prior["process_id"] != _process_id():
                    raise ProtectedDemoQuotaError()
                value = {k: v for k, v in prior.items() if k not in {"seq", "at", "previous_hash", "chain_hash"}}
                value.update(event="finish", outcome=outcome, http_status=http_status, delegate_entered=delegate_entered)
                self._append(fd, state, value)
        finally:
            _ACTIVE.discard((str(self.path), operation_id))


def initialize(path=None, limit=None):
    return (QuotaStore.from_env() if path is None else QuotaStore(path, limit)).initialize()


def verify(path=None, limit=None):
    return (QuotaStore.from_env() if path is None else QuotaStore(path, limit)).verify()


status = verify


def ensure_available():
    if not enabled():
        return None
    try:
        raise_if_request_blocked()
        return QuotaStore.from_env().ensure_available()
    except ProtectedDemoQuotaError as error:
        block_request(error)
        raise
