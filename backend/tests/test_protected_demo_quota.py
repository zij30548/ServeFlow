"""Real temporary files/processes only; no models, databases or environment file."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat

import pytest

from core import protected_demo_quota as q


H = hashlib.sha256(b"controlled payload only").hexdigest()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs): raise AssertionError("no external calls")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("ECHOMIND_ENV", "protected-demo")


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path.resolve() / "quota"
    monkeypatch.setenv("SERVEFLOW_DEMO_QUOTA_DIR", str(path))
    monkeypatch.setenv("SERVEFLOW_DEMO_MODEL_LIMIT", "3")
    value = q.QuotaStore(path, 3)
    value.initialize()
    return value


def send(store, outcome="received"):
    operation = store.reserve(model=q.MODEL, phase="agent", payload_sha256=H)
    store.finish(operation, outcome, 200 if outcome == "received" else None)
    return operation


def _reserve_and_finish(path):
    from core.protected_demo_quota import QuotaStore, MODEL, ProtectedDemoQuotaError
    try:
        s = QuotaStore(path, 3)
        op = s.reserve(model=MODEL, phase="agent", payload_sha256=H)
        s.finish(op, "received", 200)
        return True
    except ProtectedDemoQuotaError:
        return False


def _unknown_send(path):
    from core.protected_demo_quota import QuotaStore, MODEL
    QuotaStore(path, 3).reserve(model=MODEL, phase="agent", payload_sha256=H)
    return True


def test_explicit_init_is_idempotent_and_restart_never_resets(store):
    before = (store.path / "identity.json").read_bytes()
    send(store)
    reopened = q.QuotaStore(store.path, 3)
    assert reopened.initialize()["used"] == 1
    assert reopened.verify()["remaining"] == 2
    assert before == (store.path / "identity.json").read_bytes()
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in store.path.iterdir())


def test_missing_runtime_state_never_creates_or_repairs(tmp_path):
    p = tmp_path.resolve() / "new"
    value = q.QuotaStore(p, 3)
    with pytest.raises(q.ProtectedDemoQuotaError): value.verify()
    assert not p.exists()
    value.initialize()
    (p / "journal.jsonl").unlink()
    with pytest.raises(q.ProtectedDemoQuotaError): value.verify()
    with pytest.raises(q.ProtectedDemoQuotaError): value.initialize()
    assert not (p / "journal.jsonl").exists()


@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "network_error", "unknown", "http_error", "response_limit"])
def test_failed_and_uncertain_sends_always_remain_charged(store, outcome):
    send(store, outcome)
    assert store.status()["used"] == 1
    assert store.status()["remaining"] == 2
    assert store.status()["unresolved"] == 0


def test_cap_is_shared_by_clients_and_limit_cannot_be_changed(store):
    for _ in range(3): send(q.QuotaStore(store.path, 3))
    with pytest.raises(q.ProtectedDemoQuotaError, match="exhausted"):
        send(store)
    assert store.verify()["used"] == 3  # startup/status still available
    with pytest.raises(q.ProtectedDemoQuotaError): q.QuotaStore(store.path, 4).status()


def test_finished_operation_cannot_be_reused(store):
    op = send(store)
    before = (store.path / "journal.jsonl").read_bytes()
    with pytest.raises(q.ProtectedDemoQuotaError): store.finish(op, "received", 200)
    assert (store.path / "journal.jsonl").read_bytes() == before


@pytest.mark.parametrize("name", ["identity.json", "journal.jsonl", "run.lock"])
def test_symlink_hardlink_and_wrong_file_mode_reject(store, name, tmp_path):
    p = store.path / name
    other = tmp_path / "other"
    other.write_bytes(p.read_bytes())
    other.chmod(0o600)
    p.unlink()
    p.symlink_to(other)
    with pytest.raises(q.ProtectedDemoQuotaError): store.status()
    p.unlink()
    os.link(other, p)
    with pytest.raises(q.ProtectedDemoQuotaError): store.status()
    p.unlink()
    p.write_bytes(other.read_bytes())
    p.chmod(0o644)
    with pytest.raises(q.ProtectedDemoQuotaError): store.status()


@pytest.mark.parametrize("damage", ["truncated", "changed", "duplicate", "extra_file", "identity_limit"])
def test_corrupt_or_unexpected_state_is_fail_closed(store, damage):
    send(store)
    p = store.path / "journal.jsonl"
    if damage == "truncated": p.write_bytes(p.read_bytes()[:-1])
    elif damage == "changed": p.write_bytes(p.read_bytes().replace(b'"received"', b'"unknown"'))
    elif damage == "duplicate": p.write_bytes(p.read_bytes().replace(b'"event":', b'"event":"x","event":', 1))
    elif damage == "extra_file": (store.path / "surprise").touch(mode=0o600)
    else:
        p = store.path / "identity.json"
        v = json.loads(p.read_text());v["limit"] = 4;p.write_text(json.dumps(v))
    with pytest.raises(q.ProtectedDemoQuotaError): store.ensure_available()


def test_unknown_attempt_after_process_exit_blocks_new_send_without_refund(store):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(_unknown_send, str(store.path)).result(timeout=15)
    assert store.verify()["used"] == store.verify()["unresolved"] == 1
    with pytest.raises(q.ProtectedDemoQuotaError): store.ensure_available()
    assert store.status()["remaining"] == 2


def test_independent_process_race_cannot_exceed_cap(store):
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(_reserve_and_finish, str(store.path)) for _ in range(12)]
        successes = sum(job.result(timeout=20) for job in jobs)
    assert 1 <= successes <= 3
    state = store.status()
    assert successes <= state["used"] <= 3
    if state["unresolved"]:
        # A permitted send whose finish cannot be persisted stays charged and
        # blocks later sends; it is never converted into an unspent request.
        assert state["used"] - successes == state["unresolved"]
        with pytest.raises(q.ProtectedDemoQuotaError):store.ensure_available()
        return
    while store.status()["remaining"]:
        send(store)
    with pytest.raises(q.ProtectedDemoQuotaError, match="exhausted"): send(store)
    assert store.status()["used"] == 3


def test_scope_preserves_quota_failure_even_if_model_runtime_catches_it(store):
    for _ in range(3): send(store)
    with q.request_scope() as scope:
        with pytest.raises(q.ProtectedDemoQuotaError): q.ensure_available()
        assert scope.code == "demo_model_quota_exhausted"
        with pytest.raises(q.ProtectedDemoQuotaError): q.raise_if_request_blocked()
    assert not scope.active
    q.raise_if_request_blocked()


def test_closed_parent_scope_is_not_modified_by_late_background_error():
    import asyncio
    async def run():
        event = asyncio.Event()
        async def late():
            await event.wait()
            q.block_request(q.ProtectedDemoQuotaError("demo_model_quota_exhausted"))
        with q.request_scope() as original:
            child = asyncio.create_task(late())
        event.set()
        await child
        assert original.code is None
        with q.request_scope() as newer: q.raise_if_request_blocked()
        assert newer.code is None
    asyncio.run(run())


def test_ledger_does_not_store_prompt_credentials_or_provider_output(store):
    send(store)
    raw = b"".join(p.read_bytes() for p in store.path.iterdir())
    assert b"controlled payload only" not in raw
    assert H.encode() in raw
    assert not any(x in raw for x in [b"api_key", b"Authorization", b"messages", b"content"])
