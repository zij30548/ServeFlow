"""Offline ledger safety, including processes competing for one send allowance."""
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat
import time

import pytest

from evaluation import trial_ledger as ledger_module
from evaluation.trial_ledger import TrialLedger, TrialLedgerError


PROTOCOL = "a" * 64
PAYLOAD = "b" * 64
OUTPUT = "c" * 64
OPERATIONS = [f"operation-{index:02d}" for index in range(24)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("ledger tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def make(tmp_path, *, max_requests=24, operations=None):
    return TrialLedger(tmp_path / "ledger", PROTOCOL,
                       OPERATIONS if operations is None else operations, max_requests)


def read_events(root):
    return [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]


def claim_path(root, operation_id):
    return root / "claims" / (hashlib.sha256(operation_id.encode()).hexdigest() + ".json")


def assert_code(code, function, *args, **kwargs):
    with pytest.raises(TrialLedgerError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code
    assert str(caught.value) == code


def test_new_ledger_is_private_and_read_status_never_appends(tmp_path):
    ledger = make(tmp_path)
    expected = {"attempts": 0, "completed": [], "pending": [], "paused": False,
                "chain_hash": ledger.status()["chain_hash"]}
    assert ledger.status() == expected
    assert len(expected["chain_hash"]) == 64
    assert stat.S_IMODE(ledger.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((ledger.root / "claims").stat().st_mode) == 0o700
    for filename in ("identity.json", "journal.jsonl"):
        assert stat.S_IMODE((ledger.root / filename).stat().st_mode) == 0o600
    assert (ledger.root / "journal.jsonl").read_bytes() == b""


def test_restart_preserves_reservation_and_finishes_once(tmp_path):
    ledger = make(tmp_path)
    attempt = ledger.reserve(OPERATIONS[0], PAYLOAD)
    assert attempt["attempt"] == attempt["seq"] == 1
    assert attempt["operation_id"] == OPERATIONS[0]
    assert attempt["payload_sha256"] == PAYLOAD
    assert stat.S_IMODE(claim_path(ledger.root, OPERATIONS[0]).stat().st_mode) == 0o600
    before = (ledger.root / "journal.jsonl").read_bytes()
    restarted = make(tmp_path)
    assert restarted.status()["pending"] == [OPERATIONS[0]]
    assert_code("pending", restarted.reserve, OPERATIONS[1], PAYLOAD)
    event = restarted.finish(OPERATIONS[0], "received", output_sha256=OUTPUT)
    assert event["seq"] == 2
    assert event["previous_hash"] == attempt["chain_hash"]
    assert (ledger.root / "journal.jsonl").read_bytes().startswith(before)
    assert restarted.status() == {
        "attempts": 1, "completed": [OPERATIONS[0]], "pending": [],
        "paused": False, "chain_hash": event["chain_hash"],
    }
    assert make(tmp_path).status() == restarted.status()
    assert_code("protocol", restarted.finish, OPERATIONS[0], "received")
    assert_code("protocol", restarted.reserve, OPERATIONS[0], PAYLOAD)
    restarted.reserve(OPERATIONS[1], PAYLOAD)


def test_initial_chain_hash_is_stable_read_only_and_protocol_bound(tmp_path):
    ledger = make(tmp_path)
    initial = ledger.initial_chain_hash
    assert initial == ledger.status()["chain_hash"]
    assert initial != "0" * 64
    with pytest.raises(AttributeError):
        ledger.initial_chain_hash = "0" * 64
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], "received", output_sha256=OUTPUT)
    assert ledger.status()["chain_hash"] != initial
    assert ledger.initial_chain_hash == make(tmp_path).initial_chain_hash == initial
    other = TrialLedger(tmp_path / "other", "d" * 64, OPERATIONS)
    assert other.initial_chain_hash != initial


def test_results_are_verified_independent_copies_and_tampering_is_refused(tmp_path):
    ledger = make(tmp_path)
    assert ledger.results() == {}
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    assert ledger.results() == {}
    ledger.finish(OPERATIONS[0], "structure_failed", output_sha256=OUTPUT, error_code="protocol")
    expected = {OPERATIONS[0]: {"output_sha256": OUTPUT,
                                "outcome": "structure_failed", "error_code": "protocol"}}
    results = ledger.results()
    assert results == make(tmp_path).results() == expected
    results[OPERATIONS[0]]["output_sha256"] = "0" * 64
    results[OPERATIONS[1]] = {"outcome": "received"}
    assert ledger.results() == expected
    assert set(ledger.status()) == {"attempts", "completed", "pending", "paused", "chain_hash"}
    events = read_events(ledger.root)
    events[-1]["output_sha256"] = "d" * 64
    path = ledger.root / "journal.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert_code("protocol", ledger.results)


@pytest.mark.parametrize("outcome", ["received", "structure_failed"])
def test_response_outcomes_allow_next_attempt_even_if_semantic_score_is_wrong(tmp_path, outcome):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], outcome, output_sha256=OUTPUT)
    ledger.reserve(OPERATIONS[1], PAYLOAD)
    assert ledger.status()["attempts"] == 2


@pytest.mark.parametrize("outcome", ["execution_error", "cancelled", "unknown"])
def test_uncertain_or_failed_outcomes_pause_persistently(tmp_path, outcome):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], outcome, error_code="cancelled" if outcome == "cancelled" else "network")
    restarted = make(tmp_path)
    assert restarted.status()["paused"] is True
    assert restarted.status()["completed"] == [OPERATIONS[0]]
    assert_code("paused", restarted.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", restarted.finish, OPERATIONS[0], "received")


def test_strict_order_and_twenty_four_attempt_upper_bound(tmp_path):
    ledger = make(tmp_path, operations=OPERATIONS + ["extra-operation"])
    assert_code("protocol", ledger.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", ledger.finish, OPERATIONS[0], "received")
    for index, operation_id in enumerate(OPERATIONS, 1):
        ledger.reserve(operation_id, PAYLOAD)
        ledger.finish(operation_id, "received")
        assert ledger.status()["attempts"] == index
    assert_code("quota", ledger.reserve, "extra-operation", PAYLOAD)
    assert len(list((ledger.root / "claims").iterdir())) == 24
    assert len(read_events(ledger.root)) == 48


def test_smaller_explicit_limit_is_not_expandable(tmp_path):
    ledger = make(tmp_path, max_requests=1)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], "received")
    assert_code("quota", ledger.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", make, tmp_path, max_requests=24)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 25, 1.0, "24", None])
def test_invalid_limits_never_create_any_ledger(tmp_path, limit):
    assert_code("configuration", make, tmp_path, max_requests=limit)
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("operations", [["duplicate", "duplicate"], [""], [True], "not-list", ["\ud800"]])
def test_invalid_operation_id_configuration_is_safe(tmp_path, operations):
    assert_code("configuration", make, tmp_path, operations=operations)
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, "a" * 65, None, True, []])
def test_bad_hashes_are_rejected_without_changes(tmp_path, value):
    assert_code("configuration", TrialLedger, tmp_path / "bad", value, OPERATIONS)
    assert not (tmp_path / "bad").exists()
    ledger = make(tmp_path)
    assert_code("configuration", ledger.reserve, OPERATIONS[0], value)
    assert ledger.status()["attempts"] == 0
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    if value is None:  # Output hash is optional; protocol/payload hashes are not.
        ledger.finish(OPERATIONS[0], "received", output_sha256=None)
    else:
        assert_code("configuration", ledger.finish, OPERATIONS[0], "received", output_sha256=value)
        assert ledger.status()["pending"] == [OPERATIONS[0]]


@pytest.mark.parametrize("code", ["secret-user-content", {}, True])
def test_raw_error_codes_are_never_persisted_or_exposed(tmp_path, code):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    assert_code("configuration", ledger.finish, OPERATIONS[0], "execution_error", error_code=code)
    assert "secret-user-content" not in (ledger.root / "journal.jsonl").read_text()


def test_all_documented_error_codes_can_be_persisted(tmp_path):
    for index, code in enumerate(sorted(ledger_module.ERROR_CODES)):
        root = tmp_path / str(index)
        ledger = TrialLedger(root, PROTOCOL, ["one"])
        ledger.reserve("one", PAYLOAD)
        ledger.finish("one", "execution_error", error_code=code)
        assert TrialLedger(root, PROTOCOL, ["one"]).status()["paused"]


def test_identity_must_match_protocol_order_and_version(tmp_path):
    ledger = make(tmp_path)
    original = (ledger.root / "identity.json").read_bytes()
    assert_code("protocol", TrialLedger, ledger.root, "d" * 64, OPERATIONS)
    assert_code("protocol", TrialLedger, ledger.root, PROTOCOL, list(reversed(OPERATIONS)))
    assert (ledger.root / "identity.json").read_bytes() == original
    identity = json.loads(original)
    identity["version"] = 2
    (ledger.root / "identity.json").write_text(json.dumps(identity))
    assert_code("version", make, tmp_path)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(version=True),
    lambda value: value.update(max_requests=True),
    lambda value: value.update(extra="forbidden"),
    lambda value: value.update(operation_ids=["different"]),
])
def test_identity_schema_and_types_are_strict(tmp_path, mutation):
    ledger = make(tmp_path)
    path = ledger.root / "identity.json"
    identity = json.loads(path.read_text())
    mutation(identity)
    path.write_text(json.dumps(identity))
    assert_code("protocol", ledger.status)


@pytest.mark.parametrize("outcome", ["correct", "failed", None, [], True])
def test_only_documented_finish_outcomes_are_accepted(tmp_path, outcome):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    assert_code("configuration", ledger.finish, OPERATIONS[0], outcome)
    assert ledger.status()["pending"] == [OPERATIONS[0]]


@pytest.mark.parametrize("filename", ["identity.json", "journal.jsonl", "claims"])
def test_missing_existing_component_is_never_rebuilt(tmp_path, filename):
    ledger = make(tmp_path)
    path = ledger.root / filename
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(TrialLedgerError):
        make(tmp_path)
    assert not path.exists()


@pytest.mark.parametrize("body", [
    b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n',
    b'{"x":1e999}\n', b'{"x":', b'\n', b'[]\n', b'\xff\n',
])
def test_corrupt_journal_blocks_without_repair(tmp_path, body):
    ledger = make(tmp_path)
    path = ledger.root / "journal.jsonl"
    path.write_bytes(body)
    assert_code("protocol", ledger.status)
    assert_code("protocol", ledger.reserve, OPERATIONS[0], PAYLOAD)
    assert path.read_bytes() == body


@pytest.mark.parametrize("mutation", [
    lambda event: event.update(extra="forbidden"),
    lambda event: event.update(seq=True),
    lambda event: event.update(attempt=False),
    lambda event: event.update(payload_sha256="d" * 64),
    lambda event: event.update(operation_id=OPERATIONS[1]),
    lambda event: event.update(previous_hash="e" * 64),
    lambda event: event.update(protocol_sha256="f" * 64),
    lambda event: event.update(outcome="received"),
])
def test_tampering_and_extra_fields_fail_closed(tmp_path, mutation):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    event = read_events(ledger.root)[0]
    mutation(event)
    path = ledger.root / "journal.jsonl"
    path.write_text(json.dumps(event) + "\n")
    assert_code("protocol", ledger.status)


def test_duplicate_json_key_rejected_even_if_value_is_unchanged(tmp_path):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    path = ledger.root / "journal.jsonl"
    data = path.read_text()
    path.write_text('{"version":1,' + data[1:])
    assert_code("protocol", ledger.status)


def test_rehashed_bad_transition_still_fails(tmp_path):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], "received")
    events = read_events(ledger.root)
    duplicate = dict(events[-1], seq=3, previous_hash=events[-1]["chain_hash"])
    duplicate.pop("chain_hash")
    duplicate["chain_hash"] = ledger._event_hash(duplicate)
    with (ledger.root / "journal.jsonl").open("a") as file:
        file.write(json.dumps(duplicate) + "\n")
    assert_code("protocol", ledger.status)


@pytest.mark.parametrize("keep", [0, 1, 2, 3])
def test_complete_journal_suffix_removal_is_detected_by_claims_or_pending(tmp_path, keep):
    ledger = make(tmp_path)
    for operation_id in OPERATIONS[:2]:
        ledger.reserve(operation_id, PAYLOAD)
        ledger.finish(operation_id, "received")
    path = ledger.root / "journal.jsonl"
    prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:keep])
    path.write_bytes(prefix)
    if keep < 3:
        assert_code("protocol", ledger.status)
        assert_code("protocol", ledger.reserve, OPERATIONS[0], PAYLOAD)
    else:
        assert ledger.status()["pending"] == [OPERATIONS[1]]
        assert_code("pending", ledger.reserve, OPERATIONS[2], PAYLOAD)
    assert path.read_bytes() == prefix


@pytest.mark.parametrize("mutation", ["missing", "orphan", "tampered", "duplicate", "extra"])
def test_claim_set_and_strict_claim_content_must_match_journal(tmp_path, mutation):
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    path = claim_path(ledger.root, OPERATIONS[0])
    if mutation == "missing":
        path.unlink()
    elif mutation == "orphan":
        claim_path(ledger.root, OPERATIONS[1]).write_bytes(path.read_bytes())
    elif mutation == "duplicate":
        path.write_text('{"version":1,' + path.read_text()[1:])
    else:
        content = json.loads(path.read_text())
        content["payload_sha256" if mutation == "tampered" else "extra"] = "d" * 64
        path.write_text(json.dumps(content))
    assert_code("protocol", ledger.status)


def test_busy_lock_is_rejected_without_waiting(tmp_path):
    ledger = make(tmp_path)
    with (ledger.root / "identity.json").open("rb") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        assert_code("ledger_busy", ledger.status)
        assert_code("ledger_busy", ledger.reserve, OPERATIONS[0], PAYLOAD)
        assert time.monotonic() - start < 1


def _competing_reservation(root, start, results):
    start.wait(10)
    try:
        ledger = TrialLedger(Path(root), PROTOCOL, OPERATIONS)
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        results.put("reserved")
    except TrialLedgerError as error:
        results.put(error.code)


def test_competing_processes_cannot_reserve_more_than_one_send(tmp_path):
    ledger = make(tmp_path)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [context.Process(target=_competing_reservation,
                                 args=(str(ledger.root), start, results)) for _ in range(6)]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        assert outcomes.count("reserved") == 1
        assert set(outcomes) <= {"reserved", "pending", "ledger_busy"}
        assert ledger.status()["attempts"] == 1
        assert len(read_events(ledger.root)) == 1
        assert len(list((ledger.root / "claims").iterdir())) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_reserve_fsync_failure_never_returns_send_permission(tmp_path, monkeypatch, failure_call):
    ledger = make(tmp_path)
    original = os.fsync
    calls = 0

    def fail(fd):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("sensitive-path-and-exception")
        original(fd)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail)
        assert_code("storage_failure", ledger.reserve, OPERATIONS[0], PAYLOAD)
    if failure_call < 3:
        assert_code("protocol", ledger.status)  # Orphan receipt is not repaired.
        assert_code("protocol", make, tmp_path)
    else:
        assert ledger.status()["pending"] == [OPERATIONS[0]]
        assert_code("pending", ledger.reserve, OPERATIONS[1], PAYLOAD)


def test_initialization_and_finish_fsync_failures_are_safe(tmp_path, monkeypatch):
    def fail(_fd):
        raise OSError("sensitive-path-and-exception")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail)
        assert_code("storage_failure", TrialLedger, tmp_path / "new", PROTOCOL, OPERATIONS)
    ledger = make(tmp_path)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail)
        assert_code("storage_failure", ledger.finish, OPERATIONS[0], "unknown")
    assert make(tmp_path).status()["paused"]


def test_short_writes_are_completed_and_zero_writes_fail_closed(tmp_path, monkeypatch):
    ledger = make(tmp_path)
    original = os.write

    def short(fd, data):
        return original(fd, data[:7])

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "write", short)
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        ledger.finish(OPERATIONS[0], "received")
    assert ledger.status()["completed"] == [OPERATIONS[0]]
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "write", lambda *_args: 0)
        assert_code("storage_failure", ledger.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", ledger.status)


def test_claim_directory_replacement_during_write_prevents_send_permission(tmp_path, monkeypatch):
    ledger = make(tmp_path)
    original = os.fsync
    directory = ledger.root / "claims"
    calls = 0

    def replace(fd):
        nonlocal calls
        calls += 1
        original(fd)
        if calls == 2:
            directory.rename(ledger.root / "retained-claims")
            directory.mkdir(mode=0o700)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", replace)
        assert_code("storage_failure", ledger.reserve, OPERATIONS[0], PAYLOAD)
    assert_code("protocol", ledger.status)


@pytest.mark.parametrize("relative", ["identity.json", "journal.jsonl", "claims", "claim"])
def test_symlinks_are_refused_without_following_or_changing_target(tmp_path, relative):
    ledger = make(tmp_path)
    if relative == "claim":
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        path = claim_path(ledger.root, OPERATIONS[0])
    else:
        path = ledger.root / relative
    target = tmp_path / "target"
    if path.is_dir():
        path.rmdir()
        target.mkdir(mode=0o700)
    else:
        target.write_bytes(path.read_bytes())
        target.chmod(0o600)
        path.unlink()
    path.symlink_to(target)
    assert_code("storage_failure", ledger.status)
    assert target.exists()


def test_root_symlink_is_refused(tmp_path):
    root = tmp_path / "target"
    root.mkdir(mode=0o700)
    (tmp_path / "ledger").symlink_to(root, target_is_directory=True)
    assert_code("storage_failure", make, tmp_path)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("relative", ["identity.json", "journal.jsonl", "claim"])
def test_hardlinked_files_are_refused(tmp_path, relative):
    ledger = make(tmp_path)
    if relative == "claim":
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        path = claim_path(ledger.root, OPERATIONS[0])
    else:
        path = ledger.root / relative
    os.link(path, tmp_path / "linked")
    assert_code("storage_failure", ledger.status)


@pytest.mark.parametrize("relative", ["identity.json", "journal.jsonl", "claim"])
def test_fifo_is_refused_without_blocking(tmp_path, relative):
    ledger = make(tmp_path)
    if relative == "claim":
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        path = claim_path(ledger.root, OPERATIONS[0])
    else:
        path = ledger.root / relative
    path.unlink()
    os.mkfifo(path, 0o600)
    started = time.monotonic()
    assert_code("storage_failure", ledger.status)
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("relative,mode", [
    (".", 0o755), ("identity.json", 0o644), ("journal.jsonl", 0o640),
    ("claims", 0o755), ("claim", 0o644),
])
def test_existing_loose_permissions_are_refused_without_chmod(tmp_path, relative, mode):
    ledger = make(tmp_path)
    if relative == "claim":
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        path = claim_path(ledger.root, OPERATIONS[0])
    else:
        path = ledger.root / relative
    path.chmod(mode)
    assert_code("storage_failure", ledger.status)
    assert stat.S_IMODE(path.stat().st_mode) == mode
