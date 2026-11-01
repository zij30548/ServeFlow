"""Offline validation-ledger limits and the inherited persistence contract."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat

import pytest

from evaluation.trial_ledger import TrialLedger, TrialLedgerError
from evaluation.validation_ledger import ValidationLedger


PROTOCOL = "a" * 64
PAYLOAD = "b" * 64
OUTPUT = "c" * 64
OPERATIONS = [f"validation-operation-{index:02d}" for index in range(36)]
LEGACY_OPERATIONS = OPERATIONS[:24]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("ledger tests must not contact any service")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(params=[TrialLedger, ValidationLedger], ids=["development", "validation"])
def ledger_type(request):
    return request.param


def make(tmp_path, ledger_type=ValidationLedger):
    operations = OPERATIONS if ledger_type is ValidationLedger else LEGACY_OPERATIONS
    return ledger_type(tmp_path / "ledger", PROTOCOL, operations)


def assert_code(code, function, *args, **kwargs):
    with pytest.raises(TrialLedgerError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == str(caught.value) == code


def read_events(root):
    return [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]


def claim_path(root, operation_id):
    return root / "claims" / (hashlib.sha256(operation_id.encode()).hexdigest() + ".json")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_validation_only_overrides_constructor():
    assert {name for name in ValidationLedger.__dict__ if not name.startswith("__")} == set()
    for name in ("_locked", "_replay", "_append", "_event_hash", "status",
                 "results", "reserve", "finish", "initial_chain_hash"):
        assert getattr(ValidationLedger, name) is getattr(TrialLedger, name)


def test_validation_identity_genesis_and_private_files(tmp_path):
    operations = list(OPERATIONS)
    ledger = ValidationLedger(tmp_path / "ledger", PROTOCOL.upper(), operations)
    operations.reverse()
    expected = {"version": 1, "protocol_sha256": PROTOCOL,
                "operation_ids": OPERATIONS, "max_requests": 36}
    identity = json.dumps(expected, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    assert (ledger.root / "identity.json").read_bytes() == identity + b"\n"
    assert ledger.operation_ids == tuple(OPERATIONS)
    assert ledger.max_requests == 36
    assert ledger.initial_chain_hash == hashlib.sha256(
        b"serveflow-trial-ledger:identity:v1\n" + identity).hexdigest()
    assert ledger.status() == {"attempts": 0, "completed": [], "pending": [],
                               "paused": False, "chain_hash": ledger.initial_chain_hash}
    with pytest.raises(AttributeError):
        ledger.initial_chain_hash = "0" * 64
    before = snapshot(ledger.root)
    assert ledger.results() == {}
    assert make(tmp_path).initial_chain_hash == ledger.initial_chain_hash
    assert snapshot(ledger.root) == before
    assert stat.S_IMODE(ledger.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((ledger.root / "claims").stat().st_mode) == 0o700
    for filename in ("identity.json", "journal.jsonl"):
        assert stat.S_IMODE((ledger.root / filename).stat().st_mode) == 0o600


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1, 24, 35, 37, 36.0, "36", None])
def test_validation_limit_must_be_exactly_integer_thirty_six(tmp_path, limit):
    root = tmp_path / "ledger"
    assert_code("configuration", ValidationLedger, root, PROTOCOL, OPERATIONS, limit)
    assert not root.exists()


@pytest.mark.parametrize("operations", [
    [], OPERATIONS[:35], OPERATIONS + ["extra"], OPERATIONS[:-1] + [OPERATIONS[0]],
    OPERATIONS[:-1] + [""], OPERATIONS[:-1] + [True],
    OPERATIONS[:-1] + ["\ud800"], tuple(OPERATIONS), "not-list",
])
def test_validation_requires_thirty_six_unique_valid_operation_ids(tmp_path, operations):
    root = tmp_path / "ledger"
    assert_code("configuration", ValidationLedger, root, PROTOCOL, operations)
    assert not root.exists()


def test_oversized_validation_identity_is_rejected_before_creation(tmp_path):
    root = tmp_path / "ledger"
    operations = OPERATIONS[:-1] + ["x" * (1024 * 1024)]
    assert_code("configuration", ValidationLedger, root, PROTOCOL, operations)
    assert not root.exists()


def test_legacy_class_still_rejects_thirty_six_requests(tmp_path):
    assert_code("configuration", TrialLedger, tmp_path / "legacy", PROTOCOL, OPERATIONS, 36)
    assert not (tmp_path / "legacy").exists()


def test_development_and_validation_identities_cannot_be_read_by_other_class(tmp_path):
    development = TrialLedger(tmp_path / "development", PROTOCOL, OPERATIONS)
    validation = ValidationLedger(tmp_path / "validation", PROTOCOL, OPERATIONS)
    for ledger in (development, validation):
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        ledger.finish(OPERATIONS[0], "received", output_sha256=OUTPUT)
    old_contents = snapshot(development.root)
    new_contents = snapshot(validation.root)
    assert_code("protocol", ValidationLedger, development.root, PROTOCOL, OPERATIONS)
    assert_code("protocol", TrialLedger, validation.root, PROTOCOL, OPERATIONS)
    assert snapshot(development.root) == old_contents
    assert snapshot(validation.root) == new_contents
    assert development.initial_chain_hash != validation.initial_chain_hash
    assert development.status()["attempts"] == validation.status()["attempts"] == 1


def test_validation_order_and_protocol_cannot_change_on_restart(tmp_path):
    ledger = make(tmp_path)
    before = snapshot(ledger.root)
    assert_code("protocol", ledger.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", ValidationLedger, ledger.root, PROTOCOL, list(reversed(OPERATIONS)))
    assert_code("protocol", ValidationLedger, ledger.root, "d" * 64, OPERATIONS)
    assert snapshot(ledger.root) == before


def test_each_fixed_operation_can_be_reserved_once_with_exact_limit(tmp_path, ledger_type):
    ledger = make(tmp_path, ledger_type)
    operations = list(ledger.operation_ids)
    for index, operation_id in enumerate(operations, 1):
        restarted = make(tmp_path, ledger_type)
        restarted.reserve(operation_id, PAYLOAD)
        restarted.finish(operation_id, "received", output_sha256=OUTPUT)
        assert restarted.status()["attempts"] == index
    restarted = make(tmp_path, ledger_type)
    before = snapshot(ledger.root)
    assert restarted.status()["completed"] == operations
    assert restarted.status()["pending"] == []
    assert_code("quota", restarted.reserve, operations[0], PAYLOAD)
    assert_code("quota", restarted.reserve, "extra-operation", PAYLOAD)
    assert_code("protocol", restarted.finish, operations[-1], "received")
    assert len(list((ledger.root / "claims").iterdir())) == len(operations)
    assert len(read_events(ledger.root)) == 2 * len(operations)
    assert snapshot(ledger.root) == before


def test_restart_retains_pending_and_cannot_resend_or_skip(tmp_path, ledger_type):
    ledger = make(tmp_path, ledger_type)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    restarted = make(tmp_path, ledger_type)
    assert restarted.status()["pending"] == [OPERATIONS[0]]
    before = snapshot(ledger.root)
    for operation_id in OPERATIONS[:2]:
        assert_code("pending", restarted.reserve, operation_id, PAYLOAD)
    assert snapshot(ledger.root) == before
    restarted.finish(OPERATIONS[0], "received", output_sha256=OUTPUT)
    assert_code("protocol", restarted.reserve, OPERATIONS[0], PAYLOAD)
    restarted.reserve(OPERATIONS[1], PAYLOAD)
    assert restarted.status()["attempts"] == 2


@pytest.mark.parametrize("outcome", ["execution_error", "cancelled", "unknown"])
def test_uncertain_outcomes_remain_paused_after_restart(tmp_path, ledger_type, outcome):
    ledger = make(tmp_path, ledger_type)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], outcome, error_code="network")
    restarted = make(tmp_path, ledger_type)
    assert restarted.status()["paused"] is True
    before = snapshot(ledger.root)
    assert_code("paused", restarted.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", restarted.finish, OPERATIONS[0], "received")
    assert snapshot(ledger.root) == before


def test_verified_results_are_copies_and_bad_chain_blocks_reads_and_sends(tmp_path, ledger_type):
    ledger = make(tmp_path, ledger_type)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], "structure_failed", output_sha256=OUTPUT, error_code="protocol")
    results = ledger.results()
    results[OPERATIONS[0]]["output_sha256"] = "0" * 64
    assert ledger.results()[OPERATIONS[0]] == {
        "outcome": "structure_failed", "output_sha256": OUTPUT, "error_code": "protocol"}
    events = read_events(ledger.root)
    events[-1]["output_sha256"] = "d" * 64
    path = ledger.root / "journal.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    before = snapshot(ledger.root)
    assert_code("protocol", ledger.results)
    assert_code("protocol", ledger.reserve, OPERATIONS[1], PAYLOAD)
    assert_code("protocol", make, tmp_path, ledger_type)
    assert snapshot(ledger.root) == before


@pytest.mark.parametrize("filename", ["identity.json", "journal.jsonl", "claims"])
def test_missing_components_are_never_recreated(tmp_path, ledger_type, filename):
    ledger = make(tmp_path, ledger_type)
    path = ledger.root / filename
    path.rmdir() if path.is_dir() else path.unlink()
    before = snapshot(ledger.root)
    with pytest.raises(TrialLedgerError):
        make(tmp_path, ledger_type)
    assert not path.exists()
    assert snapshot(ledger.root) == before


@pytest.mark.parametrize("mutation", ["missing", "orphan", "tampered", "truncated_journal"])
def test_claim_evidence_cannot_be_reset_or_ignored(tmp_path, ledger_type, mutation):
    ledger = make(tmp_path, ledger_type)
    ledger.reserve(OPERATIONS[0], PAYLOAD)
    ledger.finish(OPERATIONS[0], "received")
    path = claim_path(ledger.root, OPERATIONS[0])
    if mutation == "missing":
        path.unlink()
    elif mutation == "orphan":
        orphan = claim_path(ledger.root, OPERATIONS[1])
        orphan.write_bytes(path.read_bytes())
        orphan.chmod(0o600)
    elif mutation == "tampered":
        claim = json.loads(path.read_text())
        claim["payload_sha256"] = "d" * 64
        path.write_text(json.dumps(claim))
    else:
        (ledger.root / "journal.jsonl").write_bytes(b"")
    before = snapshot(ledger.root)
    assert_code("protocol", make, tmp_path, ledger_type)
    assert_code("protocol", ledger.reserve, OPERATIONS[0], PAYLOAD)
    assert snapshot(ledger.root) == before


@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, None, True])
def test_invalid_hashes_do_not_create_or_mutate_validation_ledger(tmp_path, value):
    root = tmp_path / "bad"
    assert_code("configuration", ValidationLedger, root, value, OPERATIONS)
    assert not root.exists()
    ledger = make(tmp_path)
    before = snapshot(ledger.root)
    assert_code("configuration", ledger.reserve, OPERATIONS[0], value)
    assert snapshot(ledger.root) == before


def test_symlink_root_is_refused_without_touching_target(tmp_path, ledger_type):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (tmp_path / "ledger").symlink_to(target, target_is_directory=True)
    assert_code("storage_failure", make, tmp_path, ledger_type)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("relative", ["identity.json", "journal.jsonl", "claims", "claim"])
def test_symlink_components_are_refused_without_touching_target(tmp_path, relative):
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


def _competing_reservation(root, start, results):
    start.wait(10)
    try:
        ledger = ValidationLedger(Path(root), PROTOCOL, OPERATIONS)
        ledger.reserve(OPERATIONS[0], PAYLOAD)
        results.put("reserved")
    except TrialLedgerError as error:
        results.put(error.code)


def test_concurrent_validation_processes_grant_only_one_reservation(tmp_path):
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
        assert len(os.listdir(ledger.root / "claims")) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
