"""Fixed E2E accounting tests: temporary files, no service/model connections."""
import asyncio
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import socket
import stat

import pytest

from evaluation import task_budget as module
from evaluation.task_budget import TaskBudget, TaskBudgetError
from evaluation.trial_ledger import TrialLedger
from evaluation.validation_ledger import ValidationLedger


PROTOCOL = "a" * 64
PAYLOAD = "b" * 64
BYTES = "c" * 64
OUTPUT = "d" * 64
PLAN = [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
        for i in range(1, 7)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("budget tests must never contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def make(tmp_path, **kwargs):
    return TaskBudget(tmp_path / "ledger", PROTOCOL, deepcopy(PLAN), **kwargs)


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def rejected(code, function, *args, **kwargs):
    with pytest.raises(TaskBudgetError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == str(caught.value) == code


def reserve(budget, lease, *, case="c1", turn="t1", phase="agent", agent="general"):
    return budget.reserve(case, turn, phase, agent, PAYLOAD, BYTES, lease=lease)


def completed(budget, lease, **kwargs):
    event = reserve(budget, lease, **kwargs)
    budget.finish(event["operation_id"], "received", OUTPUT, lease=lease)
    return event


def test_explicit_new_identity_private_files_and_readonly_status(tmp_path):
    rejected("storage_failure", make, tmp_path)
    assert not (tmp_path / "ledger").exists()
    budget = make(tmp_path, initialize=True)
    assert budget.initial_chain_hash == budget.status()["chain_hash"]
    assert budget.status() == {"attempts": 0, "completed_cases": [], "active_case": None,
                               "pending": [], "paused": False, "chain_hash": budget.initial_chain_hash,
                               "turn_attempts": {}}
    before = snapshot(budget.root)
    assert make(tmp_path).results() == {"operations": {}, "cases": {}}
    assert snapshot(budget.root) == before
    assert stat.S_IMODE(budget.root.stat().st_mode) == 0o700
    for file in budget.root.iterdir():
        assert stat.S_IMODE(file.stat().st_mode) == (0o700 if file.is_dir() else 0o600)
    rejected("protocol", make, tmp_path, initialize=True)
    rejected("protocol", TaskBudget, budget.root, "f" * 64, PLAN)
    with pytest.raises(TaskBudgetError):
        TrialLedger(budget.root, PROTOCOL, ["x"], 24)
    with pytest.raises(TaskBudgetError):
        ValidationLedger(budget.root, PROTOCOL, [str(i) for i in range(36)])


@pytest.mark.parametrize("change", ["count", "turns", "duplicate", "field", "unsafe_id", "empty_turn"])
def test_fixed_plan_validation_never_creates_bad_identity(tmp_path, change):
    plan = deepcopy(PLAN)
    if change == "count": plan.pop()
    elif change == "turns": plan[0]["turn_ids"] = ["t1"]
    elif change == "duplicate": plan[1]["case_id"] = "c1"
    elif change == "field": plan[0]["limit"] = 10000
    elif change == "unsafe_id": plan[0]["case_id"] = "../outside"
    else: plan[0]["turn_ids"] = []
    rejected("configuration", TaskBudget, tmp_path / "bad", PROTOCOL, plan, initialize=True)
    assert not (tmp_path / "bad").exists()


def test_lease_required_cannot_be_forged_and_old_leases_expire(tmp_path):
    budget = make(tmp_path, initialize=True)
    rejected("protocol", budget.start_case, "c1", lease=None)
    with budget.exclusive_run() as lease:
        with pytest.raises(TaskBudgetError):
            with budget.exclusive_run(): pass
        other = make(tmp_path)
        rejected("protocol", other.start_case, "c1", lease=lease)
        forged = module._RunLease(lease.pid, lease.token)
        rejected("protocol", budget.start_case, "c1", lease=forged)
    rejected("protocol", budget.start_case, "c1", lease=lease)
    assert budget.status()["attempts"] == 0


def test_parallel_admission_up_to_three_and_out_of_order_finish(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        first = reserve(budget, lease, agent="technical")
        second = reserve(budget, lease, agent="billing")
        third = reserve(budget, lease, agent="general")
        assert [row["attempt"] for row in (first, second, third)] == [1, 2, 3]
        assert all(row["ordinal"] == 1 for row in (first, second, third))
        assert first["payload_sha256"] == PAYLOAD and first["request_bytes_sha256"] == BYTES
        before = snapshot(budget.root)
        rejected("ledger_busy", reserve, budget, lease)
        assert snapshot(budget.root) == before
        budget.finish(second["operation_id"], "received", OUTPUT, lease=lease)
        fourth = reserve(budget, lease, agent="billing")
        assert fourth["attempt"] == 4 and fourth["ordinal"] == 2
        for event in (third, fourth, first):
            budget.finish(event["operation_id"], "received", OUTPUT, lease=lease)
        assert not budget.status()["pending"] and budget.status()["attempts"] == 4
        rejected("protocol", budget.finish, first["operation_id"], "received", OUTPUT, lease=lease)
        budget.finish_case("c1", OUTPUT, "partial", lease=lease)
        assert budget.results()["cases"]["c1"] == {"artifact_sha256": OUTPUT, "outcome": "partial"}


@pytest.mark.parametrize("window", ["case_started", "pending", "received_unsealed"])
def test_restart_does_not_resume_unfinished_case_or_refund_unknown_attempt(tmp_path, window):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        if window != "case_started":
            event = reserve(budget, lease)
            if window == "received_unsealed": budget.finish(event["operation_id"], "received", OUTPUT, lease=lease)
    before = snapshot(budget.root)
    restarted = make(tmp_path)
    with pytest.raises(TaskBudgetError, match="pending"):
        with restarted.exclusive_run(): pass
    assert restarted.status()["attempts"] == (0 if window == "case_started" else 1)
    assert restarted.status()["active_case"] == "c1"
    assert snapshot(budget.root) == before


@pytest.mark.parametrize("outcome,code", [("execution_error", "network"), ("cancelled", "cancelled"), ("unknown", None)])
def test_execution_failures_pause_without_refund_but_other_pending_can_finish(tmp_path, outcome, code):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        one = reserve(budget, lease)
        two = reserve(budget, lease, agent="billing")
        budget.finish(one["operation_id"], outcome, error_code=code, lease=lease)
        rejected("paused", reserve, budget, lease)
        budget.finish(two["operation_id"], "received", OUTPUT, lease=lease)
        rejected("protocol", budget.finish_case, "c1", OUTPUT, "failed", lease=lease)
    assert budget.status()["attempts"] == 2 and budget.status()["paused"]
    assert not budget.status()["pending"]
    with pytest.raises(TaskBudgetError, match="paused"):
        with make(tmp_path).exclusive_run(): pass


def test_structure_and_business_failure_seal_and_continue_without_resending(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        event = reserve(budget, lease)
        budget.finish(event["operation_id"], "structure_failed", OUTPUT, lease=lease)
        budget.finish_case("c1", OUTPUT, "failed", lease=lease)
    with make(tmp_path).exclusive_run() as _lease:
        pass
    with budget.exclusive_run() as lease:
        rejected("protocol", budget.start_case, "c1", lease=lease)
        budget.start_case("c2", lease=lease)
        completed(budget, lease, case="c2")
        budget.finish_case("c2", OUTPUT, "success", lease=lease)
    assert budget.status()["attempts"] == 2 and budget.status()["completed_cases"] == ["c1", "c2"]


def test_turn_order_and_success_require_all_planned_turns(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        rejected("protocol", budget.start_case, "c2", lease=lease)
        budget.start_case("c1", lease=lease)
        rejected("protocol", reserve, budget, lease, turn="t2")
        pending = reserve(budget, lease)
        rejected("protocol", reserve, budget, lease, turn="t2")
        budget.finish(pending["operation_id"], "received", OUTPUT, lease=lease)
        rejected("protocol", budget.finish_case, "c1", OUTPUT, "success", lease=lease)
        completed(budget, lease, turn="t2")
        rejected("protocol", reserve, budget, lease, turn="t1")
        budget.finish_case("c1", OUTPUT, "success", lease=lease)


def test_phase_and_full_batch_caps_are_persistent(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        for case in PLAN:
            budget.start_case(case["case_id"], lease=lease)
            for turn in case["turn_ids"]:
                for phase, count in module.PHASE_CAPS.items():
                    agent = "general" if phase == "agent" else None
                    for _ in range(count):
                        completed(budget, lease, case=case["case_id"], turn=turn, phase=phase, agent=agent)
                    before = budget.status()["attempts"]
                    rejected("quota", reserve, budget, lease, case=case["case_id"], turn=turn, phase=phase, agent=agent)
                    assert budget.status()["attempts"] == before
            budget.finish_case(case["case_id"], OUTPUT, "success", lease=lease)
    restarted = make(tmp_path)
    assert restarted.status()["attempts"] == 168
    assert list(restarted.status()["turn_attempts"].values()) == [24] * 7
    assert len(restarted.results()["operations"]) == 168
    assert len(restarted.results()["cases"]) == 6
    with restarted.exclusive_run() as lease:
        rejected("protocol", restarted.start_case, "c1", lease=lease)


@pytest.mark.parametrize("field,value", [("phase", "memory"), ("phase", "embedding"), ("agent", "unknown"), ("phase", [])])
def test_unknown_billable_phases_or_agents_are_rejected_before_reserving(tmp_path, field, value):
    budget = make(tmp_path, initialize=True)
    kwargs = {"case": "c1", "turn": "t1", "phase": "agent", "agent": "general"}
    kwargs[field] = value
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        rejected("protocol", reserve, budget, lease, **kwargs)
    assert budget.status()["attempts"] == 0


def _contender(root, queue):
    budget = TaskBudget(Path(root), PROTOCOL, PLAN)
    try:
        with budget.exclusive_run():
            queue.put("unexpected-acquired")
    except TaskBudgetError as error:
        queue.put(error.code)


def test_separate_process_cannot_share_an_active_run_lease(tmp_path):
    budget = make(tmp_path, initialize=True)
    # A fresh interpreter also proves that no inherited flock/FD is required;
    # fork is unsafe once other regression tests have created native threads.
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with budget.exclusive_run():
        process = context.Process(target=_contender, args=(str(budget.root), queue))
        process.start(); process.join(10)
        assert process.exitcode == 0 and queue.get(timeout=1) == "ledger_busy"
    assert budget.status()["attempts"] == 0


@pytest.mark.parametrize("damage", ["journal.jsonl", "identity.json", "claims", "run.lock"])
def test_missing_components_are_never_reinitialized(tmp_path, damage):
    budget = make(tmp_path, initialize=True)
    path = budget.root / damage
    if path.is_dir(): path.rmdir()
    else: path.unlink()
    before = snapshot(budget.root)
    with pytest.raises(TaskBudgetError): make(tmp_path)
    rejected("protocol", make, tmp_path, initialize=True)
    assert snapshot(budget.root) == before and not path.exists()


def test_truncated_journal_and_missing_claims_fail_closed(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        completed(budget, lease)
        budget.finish_case("c1", OUTPUT, "partial", lease=lease)
    journal = budget.root / "journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(TaskBudgetError, match="pending"):
        with make(tmp_path).exclusive_run(): pass
    journal.write_bytes(b"".join(lines[:-2]))
    assert make(tmp_path).status()["pending"]
    journal.write_bytes(b"")
    rejected("protocol", make, tmp_path)


def test_storage_failure_after_claim_persists_blocking_evidence(tmp_path, monkeypatch):
    budget = make(tmp_path, initialize=True)
    original = module._write; calls = 0
    def broken(fd, data):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("private detail must not escape")
        return original(fd, data)
    with budget.exclusive_run() as lease:
        monkeypatch.setattr(module, "_write", broken)
        rejected("storage_failure", budget.start_case, "c1", lease=lease)
    assert (budget.root / "journal.jsonl").read_bytes() == b""
    assert len(list((budget.root / "claims").iterdir())) == 1
    rejected("protocol", make, tmp_path)


@pytest.mark.parametrize("target", ["identity.json", "journal.jsonl", "run.lock"])
def test_symlink_or_hardlink_cannot_replace_private_state(tmp_path, target):
    budget = make(tmp_path, initialize=True)
    source = budget.root / target
    external = tmp_path / "external"
    source.rename(external)
    source.symlink_to(external)
    with pytest.raises(TaskBudgetError): make(tmp_path)
    source.unlink(); os.link(external, source)
    with pytest.raises(TaskBudgetError): make(tmp_path)


def test_copying_ledger_to_another_root_cannot_reuse_identity(tmp_path):
    import shutil
    budget = make(tmp_path, initialize=True)
    copied = tmp_path / "copy"
    shutil.copytree(budget.root, copied)
    rejected("storage_failure", TaskBudget, copied, PROTOCOL, PLAN)


def test_result_snapshots_are_detached_and_received_needs_sealed_hash(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        event = reserve(budget, lease)
        rejected("protocol", budget.finish, event["operation_id"], "received", lease=lease)
        rejected("protocol", budget.finish, event["operation_id"], "received", OUTPUT, "network", lease=lease)
        budget.finish(event["operation_id"], "received", OUTPUT, lease=lease)
        result = budget.results(); result["operations"].clear()
        assert len(budget.results()["operations"]) == 1


def test_cancelled_run_releases_lease_but_not_pending_budget(tmp_path):
    budget = make(tmp_path, initialize=True)
    async def cancelled():
        with budget.exclusive_run() as lease:
            budget.start_case("c1", lease=lease)
            reserve(budget, lease)
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled())
    assert budget._lease is None and budget._lease_fd is None
    before = snapshot(budget.root)
    with pytest.raises(TaskBudgetError, match="pending"):
        with make(tmp_path).exclusive_run(): pass
    assert budget.status()["attempts"] == 1 and snapshot(budget.root) == before


def test_run_lock_replacement_cannot_keep_a_valid_lease(tmp_path):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        lock = budget.root / "run.lock"
        original = lock.read_bytes()
        lock.rename(budget.root / "old-run.lock")
        lock.write_bytes(original); lock.chmod(0o600)
        rejected("storage_failure", reserve, budget, lease)
    assert budget.status()["attempts"] == 0


def test_fsync_failure_never_grants_a_send_or_cleans_up_claim(tmp_path, monkeypatch):
    budget = make(tmp_path, initialize=True)
    with budget.exclusive_run() as lease:
        budget.start_case("c1", lease=lease)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("synthetic private error")))
            rejected("storage_failure", reserve, budget, lease)
    rejected("protocol", make, tmp_path)


@pytest.mark.parametrize("damage", [b"{", b"{\"version\":true}\n", b"{\"x\":1,\"x\":2}\n", b"{\"x\":NaN}\n"])
def test_malformed_or_unknown_journal_is_not_repaired(tmp_path, damage):
    budget = make(tmp_path, initialize=True)
    journal = budget.root / "journal.jsonl"
    journal.write_bytes(damage)
    rejected("protocol", make, tmp_path)
    assert journal.read_bytes() == damage
