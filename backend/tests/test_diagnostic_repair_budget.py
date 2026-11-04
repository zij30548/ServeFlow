"""Fixed recheck accounting over real temporary files; no model, DB or sockets."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import socket
import threading

import pytest

from evaluation import diagnostic_repair_budget as module, task_budget as old
from evaluation import prospective_budget


PROTOCOL, PAYLOAD, WIRE, OUTPUT = (value * 64 for value in "abcd")
FIRST, SECOND = module.CASE_IDS[:2]


def forbidden(*args, **kwargs):
    raise AssertionError("diagnostic accounting tests forbid external services")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def budget(tmp_path, *, send_limit=module.MAX_REQUESTS):
    return module.DiagnosticRepairBudget(tmp_path / "ledger", PROTOCOL, module.fixed_plan(),
                                         initialize=True, send_limit=send_limit)


def reopen(root, *, send_limit=module.MAX_REQUESTS):
    return module.DiagnosticRepairBudget(root, PROTOCOL, module.fixed_plan(), send_limit=send_limit)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def reserve(instance, lease, *, case=FIRST, phase="intent", agent=None):
    return instance.reserve(case, "turn-1", phase, agent, PAYLOAD, WIRE, lease=lease)


def completed(instance, lease, **kwargs):
    event = reserve(instance, lease, **kwargs)
    instance.finish(event["operation_id"], "received", OUTPUT, lease=lease)
    return event


def test_fixed_six_turn_order_and_separate_mathematical_identity():
    plan = module.fixed_plan()
    assert [row["case_id"] for row in plan] == ["D-p07", "D-n01", "D-p01", "D-n02", "D-p09", "D-n03"]
    assert len(plan) == 6 and all(row["turn_ids"] == ["turn-1"] for row in plan)
    assert module.PHASE_CAPS == {"intent": 1, "agent": 15, "composer": 1, "rag_rewrite": 3, "rag_rerank": 3}
    assert module.PER_TURN == 23 and module.MAX_REQUESTS == 138
    assert module.MAX_IN_FLIGHT == 3 and module.LIMITS == {
        "cases": 6, "turns": 6, "physical_requests": 138, "per_turn": 23,
        "in_flight": 3, "phase_caps": module.PHASE_CAPS, "judge_allowed": False,
    }
    plan[0]["turn_ids"].append("turn-2")
    assert module.fixed_plan()[0]["turn_ids"] == ["turn-1"]
    assert old.PER_TURN == 24 and old.MAX_REQUESTS == 168 and old.PHASE_CAPS["judge"] == 1
    assert prospective_budget.MAX_REQUESTS == 576 and len(prospective_budget.fixed_plan()) == 36


@pytest.mark.parametrize("damage", ["order", "missing", "duplicate", "extra", "turn", "arm", "field", "tuple"])
def test_only_exact_six_single_turn_plan_can_initialize(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "order": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "extra": plan.append({"case_id": "D-n04", "turn_ids": ["turn-1"]})
    elif damage == "turn": plan[0]["turn_ids"].append("turn-2")
    elif damage == "arm": plan[0]["case_id"] = "A-p07"
    elif damage == "field": plan[0]["allowance"] = 138
    else: plan = tuple(plan)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.DiagnosticRepairBudget(tmp_path / "bad", PROTOCOL, plan, initialize=True)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("limit", [0, -1, 139, True, False, 1.0, "1", None])
def test_lower_authorization_ceiling_is_strict_integer_not_new_identity(tmp_path, limit):
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        budget(tmp_path, send_limit=limit)
    assert not (tmp_path / "ledger").exists()


def test_private_initialization_and_readonly_reopen_do_not_change_bytes(tmp_path):
    instance = budget(tmp_path, send_limit=1)
    before = snapshot(instance.root)
    reader = reopen(instance.root)
    assert instance.max_requests == reader.max_requests == 138
    assert instance.send_limit == 1 and reader.send_limit == 138
    assert instance.initial_chain_hash == reader.initial_chain_hash
    assert reader.status()["attempts"] == 0 and reader.results() == {"operations": {}, "cases": {}}
    assert snapshot(instance.root) == before
    assert instance.root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600) for p in instance.root.iterdir())
    with pytest.raises(old.TaskBudgetError, match="protocol"):
        module.DiagnosticRepairBudget(instance.root, PROTOCOL, module.fixed_plan(), initialize=True)
    assert snapshot(instance.root) == before


def test_every_case_can_reach_its_bound_but_never_exceed_138(tmp_path):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        for row in module.fixed_plan():
            case = row["case_id"]
            instance.start_case(case, lease=lease)
            for phase, count in module.PHASE_CAPS.items():
                agent = "technical" if phase == "agent" else None
                for _ in range(count):
                    completed(instance, lease, case=case, phase=phase, agent=agent)
                before = snapshot(instance.root)
                with pytest.raises(old.TaskBudgetError, match="quota"):
                    reserve(instance, lease, case=case, phase=phase, agent=agent)
                assert snapshot(instance.root) == before
            instance.finish_case(case, OUTPUT, lease=lease)
    restarted = reopen(instance.root)
    state = restarted.status()
    assert state["attempts"] == 138 and not state["pending"] and not state["paused"]
    assert state["completed_cases"] == list(module.CASE_IDS)
    assert all(count == 23 for count in state["turn_attempts"].values())
    before = snapshot(instance.root)
    with restarted.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            restarted.start_case(FIRST, lease=lease)
    assert snapshot(instance.root) == before


@pytest.mark.parametrize("kind", ["global", "authorization", "turn", "phase"])
def test_stricter_bound_checked_in_actual_reservation_replay_path(tmp_path, kind):
    instance = budget(tmp_path, send_limit=7 if kind == "authorization" else 138)
    state = instance._empty()
    state["active_case"] = FIRST
    if kind == "global": state["attempts"] = 138
    elif kind == "authorization": state["attempts"] = 7
    elif kind == "turn": state["turn_attempts"][(FIRST, "turn-1")] = 23
    else: state["phase_counts"][(FIRST, "turn-1", "agent")] = 15
    with pytest.raises(old.TaskBudgetError, match="quota"):
        instance._reserve_check(state, FIRST, "turn-1", "agent", "technical")


def test_three_pending_requests_and_out_of_order_completion_share_one_limit(tmp_path):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        claims = [reserve(instance, lease, phase="agent", agent=role)
                  for role in ("general", "billing", "technical")]
        assert [row["attempt"] for row in claims] == [1, 2, 3]
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            reserve(instance, lease)
        for claim in reversed(claims):
            instance.finish(claim["operation_id"], "received", OUTPUT, lease=lease)
        assert instance.status()["attempts"] == 3 and not instance.status()["pending"]
        instance.finish_case(FIRST, OUTPUT, lease=lease)


def test_one_send_authorization_is_atomic_under_thread_contention_and_restart(tmp_path):
    instance = budget(tmp_path, send_limit=1)
    barrier = threading.Barrier(2)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)

        def contend(role):
            barrier.wait(timeout=2)
            try:
                return reserve(instance, lease, phase="agent", agent=role)
            except old.TaskBudgetError as error:
                return error.code

        with ThreadPoolExecutor(max_workers=2) as workers:
            outcomes = list(workers.map(contend, ("technical", "billing")))
        granted = [item for item in outcomes if isinstance(item, dict)]
        assert len(granted) == 1
        assert [item for item in outcomes if isinstance(item, str)][0] in {"quota", "ledger_busy"}
        instance.finish(granted[0]["operation_id"], "received", OUTPUT, lease=lease)
        instance.finish_case(FIRST, OUTPUT, lease=lease)
    restarted = reopen(instance.root, send_limit=1)
    with restarted.exclusive_run() as lease:
        restarted.start_case(SECOND, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            reserve(restarted, lease, case=SECOND)
    assert restarted.status()["attempts"] == 1


def competing_process(root, result):
    # Spawned children do not inherit pytest's monkeypatch fixture.
    socket.socket.connect = socket.socket.connect_ex = forbidden
    socket.create_connection = socket.getaddrinfo = forbidden
    try:
        instance = reopen(Path(root), send_limit=1)
        with instance.exclusive_run() as lease:
            instance.start_case(FIRST, lease=lease)
            completed(instance, lease)
            result.put("incorrectly-reserved")
    except old.TaskBudgetError as error:
        result.put(error.code)


def test_independent_process_cannot_acquire_existing_run_or_double_reserve(tmp_path):
    instance = budget(tmp_path, send_limit=1)
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    with instance.exclusive_run() as lease:
        child = ctx.Process(target=competing_process, args=(str(instance.root), result))
        child.start()
        child.join(10)
        try:
            assert not child.is_alive() and child.exitcode == 0
            assert result.get(timeout=1) == "ledger_busy"
            instance.start_case(FIRST, lease=lease)
            completed(instance, lease)
            instance.finish_case(FIRST, OUTPUT, lease=lease)
        finally:
            if child.is_alive():
                child.terminate()
                child.join(5)
            result.close()
            result.join_thread()
    assert reopen(instance.root, send_limit=1).status()["attempts"] == 1


@pytest.mark.parametrize("phase,agent", [("judge", None), ("memory", None), ("embedding", None),
    ("agent", None), ("agent", "unknown"), ("intent", "technical"), ([], None)])
def test_unplanned_phase_or_role_gets_no_claim(tmp_path, phase, agent):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        before = snapshot(instance.root)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reserve(instance, lease, phase=phase, agent=agent)
        assert snapshot(instance.root) == before and instance.status()["attempts"] == 0


@pytest.mark.parametrize("window", ["started", "pending", "received_unsealed"])
def test_unknown_case_or_attempt_blocks_reopen_without_refund(tmp_path, window):
    instance = budget(tmp_path, send_limit=1)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        if window != "started":
            event = reserve(instance, lease)
            if window == "received_unsealed":
                instance.finish(event["operation_id"], "received", OUTPUT, lease=lease)
    before = snapshot(instance.root)
    restarted = reopen(instance.root, send_limit=1)
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with restarted.exclusive_run():
            pass
    assert restarted.status()["attempts"] == (0 if window == "started" else 1)
    assert snapshot(instance.root) == before


@pytest.mark.parametrize("outcome,code", [
    ("execution_error", "timeout"), ("execution_error", "network"), ("execution_error", "auth"),
    ("execution_error", "quota"), ("execution_error", "rate_limit"),
    ("execution_error", "storage_failure"), ("cancelled", "cancelled"), ("unknown", None),
])
def test_failed_or_uncertain_send_pauses_and_never_refunds(tmp_path, outcome, code):
    instance = budget(tmp_path, send_limit=1)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        event = reserve(instance, lease)
        instance.finish(event["operation_id"], outcome, error_code=code, lease=lease)
        assert instance.status()["attempts"] == 1 and instance.status()["paused"]
        with pytest.raises(old.TaskBudgetError, match="paused"):
            reserve(instance, lease)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with reopen(instance.root, send_limit=1).exclusive_run():
            pass


def test_structure_failure_is_kept_while_remaining_fixed_case_can_continue(tmp_path):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        event = reserve(instance, lease)
        instance.finish(event["operation_id"], "structure_failed", OUTPUT, lease=lease)
        instance.finish_case(FIRST, OUTPUT, outcome="failed", lease=lease)
        instance.start_case(SECOND, lease=lease)
        completed(instance, lease, case=SECOND)
        instance.finish_case(SECOND, OUTPUT, lease=lease)
    assert instance.status()["attempts"] == 2 and not instance.status()["paused"]
    assert instance.results()["cases"][FIRST]["outcome"] == "failed"


def test_duplicate_finish_case_start_and_stale_or_other_instance_leases_rejected(tmp_path):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        other = reopen(instance.root)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            other.start_case(FIRST, lease=lease)
        instance.start_case(FIRST, lease=lease)
        event = completed(instance, lease)
        before = snapshot(instance.root)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            instance.finish(event["operation_id"], "received", OUTPUT, lease=lease)
        assert snapshot(instance.root) == before
        instance.finish_case(FIRST, OUTPUT, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            instance.start_case(FIRST, lease=lease)
    with pytest.raises(old.TaskBudgetError, match="protocol"):
        instance.start_case(SECOND, lease=lease)


@pytest.mark.parametrize("damage", ["journal_tail", "journal_rollback", "claim_truncation", "claim_removed"])
def test_damaged_or_rolled_back_storage_is_not_repaired_or_reinitialized(tmp_path, damage):
    instance = budget(tmp_path)
    with instance.exclusive_run() as lease:
        instance.start_case(FIRST, lease=lease)
        completed(instance, lease)
        instance.finish_case(FIRST, OUTPUT, lease=lease)
    journal = instance.root / "journal.jsonl"
    if damage == "journal_tail":
        with journal.open("ab") as stream:
            stream.write(b'{"partial":')
    elif damage == "journal_rollback":
        journal.write_bytes(journal.read_bytes().splitlines(keepends=True)[0])
    else:
        claim = next((instance.root / "claims").iterdir())
        if damage == "claim_truncation": claim.write_bytes(b'{"partial":')
        else: claim.unlink()
    before = snapshot(instance.root)
    with pytest.raises(old.TaskBudgetError):
        reopen(instance.root)
    with pytest.raises(old.TaskBudgetError):
        module.DiagnosticRepairBudget(instance.root, PROTOCOL, module.fixed_plan(), initialize=True)
    assert snapshot(instance.root) == before


def test_symlink_wrong_protocol_and_old_trial_identities_remain_rejected(tmp_path):
    instance = budget(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(instance.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        reopen(alias)
    with pytest.raises(old.TaskBudgetError, match="protocol"):
        module.DiagnosticRepairBudget(instance.root, "f" * 64, module.fixed_plan())
    old_plan = [{"case_id": f"old{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
                for i in range(1, 7)]
    with pytest.raises(old.TaskBudgetError):
        old.TaskBudget(instance.root, PROTOCOL, old_plan)
    with pytest.raises(old.TaskBudgetError):
        prospective_budget.ProspectiveBudget(instance.root, PROTOCOL, prospective_budget.fixed_plan())
    old_root = tmp_path / "old-ledger"
    old.TaskBudget(old_root, PROTOCOL, old_plan, initialize=True)
    before = snapshot(old_root)
    with pytest.raises(old.TaskBudgetError):
        reopen(old_root)
    assert snapshot(old_root) == before
