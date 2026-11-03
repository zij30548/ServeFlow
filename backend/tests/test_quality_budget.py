import multiprocessing
from pathlib import Path

import pytest

from evaluation import quality_budget as module
from evaluation import task_budget as old


HASH = "a" * 64
WIRE = "b" * 64


def budget(tmp_path):
    return module.QualityBudget(tmp_path / "global", HASH, module.fixed_plan(), initialize=True)


def test_fixed_rotation_is_48_cases_52_turns_one_global_cap():
    plan = module.fixed_plan()
    assert len(plan) == 48 and sum(len(p["turn_ids"]) for p in plan) == 52
    assert [p["case_id"] for p in plan[:8]] == ["A-e01", "B-e01", "C-e01", "D-e01", "B-e02", "C-e02", "D-e02", "A-e02"]
    assert len({p["case_id"] for p in plan}) == 48
    assert module.LIMITS == {"cases": 48, "turns": 52, "physical_requests": 1248,
        "per_turn": 24, "in_flight": 3, "phase_caps": old.PHASE_CAPS}


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "extra_turn", "old_ids"])
def test_plan_cannot_be_changed_to_restore_or_reallocate_allowance(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = dict(plan[0])
    elif damage == "extra_turn": plan[0]["turn_ids"].append("turn-2")
    else: plan[0]["case_id"] = "e01"
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.QualityBudget(tmp_path / "new", HASH, plan, initialize=True)
    assert not (tmp_path / "new").exists()


def test_global_reservations_and_arm_identifiers_survive_reopen(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for arm in "ABCD":
            cid = arm + "-e01"
            b.start_case(cid, lease=lease)
            claim = b.reserve(cid, "turn-1", "intent", None, HASH, WIRE, lease=lease)
            assert claim["operation_id"] == cid + ":turn-1:intent:none:1"
            b.finish(claim["operation_id"], "received", HASH, lease=lease)
            b.finish_case(cid, HASH, lease=lease)
    reopened = module.QualityBudget(b.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 4
    assert reopened.status()["completed_cases"] == ["A-e01", "B-e01", "C-e01", "D-e01"]
    assert reopened.initial_chain_hash == b.initial_chain_hash
    assert len(reopened.results()["operations"]) == 4
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError): reopened.start_case("A-e01", lease=lease)
        with pytest.raises(old.TaskBudgetError): reopened.start_case("C-e02", lease=lease)


def test_all_48_cases_and_52_turns_progress_without_inherited_six_case_cap(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for planned in module.fixed_plan():
            cid = planned["case_id"]; b.start_case(cid, lease=lease)
            for turn in planned["turn_ids"]:
                claim = b.reserve(cid, turn, "intent", None, HASH, WIRE, lease=lease)
                b.finish(claim["operation_id"], "received", HASH, lease=lease)
            b.finish_case(cid, HASH, lease=lease)
    reopened = module.QualityBudget(b.root, HASH, module.fixed_plan())
    state = reopened.status()
    assert state["attempts"] == 52 and len(state["completed_cases"]) == 48
    assert state["completed_cases"][-1] == "C-t12"
    assert not state["pending"] and not state["paused"]
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError): reopened.start_case("A-e01", lease=lease)


def test_pending_claim_is_sticky_and_prevents_replay_after_restart(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-e01", lease=lease)
        claim = b.reserve("A-e01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
    reopened = module.QualityBudget(b.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 1
    assert reopened.status()["pending"] == [claim["operation_id"]]
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass
    with pytest.raises(old.TaskBudgetError):
        module.QualityBudget(b.root, HASH, module.fixed_plan(), initialize=True)


@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "network", "auth", "quota", "rate_limit"])
def test_sent_execution_failure_is_counted_and_pauses_entire_batch(tmp_path, outcome):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-e01", lease=lease)
        claim = b.reserve("A-e01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], "cancelled" if outcome == "cancelled" else "execution_error", error_code=outcome, lease=lease)
        assert b.status()["attempts"] == 1 and b.status()["paused"]
        with pytest.raises(old.TaskBudgetError, match="paused"):
            b.reserve("A-e01", "turn-1", "agent", "general", HASH, WIRE, lease=lease)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with module.QualityBudget(b.root, HASH, module.fixed_plan()).exclusive_run(): pass


def test_structure_failed_response_is_not_refunded_but_fixed_plan_can_continue(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-e01", lease=lease)
        claim = b.reserve("A-e01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], "structure_failed", HASH, lease=lease)
        b.finish_case("A-e01", HASH, outcome="failed", lease=lease)
        b.start_case("B-e01", lease=lease)
        assert b.status()["attempts"] == 1 and not b.status()["paused"]


def test_three_inflight_and_per_phase_limit_are_atomic(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-e01", lease=lease)
        claims = [b.reserve("A-e01", "turn-1", "agent", role, HASH, WIRE, lease=lease)
                  for role in ("general", "billing", "technical")]
        assert len(b.status()["pending"]) == 3
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            b.reserve("A-e01", "turn-1", "judge", None, HASH, WIRE, lease=lease)
        b.finish(claims[1]["operation_id"], "received", HASH, lease=lease)
        b.reserve("A-e01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claims[0]["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            b.reserve("A-e01", "turn-1", "intent", None, HASH, WIRE, lease=lease)


@pytest.mark.parametrize("which", ["global", "turn", "phase"])
def test_all_actual_reserve_paths_enforce_hard_upper_bound(tmp_path, which):
    b = budget(tmp_path)
    state = b._empty(); state["active_case"] = "A-e01"
    if which == "global": state["attempts"] = 1248
    elif which == "turn": state["turn_attempts"][("A-e01", "turn-1")] = 24
    else: state["phase_counts"][("A-e01", "turn-1", "agent")] = 15
    with pytest.raises(old.TaskBudgetError, match="quota"):
        b._reserve_check(state, "A-e01", "turn-1", "agent", "general")


def _competing_run(root, result):
    try:
        b = module.QualityBudget(Path(root), HASH, module.fixed_plan())
        with b.exclusive_run(): result.put("incorrectly-acquired")
    except old.TaskBudgetError as error: result.put(str(error))


def test_second_independent_process_cannot_run_concurrently(tmp_path):
    b = budget(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    with b.exclusive_run():
        child = ctx.Process(target=_competing_run, args=(str(b.root), result)); child.start(); child.join(10)
        assert not child.is_alive() and child.exitcode == 0
        assert result.get(timeout=1) == "ledger_busy"
    assert b.status()["attempts"] == 0


def test_old_budget_identity_and_symlink_or_unsafe_storage_are_rejected(tmp_path):
    b = budget(tmp_path)
    with pytest.raises(old.TaskBudgetError):
        old.TaskBudget(b.root, HASH, [{"case_id": f"e{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]} for i in range(1, 7)])
    link = tmp_path / "alias"; link.symlink_to(b.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.QualityBudget(link, HASH, module.fixed_plan())
    (b.root / "journal.jsonl").chmod(0o644)
    with pytest.raises(old.TaskBudgetError): b.status()
