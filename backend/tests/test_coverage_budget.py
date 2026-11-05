"""New ledger identity and physical quotas; temporary private files only."""
from copy import deepcopy
import socket

import pytest

from evaluation import coverage_budget as budget, task_budget as original


H, W = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("budget tests cannot access services")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def make(tmp_path, limit=480):
    return budget.CoverageBudget(tmp_path / "ledger", H, budget.fixed_plan(), initialize=True, send_limit=limit)


def finish(b, lease, identity, *, turn="turn-1", phase="intent", agent=None, outcome="received"):
    row = b.reserve(identity, turn, phase, agent, H, W, lease=lease)
    b.finish(row["operation_id"], outcome, H if outcome in {"received", "structure_failed"} else None, lease=lease)
    return row


def test_closed_plan_has_exact_split_order_rotation_turns_and_independent_limits():
    specs, plan = budget.job_specs(), budget.fixed_plan()
    assert len(specs) == 136 and sum(len(row["turn_ids"]) for row in plan) == 168
    assert [r["case_id"] for r in specs[:4]] == ["A-CT-D01", "D-CT-D01", "D-CT-D02", "A-CT-D02"]
    assert [r["case_id"] for r in specs[40:42]] == ["M-CC-D01", "M-CC-D02"]
    assert specs[42]["case_id"] == "A-CT-V01" and specs[-1]["case_id"] == "M-CC-V06"
    assert budget.MATHEMATICAL_REQUESTS == 2856 and budget.MAX_REQUESTS == 480
    assert budget.MAX_IN_FLIGHT == 3 and original.MAX_REQUESTS == 168
    assert "judge" not in budget.TASK_PHASE_CAPS and original.PHASE_CAPS["judge"] == 1


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "extra", "tuple", "turns", "old-root-id"])
def test_changed_plans_rejected_before_storage(tmp_path, damage):
    plan = budget.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "extra": plan[0]["allowance"] = 999
    elif damage == "tuple": plan[0]["turn_ids"] = tuple(plan[0]["turn_ids"])
    elif damage == "turns": plan[40]["turn_ids"].pop()
    else: plan[0]["case_id"] = "RT01"
    with pytest.raises(budget.TaskBudgetError, match="configuration"):
        budget.CoverageBudget(tmp_path / "ledger", H, plan, initialize=True)
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("limit", [0, -1, 481, 2856, True, 480.0, "480", None])
def test_noninteger_and_oversize_allowance_rejected(tmp_path, limit):
    with pytest.raises(budget.TaskBudgetError, match="configuration"):
        make(tmp_path, limit)


def test_lower_authenticated_cap_is_enforced_before_new_claim_and_survives_reopen(tmp_path):
    b = make(tmp_path, 2)
    with b.exclusive_run() as lease:
        b.start_case(budget.CASE_IDS[0], lease=lease)
        finish(b, lease, budget.CASE_IDS[0])
        finish(b, lease, budget.CASE_IDS[0], phase="agent", agent="billing")
        before = (b.root / "journal.jsonl").read_bytes()
        with pytest.raises(budget.TaskBudgetError, match="quota"):
            b.reserve(budget.CASE_IDS[0], "turn-1", "composer", None, H, W, lease=lease)
        assert (b.root / "journal.jsonl").read_bytes() == before
    reopened = budget.CoverageBudget(b.root, H, budget.fixed_plan(), send_limit=2)
    assert reopened.status()["attempts"] == 2
    with pytest.raises(budget.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass


def test_hard480_cap_cannot_be_replaced_by_phase_envelope(tmp_path):
    b = make(tmp_path)
    with b.exclusive_run() as lease:
        for identity in budget.CASE_IDS:
            b.start_case(identity, lease=lease)
            for phase, cap in budget.TASK_PHASE_CAPS.items():
                for _ in range(cap):
                    if b.status()["attempts"] == 480:
                        with pytest.raises(budget.TaskBudgetError, match="quota"):
                            b.reserve(identity, "turn-1", phase, "general" if phase == "agent" else None, H, W, lease=lease)
                        assert b.status()["attempts"] == 480
                        return
                    finish(b, lease, identity, phase=phase, agent="general" if phase == "agent" else None)
            b.finish_case(identity, H, lease=lease)
    pytest.fail("hard cap not reached")


def test_three_inflight_and_unknown_reservation_never_refunded_or_resumed(tmp_path):
    b = make(tmp_path)
    identity = budget.CASE_IDS[0]
    with b.exclusive_run() as lease:
        b.start_case(identity, lease=lease)
        rows = [b.reserve(identity, "turn-1", "agent", role, H, W, lease=lease) for role in ("general", "technical", "billing")]
        with pytest.raises(budget.TaskBudgetError, match="ledger_busy"):
            b.reserve(identity, "turn-1", "composer", None, H, W, lease=lease)
        for row in rows[:2]:
            b.finish(row["operation_id"], "received", H, lease=lease)
    b = budget.CoverageBudget(b.root, H, budget.fixed_plan())
    assert b.status()["attempts"] == 3 and len(b.status()["pending"]) == 1
    with pytest.raises(budget.TaskBudgetError, match="pending"):
        with b.exclusive_run(): pass


def test_judge_wrong_actor_repeat_intent_and_inactive_case_rejected(tmp_path):
    b = make(tmp_path); identity = budget.CASE_IDS[0]
    with b.exclusive_run() as lease:
        b.start_case(identity, lease=lease)
        for phase, actor, case in (("judge", None, identity), ("agent", None, identity), ("intent", "billing", identity), ("intent", None, budget.CASE_IDS[1])):
            with pytest.raises(budget.TaskBudgetError):
                b.reserve(case, "turn-1", phase, actor, H, W, lease=lease)
        finish(b, lease, identity)
        with pytest.raises(budget.TaskBudgetError, match="quota"):
            b.reserve(identity, "turn-1", "intent", None, H, W, lease=lease)


def test_full_plan_once_including_sequential_five_turn_memory_and_rag_order(tmp_path):
    b = make(tmp_path)
    with b.exclusive_run() as lease:
        for spec, plan in zip(budget.job_specs(), budget.fixed_plan()):
            identity = spec["case_id"]
            b.start_case(identity, lease=lease)
            if spec["suite"] == "retrieval":
                with pytest.raises(budget.TaskBudgetError, match="protocol"):
                    b.reserve(identity, "turn-1", "rag_rerank", None, H, W, lease=lease)
                finish(b, lease, identity, phase="rag_rewrite")
                finish(b, lease, identity, phase="rag_rerank")
            else:
                if spec["suite"] == "conversations":
                    with pytest.raises(budget.TaskBudgetError, match="protocol"):
                        b.reserve(identity, "turn-2", "intent", None, H, W, lease=lease)
                for turn in plan["turn_ids"]:
                    finish(b, lease, identity, turn=turn)
            b.finish_case(identity, H, lease=lease)
    assert b.status()["completed_cases"] == list(budget.CASE_IDS)
    assert b.status()["attempts"] == 216
    reopened = budget.CoverageBudget(b.root, H, budget.fixed_plan())
    with reopened.exclusive_run() as lease:
        with pytest.raises(budget.TaskBudgetError):
            reopened.start_case(budget.CASE_IDS[0], lease=lease)


def test_wrong_protocol_corrupt_journal_and_old_ledger_classes_fail_closed(tmp_path):
    b = make(tmp_path)
    with pytest.raises(budget.TaskBudgetError):
        budget.CoverageBudget(b.root, W, budget.fixed_plan())
    with pytest.raises(original.TaskBudgetError):
        original.TaskBudget(b.root, H, budget.fixed_plan())
    with b.exclusive_run() as lease:
        b.start_case(budget.CASE_IDS[0], lease=lease)
    journal = b.root / "journal.jsonl"
    journal.write_bytes(journal.read_bytes()[:-1])
    with pytest.raises(budget.TaskBudgetError):
        budget.CoverageBudget(b.root, H, budget.fixed_plan())
