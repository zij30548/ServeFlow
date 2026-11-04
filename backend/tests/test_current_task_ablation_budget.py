"""Real temporary accounting files, with no DB, model or network access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import socket
import threading

import pytest

from evaluation import current_task_ablation_budget as module, task_budget as old


HASH = "a" * 64
WIRE = "b" * 64


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("temporary accounting tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def budget(tmp_path, *, send_limit=module.MAX_REQUESTS):
    return module.TaskAblationBudget(tmp_path / "ledger", HASH, module.fixed_plan(),
                                     initialize=True, send_limit=send_limit)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def completed(b, lease, cid="A-p01", phase="intent", agent=None):
    claim = b.reserve(cid, "turn-1", phase, agent, HASH, WIRE, lease=lease)
    b.finish(claim["operation_id"], "received", HASH, lease=lease)
    return claim


def test_closed_plan_and_real_cap_are_distinct_from_the_mathematical_envelope():
    plan = module.fixed_plan()
    assert len(plan) == 24 and all(row["turn_ids"] == ["turn-1"] for row in plan)
    assert [row["case_id"] for row in plan[:8]] == [
        "A-p01", "D-p01", "D-p02", "A-p02", "A-p03", "D-p03", "D-p04", "A-p04"]
    assert [row["case_id"] for row in plan[-2:]] == ["D-p12", "A-p12"]
    assert len({row["case_id"] for row in plan}) == 24
    assert module.PHASE_CAPS == {"intent": 1, "agent": 15, "composer": 1,
                                 "rag_rewrite": 3, "rag_rerank": 3}
    assert module.PER_TURN == 23 and module.MATHEMATICAL_REQUESTS == 552
    assert module.MAX_REQUESTS == 120 and module.MAX_IN_FLIGHT == 3
    assert module.LIMITS == {"cases": 24, "turns": 24, "physical_requests": 120,
        "mathematical_phase_envelope": 552, "per_turn": 23, "in_flight": 3,
        "phase_caps": module.PHASE_CAPS, "judge_allowed": False}
    assert old.MAX_REQUESTS == 168 and old.PER_TURN == 24 and old.PHASE_CAPS["judge"] == 1


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "extra_turn", "old_id",
    "retrieval_id", "extra_field", "wrong_arm", "outside_question", "tuple_turns"])
def test_plan_changes_fail_before_creating_a_new_ledger(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "extra_turn": plan[0]["turn_ids"].append("turn-2")
    elif damage == "old_id": plan[0]["case_id"] = "D-n01"
    elif damage == "retrieval_id": plan[0]["case_id"] = "R-q01"
    elif damage == "extra_field": plan[0]["new_allowance"] = 552
    elif damage == "wrong_arm": plan[0]["case_id"] = "B-p01"
    elif damage == "outside_question": plan[0]["case_id"] = "A-p13"
    else: plan[0]["turn_ids"] = ("turn-1",)
    root = tmp_path / "bad"
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.TaskAblationBudget(root, HASH, plan, initialize=True)
    assert not root.exists()


@pytest.mark.parametrize("limit", [0, -1, 121, 552, 576, True, False, 1.0, "120", None])
def test_allowance_cannot_exceed_the_real_batch_cap_or_change_type(tmp_path, limit):
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        budget(tmp_path, send_limit=limit)
    assert not (tmp_path / "ledger").exists()


def test_reopen_preserves_root_protocol_identity_and_never_reinitializes(tmp_path):
    plan = module.fixed_plan()
    b = module.TaskAblationBudget(tmp_path / "ledger", HASH, plan, initialize=True, send_limit=7)
    plan[0]["case_id"] = "untrusted-mutation"
    assert b.case_plan[0]["case_id"] == "A-p01"
    before = snapshot(b.root)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), send_limit=7)
    assert reopened.initial_chain_hash == b.initial_chain_hash
    assert reopened.max_requests == 120 and reopened.send_limit == 7
    assert reopened.status()["attempts"] == 0 and snapshot(b.root) == before
    assert b.root.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
               for path in b.root.rglob("*"))
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), initialize=True)
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(b.root, "f" * 64, module.fixed_plan())
    link = tmp_path / "alias"; link.symlink_to(b.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.TaskAblationBudget(link, HASH, module.fixed_plan())
    assert snapshot(b.root) == before


def test_lower_allowance_is_checked_inside_the_same_reservation_lock(tmp_path):
    b = budget(tmp_path, send_limit=1)
    barrier = threading.Barrier(2)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        def contend(role):
            barrier.wait(timeout=2)
            try:
                return b.reserve("A-p01", "turn-1", "agent", role, HASH, WIRE, lease=lease)
            except old.TaskBudgetError as error:
                return error.code
        with ThreadPoolExecutor(max_workers=2) as workers:
            outcomes = list(workers.map(contend, ("general", "billing")))
        granted = [value for value in outcomes if isinstance(value, dict)]
        assert len(granted) == 1
        assert next(value for value in outcomes if isinstance(value, str)) in {"quota", "ledger_busy"}
        b.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        b.finish_case("A-p01", HASH, lease=lease)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), send_limit=1)
    with reopened.exclusive_run() as lease:
        reopened.start_case("D-p01", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            completed(reopened, lease, "D-p01")
    assert reopened.status()["attempts"] == 1


def test_all_24_jobs_can_progress_but_completed_or_out_of_order_jobs_cannot_repeat(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            b.start_case("D-p01", lease=lease)
        for row in module.fixed_plan():
            cid = row["case_id"]
            b.start_case(cid, lease=lease)
            completed(b, lease, cid)
            b.finish_case(cid, HASH, lease=lease)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan())
    state = reopened.status()
    assert state["attempts"] == 24
    assert state["completed_cases"] == list(module.CASE_IDS)
    assert not state["pending"] and not state["paused"]
    before = snapshot(b.root)
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reopened.start_case("A-p01", lease=lease)
    assert snapshot(b.root) == before


def test_real_120th_reservation_wins_once_under_concurrency_and_survives_reopen(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for row in module.fixed_plan()[:-1]:
            cid = row["case_id"]; b.start_case(cid, lease=lease)
            completed(b, lease, cid)
            for _ in range(4): completed(b, lease, cid, "agent", "general")
            b.finish_case(cid, HASH, lease=lease)
        cid = module.CASE_IDS[-1]; b.start_case(cid, lease=lease)
        completed(b, lease, cid)
        for _ in range(3): completed(b, lease, cid, "agent", "general")
        assert b.status()["attempts"] == 119
        barrier = threading.Barrier(2)
        def contend(role):
            barrier.wait(timeout=2)
            try: return b.reserve(cid, "turn-1", "agent", role, HASH, WIRE, lease=lease)
            except old.TaskBudgetError as error: return error.code
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(contend, ("general", "billing")))
        granted = [value for value in results if isinstance(value, dict)]
        assert len(granted) == 1 and granted[0]["attempt"] == 120
        assert next(value for value in results if isinstance(value, str)) in {"quota", "ledger_busy"}
        b.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            completed(b, lease, cid, "agent", "technical")
        b.finish_case(cid, HASH, lease=lease)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 120
    assert len(reopened.results()["operations"]) == 120
    assert reopened.status()["completed_cases"] == list(module.CASE_IDS)


@pytest.mark.parametrize("window", ["case_start", "pending", "received_unsealed"])
def test_unfinished_job_or_unknown_claim_blocks_restart_without_refund(tmp_path, window):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        if window != "case_start":
            claim = b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
            if window == "received_unsealed": b.finish(claim["operation_id"], "received", HASH, lease=lease)
    before = snapshot(b.root)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass
    assert reopened.status()["attempts"] == (0 if window == "case_start" else 1)
    assert snapshot(b.root) == before


@pytest.mark.parametrize("outcome,code", [("execution_error", "timeout"), ("execution_error", "auth"),
    ("execution_error", "quota"), ("execution_error", "rate_limit"), ("execution_error", "network"),
    ("execution_error", "storage_failure"), ("cancelled", "cancelled"), ("unknown", None)])
def test_sent_failure_cancel_or_uncertainty_is_spent_and_pauses_the_batch(tmp_path, outcome, code):
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        claim = b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], outcome, error_code=code, lease=lease)
        assert b.status()["attempts"] == 1 and b.status()["paused"]
        with pytest.raises(old.TaskBudgetError, match="paused"):
            completed(b, lease)
    reopened = module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with reopened.exclusive_run(): pass


def test_structure_failure_stays_counted_and_next_fixed_job_can_continue(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        claim = b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], "structure_failed", HASH, lease=lease)
        b.finish_case("A-p01", HASH, outcome="failed", lease=lease)
        b.start_case("D-p01", lease=lease)
        completed(b, lease, "D-p01")
        b.finish_case("D-p01", HASH, lease=lease)
    assert b.status()["attempts"] == 2 and not b.status()["paused"]


def test_three_inflight_and_each_no_judge_phase_limit_are_retained(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        claims = [b.reserve("A-p01", "turn-1", "agent", role, HASH, WIRE, lease=lease)
                  for role in ("general", "billing", "technical")]
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        for claim in claims: b.finish(claim["operation_id"], "received", HASH, lease=lease)
        for phase, cap in module.PHASE_CAPS.items():
            agent = "general" if phase == "agent" else None
            for _ in range(cap - (3 if phase == "agent" else 0)):
                completed(b, lease, phase=phase, agent=agent)
            with pytest.raises(old.TaskBudgetError, match="quota"):
                completed(b, lease, phase=phase, agent=agent)
        assert b.status()["attempts"] == 23
        b.finish_case("A-p01", HASH, lease=lease)


@pytest.mark.parametrize("phase,agent", [("judge", None), ("memory", None), ("embedding", None),
    ("agent", None), ("agent", "unknown"), ([], None)])
def test_forbidden_phase_or_role_cannot_get_a_claim(tmp_path, phase, agent):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        before = snapshot(b.root)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            completed(b, lease, phase=phase, agent=agent)
        assert snapshot(b.root) == before and b.status()["attempts"] == 0


@pytest.mark.parametrize("damage", ["journal_tail", "claim_truncated", "claim_removed"])
def test_corrupt_claim_or_journal_fails_closed_without_repair(tmp_path, damage):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        completed(b, lease)
        b.finish_case("A-p01", HASH, lease=lease)
    if damage == "journal_tail":
        with (b.root / "journal.jsonl").open("ab") as stream: stream.write(b'{"partial":')
    else:
        claim = next((b.root / "claims").iterdir())
        if damage == "claim_truncated": claim.write_bytes(b'{"partial":')
        else: claim.unlink()
    before = snapshot(b.root)
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(b.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), initialize=True)
    assert snapshot(b.root) == before


def _competing_instance(root, queue):
    try:
        b = module.TaskAblationBudget(Path(root), HASH, module.fixed_plan(), send_limit=1)
        with b.exclusive_run(): queue.put("incorrectly-acquired")
    except old.TaskBudgetError as error: queue.put(error.code)


def test_independent_process_cannot_start_a_second_run_on_the_same_ledger(tmp_path):
    b = budget(tmp_path, send_limit=1)
    ctx = multiprocessing.get_context("spawn"); queue = ctx.Queue()
    with b.exclusive_run() as lease:
        child = ctx.Process(target=_competing_instance, args=(str(b.root), queue))
        child.start(); child.join(10)
        assert not child.is_alive() and child.exitcode == 0
        assert queue.get(timeout=1) == "ledger_busy"
        b.start_case("A-p01", lease=lease); completed(b, lease)
        b.finish_case("A-p01", HASH, lease=lease)
    assert module.TaskAblationBudget(b.root, HASH, module.fixed_plan(), send_limit=1).status()["attempts"] == 1


def test_previous_budget_classes_cannot_reinterpret_this_identity(tmp_path):
    from evaluation.prospective_budget import ProspectiveBudget, fixed_plan as prospective_plan
    from evaluation.quality_budget import QualityBudget, fixed_plan as quality_plan
    b = budget(tmp_path)
    before = snapshot(b.root)
    with pytest.raises(old.TaskBudgetError):
        ProspectiveBudget(b.root, HASH, prospective_plan())
    with pytest.raises(old.TaskBudgetError):
        QualityBudget(b.root, HASH, quality_plan())
    with pytest.raises(old.TaskBudgetError):
        old.TaskBudget(b.root, HASH, [{"case_id": f"e{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
                                    for i in range(1, 7)])
    assert snapshot(b.root) == before


def test_copying_a_spent_ledger_to_another_directory_does_not_restore_allowance(tmp_path):
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        completed(b, lease)
        b.finish_case("A-p01", HASH, lease=lease)
    before = snapshot(b.root)
    copied = tmp_path / "copied"; shutil.copytree(b.root, copied)
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(copied, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError):
        module.TaskAblationBudget(copied, HASH, module.fixed_plan(), initialize=True, send_limit=1)
    assert snapshot(b.root) == before
    assert b.status()["attempts"] == 1
