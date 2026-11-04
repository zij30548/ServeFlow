"""Real temporary ledger tests; no application database, model or network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import socket
import threading

import pytest

from evaluation import prospective_budget as module, task_budget as old


HASH = "a" * 64
WIRE = "b" * 64


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prospective accounting must not contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def budget(tmp_path, *, send_limit=module.MAX_REQUESTS):
    return module.ProspectiveBudget(tmp_path / "ledger", HASH, module.fixed_plan(),
                                    initialize=True, send_limit=send_limit)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def completed(b, lease, cid="A-p01", phase="intent", agent=None):
    claim = b.reserve(cid, "turn-1", phase, agent, HASH, WIRE, lease=lease)
    b.finish(claim["operation_id"], "received", HASH, lease=lease)
    return claim


def advance_tasks(b, lease):
    for row in module.fixed_plan()[:24]:
        cid = row["case_id"]
        b.start_case(cid, lease=lease)
        completed(b, lease, cid)
        b.finish_case(cid, HASH, lease=lease)


def test_frozen_design_order_single_turn_and_shared_mathematical_ceiling():
    plan = module.fixed_plan()
    assert len(plan) == 36 and all(row["turn_ids"] == ["turn-1"] for row in plan)
    assert [row["case_id"] for row in plan[:6]] == ["A-p01", "D-p01", "D-p02", "A-p02", "A-p03", "D-p03"]
    assert [row["case_id"] for row in plan[24:]] == [f"R-q{i:02d}" for i in range(1, 13)]
    assert len({row["case_id"] for row in plan}) == 36
    assert module.TASK_PHASE_CAPS == {"intent": 1, "agent": 15, "composer": 1, "rag_rewrite": 3, "rag_rerank": 3}
    assert module.RAG_PHASE_CAPS == {"rag_rewrite": 1, "rag_rerank": 1}
    assert module.TASK_PER_TURN == 23 and module.RAG_PER_TURN == 2
    assert module.MAX_REQUESTS == 24 * 23 + 12 * 2 == 576
    assert module.MAX_IN_FLIGHT == 3 and module.LIMITS["judge_allowed"] is False
    assert old.PER_TURN == 24 and old.MAX_REQUESTS == 168 and old.PHASE_CAPS["judge"] == 1


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "extra_turn", "old_ids", "local_arm", "field"])
def test_different_plan_cannot_create_new_allowance(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "extra_turn": plan[0]["turn_ids"].append("turn-2")
    elif damage == "old_ids": plan[0]["case_id"] = "A-e01"
    elif damage == "local_arm": plan[24]["case_id"] = "A-q01"
    else: plan[0]["limit"] = 600
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.ProspectiveBudget(tmp_path / "bad", HASH, plan, initialize=True)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("limit", [0, -1, 577, True, False, 1.0, "1", None])
def test_allowance_is_a_strict_bounded_integer(tmp_path, limit):
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        budget(tmp_path, send_limit=limit)
    assert not (tmp_path / "ledger").exists()


def test_private_identity_does_not_change_with_lower_execution_allowance(tmp_path):
    b = budget(tmp_path, send_limit=1)
    before = snapshot(b.root)
    reader = module.ProspectiveBudget(b.root, HASH, module.fixed_plan())
    assert reader.max_requests == 576 and reader.send_limit == 576
    assert b.max_requests == 576 and b.send_limit == 1
    assert reader.initial_chain_hash == b.initial_chain_hash
    assert snapshot(b.root) == before
    assert b.root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600) for p in b.root.iterdir())
    with pytest.raises(old.TaskBudgetError):
        module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), initialize=True)


def test_lower_allowance_is_atomic_and_persistent_across_instances(tmp_path):
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
        assert [value for value in outcomes if isinstance(value, str)][0] in {"quota", "ledger_busy"}
        assert b.status()["attempts"] == 1
        b.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        b.finish_case("A-p01", HASH, lease=lease)
    reopened = module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), send_limit=1)
    with reopened.exclusive_run() as lease:
        reopened.start_case("D-p01", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            completed(reopened, lease, "D-p01")
    assert reopened.status()["attempts"] == 1
    assert module.ProspectiveBudget(b.root, HASH, module.fixed_plan()).status()["attempts"] == 1


def test_all_36_jobs_progress_and_completed_jobs_cannot_be_repeated(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        advance_tasks(b, lease)
        for row in module.fixed_plan()[24:]:
            cid = row["case_id"]
            b.start_case(cid, lease=lease)
            for phase in module.RAG_PHASE_CAPS:
                completed(b, lease, cid, phase)
            b.finish_case(cid, HASH, lease=lease)
    reopened = module.ProspectiveBudget(b.root, HASH, module.fixed_plan())
    state = reopened.status()
    assert state["attempts"] == 48 and len(state["completed_cases"]) == 36
    assert state["completed_cases"] == [row["case_id"] for row in module.fixed_plan()]
    assert not state["pending"] and not state["paused"]
    before = snapshot(b.root)
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reopened.start_case("A-p01", lease=lease)
    assert snapshot(b.root) == before


@pytest.mark.parametrize("window", ["case_start", "pending", "received_unsealed"])
def test_unknown_or_unfinished_case_blocks_restart_without_refund(tmp_path, window):
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        if window != "case_start":
            claim = b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
            if window == "received_unsealed": b.finish(claim["operation_id"], "received", HASH, lease=lease)
    before = snapshot(b.root)
    reopened = module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass
    assert reopened.status()["attempts"] == (0 if window == "case_start" else 1)
    assert snapshot(b.root) == before


@pytest.mark.parametrize("outcome,code", [("execution_error", "timeout"), ("execution_error", "auth"),
    ("execution_error", "quota"), ("execution_error", "rate_limit"), ("execution_error", "network"),
    ("execution_error", "storage_failure"), ("cancelled", "cancelled"), ("unknown", None)])
def test_failed_cancelled_or_unknown_attempt_keeps_allowance_and_pauses(tmp_path, outcome, code):
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        claim = b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], outcome, error_code=code, lease=lease)
        assert b.status()["attempts"] == 1 and b.status()["paused"]
        with pytest.raises(old.TaskBudgetError, match="paused"):
            completed(b, lease)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), send_limit=1).exclusive_run(): pass


def test_structure_failure_is_not_refunded_but_fixed_plan_can_continue(tmp_path):
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


def test_shared_three_inflight_and_task_phase_caps_include_every_non_judge_stage(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        claims = [b.reserve("A-p01", "turn-1", "agent", role, HASH, WIRE, lease=lease)
                  for role in ("general", "billing", "technical")]
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            b.reserve("A-p01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        for claim in claims: b.finish(claim["operation_id"], "received", HASH, lease=lease)
        for phase, cap in module.TASK_PHASE_CAPS.items():
            agent = "general" if phase == "agent" else None
            for _ in range(cap - (3 if phase == "agent" else 0)):
                completed(b, lease, phase=phase, agent=agent)
            with pytest.raises(old.TaskBudgetError, match="quota"):
                completed(b, lease, phase=phase, agent=agent)
        assert b.status()["attempts"] == 23
        b.finish_case("A-p01", HASH, lease=lease)


@pytest.mark.parametrize("kind", ["global", "authorized", "task_turn", "task_phase", "retrieval_turn", "retrieval_phase"])
def test_each_actual_reserve_path_checks_its_correct_bound(kind, tmp_path):
    b = budget(tmp_path, send_limit=7 if kind == "authorized" else 576)
    cid = "R-q01" if kind.startswith("retrieval") else "A-p01"
    state = b._empty(); state["active_case"] = cid
    phase, agent = ("rag_rewrite", None) if cid.startswith("R-") else ("agent", "general")
    if kind == "global": state["attempts"] = 576
    elif kind == "authorized": state["attempts"] = 7
    elif kind == "task_turn": state["turn_attempts"][(cid, "turn-1")] = 23
    elif kind == "task_phase": state["phase_counts"][(cid, "turn-1", phase)] = 15
    elif kind == "retrieval_turn": state["turn_attempts"][(cid, "turn-1")] = 2
    else: state["phase_counts"][(cid, "turn-1", phase)] = 1
    with pytest.raises(old.TaskBudgetError, match="quota"):
        b._reserve_check(state, cid, "turn-1", phase, agent)


def test_standalone_retrieval_only_allows_ordered_single_rewrite_and_rerank(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        advance_tasks(b, lease); b.start_case("R-q01", lease=lease)
        for phase, agent in [("intent", None), ("judge", None), ("agent", "general"), ("rag_rewrite", "billing")]:
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                completed(b, lease, "R-q01", phase, agent)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            completed(b, lease, "R-q01", "rag_rerank")
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            b.finish_case("R-q01", HASH, lease=lease)
        claim = b.reserve("R-q01", "turn-1", "rag_rewrite", None, HASH, WIRE, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            completed(b, lease, "R-q01", "rag_rerank")
        b.finish(claim["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            completed(b, lease, "R-q01", "rag_rewrite")
        completed(b, lease, "R-q01", "rag_rerank")
        with pytest.raises(old.TaskBudgetError, match="quota"):
            completed(b, lease, "R-q01", "rag_rerank")
        b.finish_case("R-q01", HASH, lease=lease)
    assert b.status()["attempts"] == 26


def test_legal_skipped_rerank_can_seal_success_and_is_not_sent_after_restart(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        advance_tasks(b, lease); b.start_case("R-q01", lease=lease)
        completed(b, lease, "R-q01", "rag_rewrite")
        # Production may return <= top_k merged items without a rerank call.
        # The runner seals actual source observations; accounting cannot add a
        # second request merely to fill the conservative two-call envelope.
        b.finish_case("R-q01", HASH, lease=lease)
    reopened = module.ProspectiveBudget(b.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 25
    assert reopened.results()["cases"]["R-q01"]["outcome"] == "success"
    assert [op for op in reopened.results()["operations"] if op.startswith("R-q01:")] == [
        "R-q01:turn-1:rag_rewrite:none:1"]
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reopened.start_case("R-q01", lease=lease)
        reopened.start_case("R-q02", lease=lease)
        completed(reopened, lease, "R-q02", "rag_rewrite")
        reopened.finish_case("R-q02", HASH, lease=lease)
    assert reopened.status()["attempts"] == 26


@pytest.mark.parametrize("phase,agent", [("judge", None), ("memory", None), ("embedding", None),
    ("agent", "unknown"), ("agent", None), ([], None)])
def test_disallowed_stages_or_roles_never_get_a_claim(tmp_path, phase, agent):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        before = snapshot(b.root)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            completed(b, lease, phase=phase, agent=agent)
        assert snapshot(b.root) == before and b.status()["attempts"] == 0


@pytest.mark.parametrize("damage", ["journal_tail", "claim_truncation", "claim_removed"])
def test_partial_journal_or_claim_is_not_repaired_or_reinitialized(tmp_path, damage):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("A-p01", lease=lease)
        completed(b, lease)
        b.finish_case("A-p01", HASH, lease=lease)
    if damage == "journal_tail":
        with (b.root / "journal.jsonl").open("ab") as stream: stream.write(b'{"partial":')
    else:
        claim = next((b.root / "claims").iterdir())
        if damage == "claim_truncation": claim.write_bytes(b'{"partial":')
        else: claim.unlink()
    before = snapshot(b.root)
    with pytest.raises(old.TaskBudgetError):
        module.ProspectiveBudget(b.root, HASH, module.fixed_plan())
    assert snapshot(b.root) == before
    with pytest.raises(old.TaskBudgetError):
        module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), initialize=True)


def _competing_instance(root, result):
    try:
        b = module.ProspectiveBudget(Path(root), HASH, module.fixed_plan(), send_limit=1)
        with b.exclusive_run() as lease:
            b.start_case("A-p01", lease=lease)
            completed(b, lease)
            result.put("incorrectly-reserved")
    except old.TaskBudgetError as error:
        result.put(error.code)


def test_independent_process_with_same_one_send_allowance_cannot_double_reserve(tmp_path):
    b = budget(tmp_path, send_limit=1)
    ctx = multiprocessing.get_context("spawn"); result = ctx.Queue()
    with b.exclusive_run() as lease:
        child = ctx.Process(target=_competing_instance, args=(str(b.root), result))
        child.start(); child.join(10)
        assert not child.is_alive() and child.exitcode == 0
        assert result.get(timeout=1) == "ledger_busy"
        b.start_case("A-p01", lease=lease); completed(b, lease)
        b.finish_case("A-p01", HASH, lease=lease)
    assert module.ProspectiveBudget(b.root, HASH, module.fixed_plan(), send_limit=1).status()["attempts"] == 1


def test_symlink_and_old_identity_do_not_open_this_ledger(tmp_path):
    b = budget(tmp_path)
    link = tmp_path / "alias"; link.symlink_to(b.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.ProspectiveBudget(link, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError):
        module.ProspectiveBudget(b.root, "f" * 64, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError):
        old.TaskBudget(b.root, HASH, [{"case_id": f"c{i}", "turn_ids": ["t1", "t2"] if i == 1 else ["t1"]}
                                    for i in range(1, 7)])
