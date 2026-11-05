"""Mixed-suite accounting uses real private temporary files, never services."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import socket
import threading

import pytest

from evaluation import benchmark as bench, expanded_budget as module, task_budget as old


HASH = "a" * 64
WIRE = "b" * 64


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("temporary accounting tests cannot contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def budget(tmp_path, *, send_limit=module.MAX_REQUESTS):
    return module.ExpandedBudget(tmp_path / "ledger", HASH, module.fixed_plan(),
                                 initialize=True, send_limit=send_limit)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def received(b, lease, cid, phase="intent", agent=None):
    claim = b.reserve(cid, "turn-1", phase, agent, HASH, WIRE, lease=lease)
    b.finish(claim["operation_id"], "received", HASH, lease=lease)
    return claim


def minimally_completed(b, lease, cid):
    b.start_case(cid, lease=lease)
    if cid.startswith("R-"):
        received(b, lease, cid, "rag_rewrite")
        received(b, lease, cid, "rag_rerank")
    else:
        received(b, lease, cid)
    b.finish_case(cid, HASH, lease=lease)


def task_to_caps(b, lease, cid):
    b.start_case(cid, lease=lease)
    for phase, cap in module.TASK_PHASE_CAPS.items():
        for _ in range(cap):
            received(b, lease, cid, phase, "general" if phase == "agent" else None)
    b.finish_case(cid, HASH, lease=lease)


def test_closed_132_job_plan_matches_all_108_frozen_question_splits():
    specs = module.job_specs()
    assert len(specs) == len(module.fixed_plan()) == len(module.CASE_IDS) == 132
    assert len(set(module.CASE_IDS)) == 132
    assert Counter(row["suite"] for row in specs) == {"intents": 60, "tasks": 48, "retrieval": 24}
    assert Counter(row["split"] for row in specs) == {"development": 48, "validation": 84}
    assert [row["split"] for row in specs] == ["development"] * 48 + ["validation"] * 84
    assert module.CASE_IDS[:3] == ("I-xi001", "I-xi002", "I-xi006")
    assert module.CASE_IDS[24:28] == ("A-et01", "D-et01", "D-et02", "A-et02")
    assert module.CASE_IDS[-2:] == ("R-er23", "R-er24")
    assert all(row["turn_ids"] == ["turn-1"] for row in module.fixed_plan())
    for suite in ("intents", "tasks", "retrieval"):
        data = bench.read_json(bench.DATASET_ROOT / f"expanded-{suite}-v1.json")
        indexed = {c["id"]: c for c in data["cases"]}
        for row in (r for r in specs if r["suite"] == suite):
            assert indexed[row["source_case_id"]]["split"] == row["split"]
    assert module.MAX_REQUESTS == 480 and module.MATHEMATICAL_REQUESTS == 1212
    assert module.TASK_PER_TURN == 23 and module.INTENT_PER_TURN == 1 and module.RAG_PER_TURN == 2
    assert module.MAX_IN_FLIGHT == 3 and not module.LIMITS["judge_allowed"]
    assert old.MAX_REQUESTS == 168 and old.PHASE_CAPS["judge"] == 1


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "extra_turn", "old_id",
                                   "extra_field", "wrong_arm", "outside_question", "tuple_turns"])
def test_plan_changes_fail_before_creating_storage(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "extra_turn": plan[0]["turn_ids"].append("turn-2")
    elif damage == "old_id": plan[0]["case_id"] = "A-p01"
    elif damage == "extra_field": plan[0]["allowance"] = 1212
    elif damage == "wrong_arm": plan[0]["case_id"] = "B-xi001"
    elif damage == "outside_question": plan[0]["case_id"] = "I-xi061"
    else: plan[0]["turn_ids"] = ("turn-1",)
    target = tmp_path / "invalid"
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.ExpandedBudget(target, HASH, plan, initialize=True)
    assert not target.exists()


@pytest.mark.parametrize("limit", [0, -1, 481, 1212, True, False, 1.0, "480", None])
def test_hard_allowance_and_exact_types_cannot_be_relaxed(tmp_path, limit):
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        budget(tmp_path, send_limit=limit)
    assert not (tmp_path / "ledger").exists()


def test_reopen_and_mutable_plan_do_not_change_identity_or_occupied_allowance(tmp_path):
    plan = module.fixed_plan()
    b = module.ExpandedBudget(tmp_path / "ledger", HASH, plan, initialize=True, send_limit=2)
    plan[0]["case_id"] = "modified"
    assert b.case_plan[0]["case_id"] == "I-xi001"
    with b.exclusive_run() as lease:
        minimally_completed(b, lease, "I-xi001")
    before = snapshot(b.root)
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan(), send_limit=2)
    assert reopened.status()["attempts"] == 1 and reopened.max_requests == 480
    assert reopened.initial_chain_hash == b.initial_chain_hash and snapshot(b.root) == before
    assert b.root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600) for p in b.root.rglob("*"))
    for args in ((HASH, True), ("f" * 64, False)):
        with pytest.raises(old.TaskBudgetError):
            module.ExpandedBudget(b.root, args[0], module.fixed_plan(), initialize=args[1])
    link = tmp_path / "alias"; link.symlink_to(b.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.ExpandedBudget(link, HASH, module.fixed_plan())
    assert snapshot(b.root) == before


def test_intent_has_one_shared_observation_and_rejects_other_stages(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("I-xi001", lease=lease)
        for phase, agent in (("agent", "general"), ("composer", None), ("rag_rewrite", None),
                             ("rag_rerank", None), ("judge", None), ("intent", "general")):
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                received(b, lease, "I-xi001", phase, agent)
        received(b, lease, "I-xi001")
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(b, lease, "I-xi001")
        b.finish_case("I-xi001", HASH, lease=lease)
    assert b.status()["attempts"] == 1


def test_retrieval_requires_rewrite_before_rerank_and_only_one_each(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for cid in module.CASE_IDS:
            if cid.startswith("R-"):
                break
            minimally_completed(b, lease, cid)
        b.start_case(cid, lease=lease)
        for phase, agent in (("intent", None), ("agent", "general"), ("composer", None),
                             ("judge", None), ("rag_rewrite", "general"), ("rag_rerank", "billing")):
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                received(b, lease, cid, phase, agent)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            received(b, lease, cid, "rag_rerank")
        claim = b.reserve(cid, "turn-1", "rag_rewrite", None, HASH, WIRE, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            received(b, lease, cid, "rag_rerank")
        b.finish(claim["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(b, lease, cid, "rag_rewrite")
        received(b, lease, cid, "rag_rerank")
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(b, lease, cid, "rag_rerank")
        b.finish_case(cid, HASH, lease=lease)


def test_retrieval_can_skip_rerank_without_a_fake_extra_reservation(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for cid in module.CASE_IDS:
            if cid.startswith("R-"):
                break
            minimally_completed(b, lease, cid)
        before = b.status()["attempts"]
        b.start_case(cid, lease=lease)
        received(b, lease, cid, "rag_rewrite")
        b.finish_case(cid, HASH, lease=lease)
        assert b.status()["attempts"] == before + 1
        operations = b.results()["operations"]
        assert [v["phase"] for v in operations.values() if v["case_id"] == cid] == ["rag_rewrite"]


def test_lower_allowance_is_shared_atomically_by_agents_and_suite_jobs(tmp_path):
    b = budget(tmp_path, send_limit=25)
    with b.exclusive_run() as lease:
        for cid in module.CASE_IDS[:24]:
            minimally_completed(b, lease, cid)
        cid = "A-et01"; b.start_case(cid, lease=lease)
        barrier = threading.Barrier(2)
        def contend(role):
            barrier.wait(timeout=2)
            try: return b.reserve(cid, "turn-1", "agent", role, HASH, WIRE, lease=lease)
            except old.TaskBudgetError as error: return error.code
        with ThreadPoolExecutor(max_workers=2) as workers:
            values = list(workers.map(contend, ("general", "billing")))
        granted = [v for v in values if isinstance(v, dict)]
        assert len(granted) == 1 and granted[0]["attempt"] == 25
        assert next(v for v in values if isinstance(v, str)) in {"quota", "ledger_busy"}
        b.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        b.finish_case(cid, HASH, lease=lease)
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan(), send_limit=25)
    with reopened.exclusive_run() as lease:
        reopened.start_case("D-et01", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(reopened, lease, "D-et01")
    assert reopened.status()["attempts"] == 25


def test_all_jobs_and_both_splits_finish_in_fixed_order_without_repeat(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            b.start_case("I-xi003", lease=lease)
        for cid in module.CASE_IDS:
            minimally_completed(b, lease, cid)
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan())
    state = reopened.status()
    assert state["completed_cases"] == list(module.CASE_IDS)
    assert state["attempts"] == 156 and not state["pending"] and not state["paused"]
    before = snapshot(b.root)
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reopened.start_case("I-xi001", lease=lease)
    assert snapshot(b.root) == before


def test_real_480th_request_wins_once_and_481st_is_rejected_after_reopen(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for cid in module.CASE_IDS:
            if cid == "D-et09":
                break
            if cid.startswith(("A-", "D-")):
                task_to_caps(b, lease, cid)
            else:
                minimally_completed(b, lease, cid)
        assert b.status()["attempts"] == 467
        b.start_case(cid, lease=lease)
        received(b, lease, cid)
        for _ in range(11):
            received(b, lease, cid, "agent", "general")
        assert b.status()["attempts"] == 479
        barrier = threading.Barrier(2)
        def contend(role):
            barrier.wait(timeout=2)
            try: return b.reserve(cid, "turn-1", "agent", role, HASH, WIRE, lease=lease)
            except old.TaskBudgetError as error: return error.code
        with ThreadPoolExecutor(max_workers=2) as workers:
            values = list(workers.map(contend, ("general", "billing")))
        granted = [v for v in values if isinstance(v, dict)]
        assert len(granted) == 1 and granted[0]["attempt"] == 480
        assert next(v for v in values if isinstance(v, str)) in {"quota", "ledger_busy"}
        b.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(b, lease, cid, "agent", "technical")
        b.finish_case(cid, HASH, lease=lease)
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 480
    assert len(reopened.results()["operations"]) == 480
    with reopened.exclusive_run() as lease:
        reopened.start_case("D-et10", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(reopened, lease, "D-et10")
    assert reopened.status()["attempts"] == 480


def test_tasks_keep_shared_phase_caps_and_three_inflight(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        for cid in module.CASE_IDS[:24]:
            minimally_completed(b, lease, cid)
        cid = "A-et01"; b.start_case(cid, lease=lease)
        claims = [b.reserve(cid, "turn-1", "agent", role, HASH, WIRE, lease=lease)
                  for role in ("general", "billing", "technical")]
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            received(b, lease, cid)
        for claim in claims:
            b.finish(claim["operation_id"], "received", HASH, lease=lease)
        for phase, cap in module.TASK_PHASE_CAPS.items():
            for _ in range(cap - (3 if phase == "agent" else 0)):
                received(b, lease, cid, phase, "general" if phase == "agent" else None)
            with pytest.raises(old.TaskBudgetError, match="quota"):
                received(b, lease, cid, phase, "general" if phase == "agent" else None)
        assert b.status()["turn_attempts"][cid + ":turn-1"] == 23
        for phase, agent in (("judge", None), ("memory", None), ("embedding", None),
                             ("agent", None), ("agent", "unknown")):
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                received(b, lease, cid, phase, agent)
        b.finish_case(cid, HASH, lease=lease)


@pytest.mark.parametrize("window", ["case_start", "pending", "received_unsealed"])
def test_uncertain_or_unsealed_jobs_block_automatic_restart(tmp_path, window):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("I-xi001", lease=lease)
        if window != "case_start":
            claim = b.reserve("I-xi001", "turn-1", "intent", None, HASH, WIRE, lease=lease)
            if window == "received_unsealed":
                b.finish(claim["operation_id"], "received", HASH, lease=lease)
    before = snapshot(b.root)
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass
    assert reopened.status()["attempts"] == (0 if window == "case_start" else 1)
    assert snapshot(b.root) == before


@pytest.mark.parametrize("outcome,code", [("execution_error", "timeout"), ("execution_error", "auth"),
    ("execution_error", "quota"), ("execution_error", "rate_limit"), ("execution_error", "network"),
    ("execution_error", "storage_failure"), ("cancelled", "cancelled"), ("unknown", None)])
def test_sent_failure_cancel_unknown_remain_spent_and_pause(tmp_path, outcome, code):
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        b.start_case("I-xi001", lease=lease)
        claim = b.reserve("I-xi001", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], outcome, error_code=code, lease=lease)
        assert b.status()["attempts"] == 1 and b.status()["paused"]
        with pytest.raises(old.TaskBudgetError, match="paused"):
            received(b, lease, "I-xi001")
    reopened = module.ExpandedBudget(b.root, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with reopened.exclusive_run(): pass


def test_structure_failure_can_finish_without_success_repair_and_next_job_continues(tmp_path):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        b.start_case("I-xi001", lease=lease)
        claim = b.reserve("I-xi001", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        b.finish(claim["operation_id"], "structure_failed", HASH, lease=lease)
        b.finish_case("I-xi001", HASH, outcome="failed", lease=lease)
        minimally_completed(b, lease, "I-xi002")
    assert b.status()["attempts"] == 2 and not b.status()["paused"]


@pytest.mark.parametrize("damage", ["journal_tail", "claim_truncated", "claim_removed"])
def test_storage_corruption_fails_closed_without_repair(tmp_path, damage):
    b = budget(tmp_path)
    with b.exclusive_run() as lease:
        minimally_completed(b, lease, "I-xi001")
    if damage == "journal_tail":
        with (b.root / "journal.jsonl").open("ab") as stream:
            stream.write(b'{"partial":')
    else:
        claim = next((b.root / "claims").iterdir())
        if damage == "claim_truncated": claim.write_bytes(b'{"partial":')
        else: claim.unlink()
    before = snapshot(b.root)
    with pytest.raises(old.TaskBudgetError):
        module.ExpandedBudget(b.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError):
        module.ExpandedBudget(b.root, HASH, module.fixed_plan(), initialize=True)
    assert snapshot(b.root) == before


def _contending_process(root, queue):
    try:
        b = module.ExpandedBudget(Path(root), HASH, module.fixed_plan())
        with b.exclusive_run(): queue.put("incorrectly-acquired")
    except old.TaskBudgetError as error:
        queue.put(error.code)


def test_independent_process_cannot_hold_a_second_run(tmp_path):
    b = budget(tmp_path)
    context = multiprocessing.get_context("spawn"); queue = context.Queue()
    with b.exclusive_run() as lease:
        child = context.Process(target=_contending_process, args=(str(b.root), queue))
        child.start(); child.join(10)
        assert child.exitcode == 0 and not child.is_alive()
        assert queue.get(timeout=1) == "ledger_busy"
        minimally_completed(b, lease, "I-xi001")
    assert module.ExpandedBudget(b.root, HASH, module.fixed_plan()).status()["attempts"] == 1


def test_old_ledgers_and_directory_copy_cannot_restore_allowance(tmp_path):
    from evaluation.current_task_ablation_budget import TaskAblationBudget, fixed_plan as old_plan
    from evaluation.prospective_budget import ProspectiveBudget, fixed_plan as prospective_plan
    b = budget(tmp_path, send_limit=1)
    with b.exclusive_run() as lease:
        minimally_completed(b, lease, "I-xi001")
    before = snapshot(b.root)
    for cls, plan in ((TaskAblationBudget, old_plan()), (ProspectiveBudget, prospective_plan())):
        with pytest.raises(old.TaskBudgetError):
            cls(b.root, HASH, plan)
    copied = tmp_path / "copied"; shutil.copytree(b.root, copied)
    with pytest.raises(old.TaskBudgetError):
        module.ExpandedBudget(copied, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError):
        module.ExpandedBudget(copied, HASH, module.fixed_plan(), initialize=True)
    assert snapshot(b.root) == before
