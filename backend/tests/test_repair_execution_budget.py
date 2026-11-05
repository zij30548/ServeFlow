"""New sixteen-position quota, real temporary files; no real sends or stores."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import socket
import threading

import pytest

from evaluation import repair_execution_budget as module, task_budget as old


HASH, WIRE = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repair budget tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def make(tmp_path, send_limit=96):
    return module.RepairExecutionBudget(tmp_path / "ledger", HASH, module.fixed_plan(),
                                        initialize=True, send_limit=send_limit)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def received(budget, lease, identity, phase="intent", agent=None):
    claim = budget.reserve(identity, "turn-1", phase, agent, HASH, WIRE, lease=lease)
    budget.finish(claim["operation_id"], "received", HASH, lease=lease)
    return claim


def complete(budget, lease, identity, *, full=False):
    budget.start_case(identity, lease=lease)
    if full:
        for phase, maximum in module.TASK_PHASE_CAPS.items():
            for _ in range(maximum):
                received(budget, lease, identity, phase, "general" if phase == "agent" else None)
    elif identity.startswith("RR"):
        received(budget, lease, identity, "rag_rewrite")
        received(budget, lease, identity, "rag_rerank")
    else:
        received(budget, lease, identity)
    budget.finish_case(identity, HASH, lease=lease)


def test_fixed_plan_matches_prepared_positions_and_keeps_old_limits():
    from evaluation import repair_validation as prep
    cases = prep.bench.read_json(prep.bench.BACKEND_ROOT / prep.DATASET)
    assert list(module.CASE_IDS) == [row["id"] for suite in ("tasks", "retrieval") for row in cases[suite]]
    assert len(module.fixed_plan()) == 16 and all(row["turn_ids"] == ["turn-1"] for row in module.fixed_plan())
    assert module.CASE_IDS[:2] == ("RT01", "RT02") and module.CASE_IDS[-2:] == ("RR07", "RR08")
    assert module.MAX_REQUESTS == 96 and module.MATHEMATICAL_REQUESTS == prep.MATHEMATICAL_ENVELOPE == 200
    assert module.TASK_PER_TURN == 23 and module.RAG_PER_TURN == 2 and module.MAX_IN_FLIGHT == 3
    assert not module.LIMITS["judge_allowed"] and old.MAX_REQUESTS == 168 and old.PHASE_CAPS["judge"] == 1


@pytest.mark.parametrize("damage", ["reverse", "missing", "duplicate", "turn", "old_id", "fields", "tuple"])
def test_changed_plan_rejected_before_any_storage(tmp_path, damage):
    plan = module.fixed_plan()
    if damage == "reverse": plan.reverse()
    elif damage == "missing": plan.pop()
    elif damage == "duplicate": plan[1] = deepcopy(plan[0])
    elif damage == "turn": plan[0]["turn_ids"].append("turn-2")
    elif damage == "old_id": plan[0]["case_id"] = "A-et01"
    elif damage == "fields": plan[0]["allowance"] = 200
    else: plan[0]["turn_ids"] = ("turn-1",)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.RepairExecutionBudget(tmp_path / "bad", HASH, plan, initialize=True)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("limit", [0, -1, 97, 200, True, False, 1.0, "96", None])
def test_noninteger_or_oversize_grants_rejected_before_storage(tmp_path, limit):
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        make(tmp_path, limit)
    assert not (tmp_path / "ledger").exists()


def test_fixed_order_reopen_permissions_and_no_resend(tmp_path):
    plan = module.fixed_plan()
    budget = module.RepairExecutionBudget(tmp_path / "ledger", HASH, plan, initialize=True)
    plan[0]["case_id"] = "tampered-caller-object"
    with budget.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            budget.start_case("RT02", lease=lease)
        for identity in module.CASE_IDS:
            complete(budget, lease, identity)
    before = snapshot(budget.root)
    reopened = module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan())
    assert reopened.status()["attempts"] == 24 and reopened.status()["completed_cases"] == list(module.CASE_IDS)
    assert reopened.initial_chain_hash == budget.initial_chain_hash and snapshot(budget.root) == before
    assert budget.root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600) for p in budget.root.rglob("*"))
    with reopened.exclusive_run() as lease:
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            reopened.start_case("RT01", lease=lease)
    assert snapshot(budget.root) == before


def test_true_96th_request_is_atomic_and_97th_stays_rejected_after_restart(tmp_path):
    budget = make(tmp_path)
    with budget.exclusive_run() as lease:
        for identity in module.CASE_IDS[:4]:
            complete(budget, lease, identity, full=True)
        assert budget.status()["attempts"] == 92
        budget.start_case("RT05", lease=lease)
        received(budget, lease, "RT05")
        for _ in range(2): received(budget, lease, "RT05", "agent", "general")
        assert budget.status()["attempts"] == 95
        barrier = threading.Barrier(2)
        def contend(role):
            barrier.wait(timeout=2)
            try: return budget.reserve("RT05", "turn-1", "agent", role, HASH, WIRE, lease=lease)
            except old.TaskBudgetError as error: return error.code
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(contend, ("general", "billing")))
        granted = [row for row in results if isinstance(row, dict)]
        assert len(granted) == 1 and granted[0]["attempt"] == 96
        assert next(row for row in results if isinstance(row, str)) in {"quota", "ledger_busy"}
        budget.finish(granted[0]["operation_id"], "received", HASH, lease=lease)
        budget.finish_case("RT05", HASH, lease=lease)
    reopened = module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan())
    with reopened.exclusive_run() as lease:
        reopened.start_case("RT06", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(reopened, lease, "RT06")
    assert reopened.status()["attempts"] == 96 and len(reopened.results()["operations"]) == 96


def test_lower_protocol_bound_limit_and_three_inflight_are_not_bypassed(tmp_path):
    budget = make(tmp_path, 3)
    with budget.exclusive_run() as lease:
        budget.start_case("RT01", lease=lease)
        claims = [budget.reserve("RT01", "turn-1", "agent", role, HASH, WIRE, lease=lease)
                  for role in ("general", "billing", "technical")]
        with pytest.raises(old.TaskBudgetError, match="ledger_busy"):
            received(budget, lease, "RT01")
        for claim in claims: budget.finish(claim["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(budget, lease, "RT01")
        budget.finish_case("RT01", HASH, lease=lease)
    reopened = module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan(), send_limit=3)
    with reopened.exclusive_run() as lease:
        reopened.start_case("RT02", lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(reopened, lease, "RT02")
    assert reopened.status()["attempts"] == 3


def test_phase_caps_reject_judge_and_rag_has_exact_one_rewrite_then_optional_rerank(tmp_path):
    budget = make(tmp_path)
    with budget.exclusive_run() as lease:
        budget.start_case("RT01", lease=lease)
        for phase, cap in module.TASK_PHASE_CAPS.items():
            for _ in range(cap): received(budget, lease, "RT01", phase, "general" if phase == "agent" else None)
            with pytest.raises(old.TaskBudgetError, match="quota"):
                received(budget, lease, "RT01", phase, "general" if phase == "agent" else None)
        for phase, role in (("judge", None), ("embedding", None), ("agent", None), ("agent", "unknown")):
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                received(budget, lease, "RT01", phase, role)
        budget.finish_case("RT01", HASH, lease=lease)
        for identity in module.CASE_IDS[1:8]: complete(budget, lease, identity)
        budget.start_case("RR01", lease=lease)
        for phase, role in (("intent", None), ("agent", "general"), ("rag_rewrite", "general")):
            with pytest.raises(old.TaskBudgetError, match="protocol"):
                received(budget, lease, "RR01", phase, role)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            received(budget, lease, "RR01", "rag_rerank")
        claim = budget.reserve("RR01", "turn-1", "rag_rewrite", None, HASH, WIRE, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="protocol"):
            received(budget, lease, "RR01", "rag_rerank")
        budget.finish(claim["operation_id"], "received", HASH, lease=lease)
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(budget, lease, "RR01", "rag_rewrite")
        received(budget, lease, "RR01", "rag_rerank")
        with pytest.raises(old.TaskBudgetError, match="quota"):
            received(budget, lease, "RR01", "rag_rerank")
        budget.finish_case("RR01", HASH, lease=lease)
        budget.start_case("RR02", lease=lease)
        before = budget.status()["attempts"]
        received(budget, lease, "RR02", "rag_rewrite")
        budget.finish_case("RR02", HASH, lease=lease)
        assert budget.status()["attempts"] == before + 1


@pytest.mark.parametrize("window", ["case_start", "pending", "received_unsealed"])
def test_uncertain_prior_case_cannot_automatically_resume(tmp_path, window):
    budget = make(tmp_path)
    with budget.exclusive_run() as lease:
        budget.start_case("RT01", lease=lease)
        if window != "case_start":
            claim = budget.reserve("RT01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
            if window == "received_unsealed": budget.finish(claim["operation_id"], "received", HASH, lease=lease)
    before = snapshot(budget.root)
    reopened = module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError, match="pending"):
        with reopened.exclusive_run(): pass
    assert reopened.status()["attempts"] == (0 if window == "case_start" else 1)
    assert snapshot(budget.root) == before


@pytest.mark.parametrize("outcome,code", [("execution_error", "timeout"), ("execution_error", "network"),
    ("execution_error", "auth"), ("execution_error", "quota"), ("execution_error", "rate_limit"), ("execution_error", "storage_failure"),
    ("cancelled", "cancelled"), ("unknown", None)])
def test_failed_cancelled_unknown_attempts_stay_spent_and_pause(tmp_path, outcome, code):
    budget = make(tmp_path, 1)
    with budget.exclusive_run() as lease:
        budget.start_case("RT01", lease=lease)
        claim = budget.reserve("RT01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        budget.finish(claim["operation_id"], outcome, error_code=code, lease=lease)
        assert budget.status()["attempts"] == 1 and budget.status()["paused"]
    reopened = module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan(), send_limit=1)
    with pytest.raises(old.TaskBudgetError, match="paused"):
        with reopened.exclusive_run(): pass


def test_structure_failure_is_retained_and_can_continue_without_repair(tmp_path):
    budget = make(tmp_path)
    with budget.exclusive_run() as lease:
        budget.start_case("RT01", lease=lease)
        claim = budget.reserve("RT01", "turn-1", "intent", None, HASH, WIRE, lease=lease)
        budget.finish(claim["operation_id"], "structure_failed", HASH, lease=lease)
        budget.finish_case("RT01", HASH, outcome="failed", lease=lease)
        complete(budget, lease, "RT02")
    assert budget.status()["attempts"] == 2 and not budget.status()["paused"]


@pytest.mark.parametrize("damage", ["journal", "claim_removed", "claim_truncated"])
def test_counter_corruption_stops_without_repair(tmp_path, damage):
    budget = make(tmp_path)
    with budget.exclusive_run() as lease: complete(budget, lease, "RT01")
    if damage == "journal":
        with (budget.root / "journal.jsonl").open("ab") as stream: stream.write(b'{"partial":')
    else:
        path = next((budget.root / "claims").iterdir())
        path.unlink() if damage == "claim_removed" else path.write_bytes(b'{"partial":')
    before = snapshot(budget.root)
    with pytest.raises(old.TaskBudgetError): module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError): module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan(), initialize=True)
    assert snapshot(budget.root) == before


def _contending_process(root, queue):
    try:
        budget = module.RepairExecutionBudget(Path(root), HASH, module.fixed_plan())
        with budget.exclusive_run(): queue.put("incorrectly-acquired")
    except old.TaskBudgetError as error:
        queue.put(error.code)


def test_independent_process_cannot_own_a_second_run_lease(tmp_path):
    budget = make(tmp_path)
    ctx = multiprocessing.get_context("spawn"); queue = ctx.Queue()
    with budget.exclusive_run() as lease:
        child = ctx.Process(target=_contending_process, args=(str(budget.root), queue))
        child.start(); child.join(10)
        assert child.exitcode == 0 and not child.is_alive()
        assert queue.get(timeout=1) == "ledger_busy"
        complete(budget, lease, "RT01")
    assert module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan()).status()["attempts"] == 1


def test_copy_alias_old_identity_or_reinitialize_cannot_restore_allowance(tmp_path):
    from evaluation.expanded_budget import ExpandedBudget, fixed_plan as old_plan
    budget = make(tmp_path, 1)
    with budget.exclusive_run() as lease: complete(budget, lease, "RT01")
    before = snapshot(budget.root)
    with pytest.raises(old.TaskBudgetError): ExpandedBudget(budget.root, HASH, old_plan())
    with pytest.raises(old.TaskBudgetError): module.RepairExecutionBudget(budget.root, "c"*64, module.fixed_plan())
    with pytest.raises(old.TaskBudgetError): module.RepairExecutionBudget(budget.root, HASH, module.fixed_plan(), initialize=True)
    target = tmp_path / "copied"; shutil.copytree(budget.root, target)
    with pytest.raises(old.TaskBudgetError): module.RepairExecutionBudget(target, HASH, module.fixed_plan())
    link = tmp_path / "alias"; link.symlink_to(budget.root, target_is_directory=True)
    with pytest.raises(old.TaskBudgetError, match="configuration"):
        module.RepairExecutionBudget(link, HASH, module.fixed_plan())
    assert snapshot(budget.root) == before


def test_new_counter_does_not_authorize_or_modify_a_legacy_ledger(tmp_path):
    from evaluation.expanded_budget import ExpandedBudget, fixed_plan as old_plan
    legacy = ExpandedBudget(tmp_path / "legacy", HASH, old_plan(), initialize=True)
    before = snapshot(legacy.root)
    with pytest.raises(old.TaskBudgetError):
        module.RepairExecutionBudget(legacy.root, HASH, module.fixed_plan())
    assert snapshot(legacy.root) == before
    repaired = make(tmp_path)
    assert repaired.status()["attempts"] == 0
    assert not (repaired.root.parent / "authorization.json").exists()
    # A lower limit is only a counter input; the runner must authenticate and
    # reapply its immutable protocol/root/genesis-bound grant on every reopen.
    limited = module.RepairExecutionBudget(repaired.root, HASH, module.fixed_plan(), send_limit=1)
    assert limited.status()["attempts"] == 0 and limited.send_limit == 1
