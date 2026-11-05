"""Append-only continuation admission, with temporary private ledgers only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from evaluation import coverage_budget as previous
from evaluation import coverage_continuation_budget as module


H, W = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("continuation accounting has no service calls")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def send(budget, lease, case, *, phase="intent", actor=None, turn="turn-1",
         outcome="received", code=None, sealed=True):
    event = budget.reserve(case, turn, phase, actor, H, W, lease=lease)
    budget.finish(event["operation_id"], outcome,
                  H if sealed else None, code, lease=lease)
    return event


def original(tmp_path, *, last_outcome="execution_error", last_code="timeout", sealed=True):
    root = tmp_path / "batch" / "ledger"
    root.parent.mkdir(mode=0o700)
    b = previous.CoverageBudget(root, H, previous.fixed_plan(), initialize=True)
    with b.exclusive_run() as lease:
        for identity, count in zip(previous.CASE_IDS[:6], (3, 2, 2, 3, 6, 3)):
            b.start_case(identity, lease=lease)
            send(b, lease, identity)
            for _ in range(count - 1):
                send(b, lease, identity, phase="agent", actor="general")
            b.finish_case(identity, H, lease=lease)
        b.start_case(module.ORIGINAL_CASE, lease=lease)
        send(b, lease, module.ORIGINAL_CASE, outcome=last_outcome,
             code=last_code, sealed=sealed)
    raw = (root / "journal.jsonl").read_bytes()
    assert len(raw.splitlines()) == 53
    assert b.status()["attempts"] == 20
    return root, raw, b.status()["chain_hash"]


def manifest(root, prefix, chain, mutate=None):
    value = {"schema_version": module.SCHEMA,
        "budget": {"original_protocol_sha256": H,
            "prefix_size": len(prefix), "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "prefix_chain_hash": chain, "already_spent": 20, "max_requests": 480},
        "previous_active_case": module.ORIGINAL_CASE,
        "remaining_jobs": previous.job_specs()[7:],
        "public_parameters": {"temperature": 0.1},
        "controlled_authorization": "new user continuation instruction, no new allowance"}
    if mutate:
        mutate(value)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    path = root.parent / module.MANIFEST_NAME
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def continued(root, prefix, chain, digest, **kwargs):
    return module.ContinuationBudget(root, H, previous.fixed_plan(),
        continuation_sha256=digest, prefix_size=len(prefix),
        prefix_sha256=hashlib.sha256(prefix).hexdigest(), prefix_chain_hash=chain,
        **kwargs)


@pytest.fixture
def frozen(tmp_path):
    root, raw, chain = original(tmp_path)
    digest = manifest(root, raw, chain)
    return root, raw, chain, digest


def test_same_genesis20_attempts_append_resume_without_rewriting_prefix_or_claims(frozen):
    root, raw, chain, digest = frozen
    identity = (root / "identity.json").read_bytes()
    old_claims = {p.name: p.read_bytes() for p in (root / "claims").iterdir()}
    budget = continued(*frozen)
    assert budget.status()["attempts"] == 20
    with budget.exclusive_run() as lease:
        event = budget.resume_after_timeout(W, lease=lease)
        assert event["event"] == "resume_after_timeout" and event["attempt"] == 20
        state = budget.status()
        assert state["completed_cases"] == list(previous.CASE_IDS[:7])
        assert state["active_case"] is None and state["paused"] is False
        assert state["consecutive_execution_failures"] == 1
        assert budget.results()["cases"][module.ORIGINAL_CASE] == {"outcome": "failed", "artifact_sha256": W}
        next_case = previous.CASE_IDS[7]
        budget.start_case(next_case, lease=lease)
        op = send(budget, lease, next_case)
        assert op["attempt"] == 21
        budget.finish_case(next_case, H, lease=lease)
    assert (root / "identity.json").read_bytes() == identity
    assert (root / "journal.jsonl").read_bytes().startswith(raw)
    assert all((root / "claims" / name).read_bytes() == data for name, data in old_claims.items())
    reopened = continued(*frozen)
    assert reopened.status()["attempts"] == 21
    assert reopened.status()["consecutive_execution_failures"] == 0
    with pytest.raises(previous.TaskBudgetError):
        previous.CoverageBudget(root, H, previous.fixed_plan())


@pytest.mark.parametrize("damage", ["allowance", "spent", "root-case", "plan", "float-budget", "duplicate"])
def test_manifest_bounds_and_suffix_cannot_be_reconfigured(tmp_path, damage):
    root, raw, chain = original(tmp_path)
    def change(v):
        if damage == "allowance": v["budget"]["max_requests"] = 960
        elif damage == "spent": v["budget"]["already_spent"] = 0
        elif damage == "root-case": v["previous_active_case"] = "A-CT-D01"
        elif damage == "plan": v["remaining_jobs"].reverse()
        elif damage == "float-budget": v["budget"]["already_spent"] = 20.0
    digest = manifest(root, raw, chain, change)
    if damage == "duplicate":
        p = root.parent / module.MANIFEST_NAME
        value = p.read_text().replace('"already_spent": 20', '"already_spent": 20, "already_spent": 20')
        p.write_text(value)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(module.TaskBudgetError):
        continued(root, raw, chain, digest)
    assert (root / "journal.jsonl").read_bytes() == raw
    assert not (root / module.RUN_CLAIM_NAME).exists()


def test_changed_manifest_after_construction_blocks_every_new_ledger_operation(frozen):
    root, raw, chain, digest = frozen
    b = continued(*frozen)
    p = root.parent / module.MANIFEST_NAME
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(module.TaskBudgetError):
        b.status()
    assert (root / "journal.jsonl").read_bytes() == raw


@pytest.mark.parametrize("kind", ["truncate", "change-prefix", "wrong-digest", "wrong-chain"])
def test_immutable_original_prefix_cannot_be_replaced(frozen, kind):
    root, raw, chain, digest = frozen
    if kind == "truncate": (root / "journal.jsonl").write_bytes(raw[:-1])
    elif kind == "change-prefix": (root / "journal.jsonl").write_bytes(raw.replace(b'"version":1', b'"version":2', 1))
    elif kind == "wrong-digest": raw += b"\n"
    else: chain = W
    with pytest.raises(module.TaskBudgetError):
        continued(root, raw, chain, digest)


@pytest.mark.parametrize("outcome,code,sealed", [
    ("execution_error", "auth", True), ("execution_error", "network", True),
    ("cancelled", "cancelled", True), ("unknown", None, True),
    ("execution_error", "timeout", False),
])
def test_only_exact_original_finished_timeout_is_resumable(tmp_path, outcome, code, sealed):
    root, raw, chain = original(tmp_path, last_outcome=outcome, last_code=code, sealed=sealed)
    digest = manifest(root, raw, chain)
    with pytest.raises(module.TaskBudgetError):
        continued(root, raw, chain, digest)


def test_restarts_or_concurrent_runs_cannot_resume_twice(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        other = continued(*frozen)
        with pytest.raises(module.TaskBudgetError, match="ledger_busy|paused"):
            with other.exclusive_run(): pass
        b.resume_after_timeout(H, lease=lease)
        with pytest.raises(module.TaskBudgetError):
            b.resume_after_timeout(H, lease=lease)
    with pytest.raises(module.TaskBudgetError, match="paused"):
        with continued(*frozen).exclusive_run(): pass


def test_failed_start_claim_is_preserved_and_prevents_restart(frozen):
    root, raw, chain, digest = frozen
    b = continued(*frozen)
    with b.exclusive_run():
        pass  # e.g. process failed before its resume event, no model request.
    assert (root / module.RUN_CLAIM_NAME).exists()
    assert (root / "journal.jsonl").read_bytes() == raw
    with pytest.raises(module.TaskBudgetError, match="paused"):
        with continued(*frozen).exclusive_run(): pass


def test_completed_cases_and_original_uncertain_case_cannot_be_resent(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for old in previous.CASE_IDS[:7]:
            with pytest.raises(module.TaskBudgetError): b.start_case(old, lease=lease)
            with pytest.raises(module.TaskBudgetError):
                b.reserve(old, "turn-1", "intent", None, H, W, lease=lease)
        assert b.status()["attempts"] == 20


@pytest.mark.parametrize("code", ["timeout", "network"])
def test_soft_failure_is_charged_and_next_independent_case_can_start(frozen, code):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        case = previous.CASE_IDS[7]
        b.start_case(case, lease=lease)
        send(b, lease, case, outcome="execution_error", code=code)
        assert b.status()["paused"]
        b.recover_case_failure(case, W, lease=lease)
        assert b.status()["attempts"] == 21
        assert not b.status()["paused"]
        assert b.results()["cases"][case]["outcome"] == "failed"
        b.start_case(previous.CASE_IDS[8], lease=lease)


def test_three_consecutive_case_faults_pause_with_no_refund(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for number, case in enumerate(previous.CASE_IDS[7:9], 2):
            b.start_case(case, lease=lease)
            send(b, lease, case, outcome="execution_error", code="timeout")
            b.recover_case_failure(case, H, lease=lease)
            assert b.status()["consecutive_execution_failures"] == number
        assert b.status()["paused"] and b.status()["attempts"] == 22
        with pytest.raises(module.TaskBudgetError): b.start_case(previous.CASE_IDS[9], lease=lease)


def test_successful_or_ordinary_failed_case_breaks_network_failure_streak(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        case = previous.CASE_IDS[7]
        b.start_case(case, lease=lease)
        send(b, lease, case, outcome="structure_failed")
        b.finish_case(case, H, outcome="failed", lease=lease)
        assert b.status()["consecutive_execution_failures"] == 0
        for case in previous.CASE_IDS[8:10]:
            b.start_case(case, lease=lease)
            send(b, lease, case, outcome="execution_error", code="network")
            b.recover_case_failure(case, H, lease=lease)
        assert not b.status()["paused"]


@pytest.mark.parametrize("outcome,code,sealed", [
    ("execution_error", "auth", True), ("execution_error", "quota", True),
    ("execution_error", "rate_limit", True), ("execution_error", "protocol", True),
    ("execution_error", "storage_failure", True), ("cancelled", "cancelled", True),
    ("unknown", None, True), ("execution_error", "timeout", False),
])
def test_nonrecoverable_or_unsealed_fault_never_reopens_admission(frozen, outcome, code, sealed):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        case = previous.CASE_IDS[7]
        b.start_case(case, lease=lease)
        send(b, lease, case, outcome=outcome, code=code, sealed=sealed)
        before = (b.root / "journal.jsonl").read_bytes()
        with pytest.raises(module.TaskBudgetError): b.recover_case_failure(case, H, lease=lease)
        assert b.status()["paused"]
        assert (b.root / "journal.jsonl").read_bytes() == before


def test_inflight_sibling_blocks_recovery_and_remains_charged(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        case = previous.CASE_IDS[7]
        b.start_case(case, lease=lease)
        a = b.reserve(case, "turn-1", "agent", "general", H, W, lease=lease)
        b.reserve(case, "turn-1", "agent", "billing", H, W, lease=lease)
        b.finish(a["operation_id"], "execution_error", H, "timeout", lease=lease)
        with pytest.raises(module.TaskBudgetError, match="pending"):
            b.recover_case_failure(case, H, lease=lease)
        assert b.status()["attempts"] == 22 and len(b.status()["pending"]) == 1
    with pytest.raises(module.TaskBudgetError):
        with continued(*frozen).exclusive_run(): pass


def test_failed_conversation_discards_remaining_turns_not_the_next_conversation(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for spec in previous.job_specs()[7:40]:
            case = spec["case_id"]
            b.start_case(case, lease=lease)
            for phase in ("rag_rewrite", "rag_rerank") if spec["suite"] == "retrieval" else ("intent",):
                send(b, lease, case, phase=phase)
            b.finish_case(case, H, lease=lease)
        case = previous.CASE_IDS[40]
        b.start_case(case, lease=lease)
        send(b, lease, case, turn="turn-1")
        send(b, lease, case, turn="turn-2", outcome="execution_error", code="timeout")
        b.recover_case_failure(case, H, lease=lease)
        count = b.status()["attempts"]
        for turn in ("turn-1", "turn-2", "turn-3", "turn-4", "turn-5"):
            with pytest.raises(module.TaskBudgetError):
                b.reserve(case, turn, "intent", None, H, W, lease=lease)
        assert b.status()["attempts"] == count
        assert b.results()["cases"][case]["outcome"] == "failed"
        b.start_case(previous.CASE_IDS[41], lease=lease)


def test_all_remaining129_positions_keep_plan_and_original20_charged(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for spec, plan in zip(previous.job_specs()[7:], previous.fixed_plan()[7:]):
            case = spec["case_id"]
            b.start_case(case, lease=lease)
            if spec["suite"] == "retrieval":
                send(b, lease, case, phase="rag_rewrite")
                send(b, lease, case, phase="rag_rerank")
            else:
                for turn in plan["turn_ids"]:
                    send(b, lease, case, turn=turn)
            b.finish_case(case, H, lease=lease)
    assert b.status()["completed_cases"] == list(previous.CASE_IDS)
    assert b.status()["attempts"] == 229  # 20 unchanged + 209 minimum suffix sends.
    assert b.results()["cases"][module.ORIGINAL_CASE]["outcome"] == "failed"


def test_same480_cap_includes_twenty_historical_reservations(frozen):
    b = continued(*frozen, send_limit=21)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        case = previous.CASE_IDS[7]
        b.start_case(case, lease=lease)
        send(b, lease, case)
        with pytest.raises(module.TaskBudgetError, match="quota"):
            b.reserve(case, "turn-1", "agent", "general", H, W, lease=lease)
        assert b.status()["attempts"] == 21


def test_default480_hard_cap_allows_only460_additional_attempts(frozen):
    b = continued(*frozen)
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for spec, plan in zip(previous.job_specs()[7:], previous.fixed_plan()[7:]):
            case = spec["case_id"]
            b.start_case(case, lease=lease)
            caps = previous.RAG_PHASE_CAPS if case.startswith("R-") else previous.TASK_PHASE_CAPS
            for turn in plan["turn_ids"]:
                for phase, cap in caps.items():
                    for _ in range(cap):
                        if b.status()["attempts"] == 480:
                            before = (b.root / "journal.jsonl").read_bytes()
                            with pytest.raises(module.TaskBudgetError, match="quota"):
                                b.reserve(case, turn, phase, "general" if phase == "agent" else None, H, W, lease=lease)
                            assert (b.root / "journal.jsonl").read_bytes() == before
                            assert b.status()["attempts"] - 20 == 460
                            return
                        send(b, lease, case, turn=turn, phase=phase, actor="general" if phase == "agent" else None)
            b.finish_case(case, H, lease=lease)
    pytest.fail("the fixed plan should reach the hard physical-send ceiling")


def test_corrupt_or_missing_run_claim_rejects_resumed_journal(frozen):
    root, _, _, _ = frozen
    b = continued(*frozen)
    with b.exclusive_run() as lease: b.resume_after_timeout(H, lease=lease)
    p = root / module.RUN_CLAIM_NAME
    p.write_bytes(b"broken\n")
    with pytest.raises(module.TaskBudgetError): continued(*frozen)


def test_continuation_cannot_initialize_missing_storage(tmp_path):
    with pytest.raises(module.TaskBudgetError):
        continued(tmp_path / "missing", b"placeholder\n", H, H)
    assert not (tmp_path / "missing").exists()
