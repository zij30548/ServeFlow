"""Second continuation accounting; synthetic private ledgers, zero services."""
import hashlib
import json
import fcntl
import os
from pathlib import Path
import shutil
import socket

import pytest

from evaluation import coverage_budget as plan
from evaluation import coverage_continuation_budget as first
from evaluation import coverage_continuation2_budget as module
from test_coverage_continuation_budget import original, manifest, continued, send, H, W


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError("second-continuation tests cannot access services")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture(scope="module")
def snapshot201(tmp_path_factory):
    temp = tmp_path_factory.mktemp("synthetic-coverage-201")
    root, initial, chain = original(temp)
    first_hash = manifest(root, initial, chain)
    b = continued(root, initial, chain, first_hash)
    extra = 88
    with b.exclusive_run() as lease:
        b.resume_after_timeout(H, lease=lease)
        for spec, turns in zip(plan.job_specs()[7:69], plan.fixed_plan()[7:69]):
            case = spec["case_id"]
            b.start_case(case, lease=lease)
            if spec["suite"] == "retrieval":
                send(b, lease, case, phase="rag_rewrite")
                send(b, lease, case, phase="rag_rerank")
            else:
                for turn in turns["turn_ids"]:
                    send(b, lease, case, turn=turn)
                    if extra:
                        for _ in range(min(2, extra)):
                            send(b, lease, case, phase="agent", actor="general", turn=turn)
                            extra -= 1
            b.finish_case(case, H, lease=lease)
        assert extra == 0 and b.status()["attempts"] == 194
        case = plan.CASE_IDS[69]
        assert case == module.FAILED_CASE
        b.start_case(case, lease=lease)
        for phase, actor in (("intent", None), ("agent", "technical"),
                ("agent", "billing"), ("rag_rewrite", "billing"),
                ("rag_rerank", "billing"), ("agent", "billing")):
            send(b, lease, case, phase=phase, actor=actor)
        send(b, lease, case, phase="rag_rewrite", actor="billing",
             outcome="execution_error", code="timeout")
    prefix = (root / "journal.jsonl").read_bytes()
    assert len(prefix.splitlines()) == 541 and b.status()["attempts"] == 201
    archive = temp / "fixture-snapshot-not-a-live-ledger"
    shutil.copytree(root.parent, archive)
    # Only this pytest-owned synthetic path is restored between independent
    # tests. No real evidence directory is copied, initialized or reset.
    return root, initial, prefix, b.status()["chain_hash"], archive


@pytest.fixture
def frozen(snapshot201):
    root, initial, prefix, chain, archive = snapshot201
    shutil.rmtree(root.parent)
    shutil.copytree(archive, root.parent)
    v = {"schema_version": module.SCHEMA,
        "original_continuation_sha256": hashlib.sha256((root.parent / first.MANIFEST_NAME).read_bytes()).hexdigest(),
        "budget": {"original_protocol_sha256": H, "prefix_size": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "prefix_chain_hash": chain,
            "prefix_events": 541, "already_spent": 201, "max_requests": 480},
        "previous_active_case": module.FAILED_CASE, "remaining_jobs": plan.job_specs()[70:],
        "parameters": {"temperature": 0.1}}
    p = root.parent / module.MANIFEST
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")
    p.chmod(0o600)
    return root, initial, prefix, chain, hashlib.sha256(p.read_bytes()).hexdigest()


def make(frozen, **kwargs):
    root, _, _, _, digest = frozen
    return module.Continuation2Budget(root, H, plan.fixed_plan(),
        continuation2_sha256=digest, **kwargs)


def revise(frozen, change):
    root, a, b, c, _ = frozen
    p = root.parent / module.MANIFEST
    value = json.loads(p.read_text())
    change(value)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return root, a, b, c, hashlib.sha256(p.read_bytes()).hexdigest()


def test_both_prefixes_original_claim_and_identity_are_unchanged(frozen):
    root, old53, prefix201, _, _ = frozen
    before = {p: p.read_bytes() for p in [root / "identity.json", root / "run.lock",
        root / first.RUN_CLAIM_NAME, root.parent / first.MANIFEST_NAME]}
    claims = {p.name: p.read_bytes() for p in (root / "claims").iterdir()}
    b = make(frozen)
    assert b.status()["attempts"] == 201
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, W, lease=lease)
        assert b.status()["completed_cases"] == list(plan.CASE_IDS[:70])
        assert b.status()["attempts"] == 201 and not b.status()["paused"]
        b.start_case(plan.CASE_IDS[70], lease=lease)
        send(b, lease, plan.CASE_IDS[70])
        b.finish_case(plan.CASE_IDS[70], H, lease=lease)
    assert b.status()["attempts"] == 202
    assert all(p.read_bytes() == raw for p, raw in before.items())
    journal = (root / "journal.jsonl").read_bytes()
    assert journal.startswith(prefix201) and journal.startswith(old53)
    assert all((root / "claims" / name).read_bytes() == raw for name, raw in claims.items())
    assert b.results()["cases"][module.FAILED_CASE] == {"outcome": "failed", "artifact_sha256": W}


@pytest.mark.parametrize("damage", ["allowance", "spent", "events", "case", "suffix", "oldmanifest", "bool", "protocol"])
def test_manifest_cannot_change_authority_or_prefix(frozen, damage):
    def change(v):
        if damage == "allowance": v["budget"]["max_requests"] = 681
        elif damage == "spent": v["budget"]["already_spent"] = 0
        elif damage == "events": v["budget"]["prefix_events"] = 540
        elif damage == "case": v["previous_active_case"] = "D-CT-D04"
        elif damage == "suffix": v["remaining_jobs"].insert(0, plan.job_specs()[69])
        elif damage == "oldmanifest": v["original_continuation_sha256"] = H
        elif damage == "bool": v["budget"]["prefix_events"] = True
        elif damage == "protocol": v["budget"]["original_protocol_sha256"] = W
    changed = revise(frozen, change)
    before = (frozen[0] / "journal.jsonl").read_bytes()
    with pytest.raises(module.TaskBudgetError): make(changed)
    assert (frozen[0] / "journal.jsonl").read_bytes() == before
    assert not (frozen[0] / module.RUN_CLAIM).exists()


@pytest.mark.parametrize("damage", ["truncate201", "old53", "firstmanifest", "firstclaim", "caseclaim"])
def test_original_evidence_damage_rejects_before_lease(frozen, damage):
    root = frozen[0]
    if damage == "truncate201": (root / "journal.jsonl").write_bytes(frozen[2][:-1])
    elif damage == "old53": (root / "journal.jsonl").write_bytes(b" " + frozen[2][1:])
    elif damage == "firstmanifest":
        p = root.parent / first.MANIFEST_NAME; p.write_bytes(p.read_bytes() + b" ")
    elif damage == "firstclaim": (root / first.RUN_CLAIM_NAME).write_bytes(b"corrupt\n")
    else:
        p = next((root / "claims").iterdir()); p.write_bytes(b"corrupt\n")
    with pytest.raises(module.TaskBudgetError): make(frozen)
    assert not (root / module.RUN_CLAIM).exists()


def test_second_manifest_change_during_lease_blocks_new_send(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        revise(frozen, lambda v: v.update(extra="changed"))
        with pytest.raises(module.TaskBudgetError): b.start_case(plan.CASE_IDS[70], lease=lease)


def test_same_run_lock_and_new_claim_prevent_parallel_or_restart(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        other = make(frozen)
        with pytest.raises(module.TaskBudgetError):
            with other.exclusive_run(): pass
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
    with pytest.raises(module.TaskBudgetError):
        with make(frozen).exclusive_run(): pass


def test_existing_run_lock_blocks_second_continuation_before_claim(frozen):
    b = make(frozen)
    fd = os.open(b.root / "run.lock", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(module.TaskBudgetError, match="ledger_busy"):
            with b.exclusive_run(): pass
        assert not (b.root / module.RUN_CLAIM).exists()
        assert b.status()["attempts"] == 201
    finally:
        os.close(fd)


def test_crash_before_first_event_keeps201_charge_and_one_shot_claim(frozen):
    b = make(frozen)
    with b.exclusive_run(): pass
    assert b.status()["attempts"] == 201
    assert (frozen[0] / module.RUN_CLAIM).exists()
    with pytest.raises(module.TaskBudgetError):
        with make(frozen).exclusive_run(): pass


def test_unfinished_send_stays_charged_and_blocks_recovery(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        next_case = plan.CASE_IDS[70]
        b.start_case(next_case, lease=lease)
        b.reserve(next_case, "turn-1", "intent", None, H, W, lease=lease)
    assert make(frozen).status()["attempts"] == 202
    with pytest.raises(module.TaskBudgetError):
        with make(frozen).exclusive_run(): pass


def test_old_cases_and_A14_never_reexecute(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        for identity in plan.CASE_IDS[:70]:
            with pytest.raises(module.TaskBudgetError): b.start_case(identity, lease=lease)
        assert b.status()["attempts"] == 201


def test_v1_can_read_same_event_vocabulary_but_cannot_start_another_run(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease: b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
    root, old53, _, _, _ = frozen
    value = json.loads((root.parent / first.MANIFEST_NAME).read_text())
    p = value["budget"]
    old = first.ContinuationBudget(root, H, plan.fixed_plan(),
        continuation_sha256=hashlib.sha256((root.parent / first.MANIFEST_NAME).read_bytes()).hexdigest(),
        prefix_size=p["prefix_size"], prefix_sha256=p["prefix_sha256"], prefix_chain_hash=p["prefix_chain_hash"])
    assert old.status()["attempts"] == 201
    with pytest.raises(module.TaskBudgetError):
        with old.exclusive_run(): pass


def test_all66_suffix_positions_finish_once_with_no_new_allowance(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        for spec, turns in zip(plan.job_specs()[70:], plan.fixed_plan()[70:]):
            identity = spec["case_id"]
            b.start_case(identity, lease=lease)
            if spec["suite"] == "retrieval":
                send(b, lease, identity, phase="rag_rewrite")
                send(b, lease, identity, phase="rag_rerank")
            else:
                for turn in turns["turn_ids"]: send(b, lease, identity, turn=turn)
            b.finish_case(identity, H, lease=lease)
    assert b.status()["completed_cases"] == list(plan.CASE_IDS)
    assert b.status()["attempts"] == 323


def test201_plus279_is_the_same480_hard_cap(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        for spec, turns in zip(plan.job_specs()[70:], plan.fixed_plan()[70:]):
            identity = spec["case_id"]
            b.start_case(identity, lease=lease)
            caps = plan.RAG_PHASE_CAPS if spec["suite"] == "retrieval" else plan.TASK_PHASE_CAPS
            for turn in turns["turn_ids"]:
                for phase, cap in caps.items():
                    for _ in range(cap):
                        if b.status()["attempts"] == 480:
                            before = (b.root / "journal.jsonl").read_bytes()
                            with pytest.raises(module.TaskBudgetError, match="quota"):
                                b.reserve(identity, turn, phase, "general" if phase == "agent" else None, H, W, lease=lease)
                            assert (b.root / "journal.jsonl").read_bytes() == before
                            assert b.status()["attempts"] - 201 == 279
                            return
                        send(b, lease, identity, phase=phase, actor="general" if phase == "agent" else None, turn=turn)
            b.finish_case(identity, H, lease=lease)
    pytest.fail("the fixed phase envelope should reach 480")


@pytest.mark.parametrize("outcome,code", [("execution_error", "auth"),
    ("unknown", "timeout"), ("cancelled", "cancelled")])
def test_later_auth_unknown_or_cancel_remains_fatal(frozen, outcome, code):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        identity = plan.CASE_IDS[70]
        b.start_case(identity, lease=lease)
        send(b, lease, identity, outcome=outcome, code=code)
        with pytest.raises(module.TaskBudgetError): b.recover_case_failure(identity, H, lease=lease)
        assert b.status()["paused"]


def test_third_consecutive_case_execution_fault_still_pauses(frozen):
    b = make(frozen)
    with b.exclusive_run() as lease:
        b.recover_case_failure(module.FAILED_CASE, H, lease=lease)
        assert b.status()["consecutive_execution_failures"] == 1
        for identity, code in zip(plan.CASE_IDS[70:72], ["network", "timeout"]):
            b.start_case(identity, lease=lease)
            send(b, lease, identity, outcome="execution_error", code=code)
            b.recover_case_failure(identity, H, lease=lease)
        assert b.status()["consecutive_execution_failures"] == 3
        assert b.status()["attempts"] == 203 and b.status()["paused"]
        assert b.status()["active_case"] is None
        with pytest.raises(module.TaskBudgetError):
            b.start_case(plan.CASE_IDS[72], lease=lease)
