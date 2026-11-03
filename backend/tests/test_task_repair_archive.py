"""Real temporary ledger chains; no historical writes, services or model calls."""
import socket

import pytest

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_repair_archive as module, task_trial as engine
from evaluation.task_budget import TaskBudget


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("archive verification cannot access services")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def archived(tmp_path, monkeypatch):
    root = tmp_path / "paused"
    root.mkdir(mode=0o700)
    legacy = {"controlled_old_archives": True}
    protocol = {"schema_version": "serveflow-task-protocol-v2", "batch": module.previous.BATCH,
        "git_head": module.EXECUTION_COMMIT, "root_binding_sha256": base._root_hash(root),
        "authorized_model_requests": 0, "prior_proof": legacy,
        "artifacts": {"old-code.py": "a" * 64}, "plan": engine._plan(engine.cases())}
    base._write(root, "protocol.json", protocol)
    digest = bench.sha256(root / "protocol.json")
    ledger = TaskBudget(root / "ledger", digest, protocol["plan"], initialize=True)
    with ledger.exclusive_run() as lease:
        # Preserve the original 4+10+5+9 per-case allocation and its unknown
        # e04 turn-2 Agent operation. Controlled outputs never claim quality.
        for number, count in ((1, 4), (2, 10), (3, 5), (4, 9)):
            case_id = f"e{number:02d}"
            ledger.start_case(case_id, lease=lease)
            for index in range(count):
                turn = "turn-2" if number == 4 and index >= 7 else "turn-1"
                phase, role = ("intent", None) if index == 0 or number == 4 and index == 7 else ("agent", "general")
                operation = ledger.reserve(case_id, turn, phase, role, "a" * 64, "b" * 64, lease=lease)
                if number == 4 and index == 8:
                    continue  # Retain the uncertain send without a physical file.
                name = f"physical-{operation['attempt']:03d}.json"
                base._write(root, name, {"kind": "controlled-archived-native-result"})
                ledger.finish(operation["operation_id"], "received", bench.sha256(root / name), lease=lease)
            if number < 4:
                base._write(root, f"case-{case_id}.json", {"kind": "controlled-old-case-not-rescored"})
                ledger.finish_case(case_id, bench.sha256(root / f"case-{case_id}.json"), outcome="success", lease=lease)
    base._write(root, "authorization.json", {"protocol_sha256": digest, "max_requests": 168,
        "root_binding_sha256": base._root_hash(root), "initial_chain_hash": ledger.initial_chain_hash,
        "plan": protocol["plan"]})
    for name in module.STOP_FILES:
        base._write(root, name, {"kind": "controlled-stop-marker"})
    for key, value in {"ROOT": root, "PROTOCOL_SHA256": digest,
            "AUTHORIZATION_SHA256": bench.sha256(root / "authorization.json"),
            "JOURNAL_SHA256": bench.sha256(root / "ledger/journal.jsonl"),
            "CHAIN_SHA256": ledger.status()["chain_hash"], "TREE_SHA256": module.previous._tree(root)}.items():
        monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(module.previous, "archived_proof", lambda: dict(legacy))
    reads = []
    monkeypatch.setattr(module.previous.history, "_archived_sources", lambda p, field: reads.append((p["git_head"], field)))
    return root, ledger, reads


def test_paused_28th_unknown_is_preserved_without_old_verify_rescore_or_allowance(tmp_path, monkeypatch):
    root, ledger, reads = archived(tmp_path, monkeypatch)
    before = tree(root)
    monkeypatch.setattr(module.previous.TaskRecheck, "verify", lambda *_: pytest.fail("old current-code verify forbidden"))
    monkeypatch.setattr(module.previous.TaskRecheck, "report", lambda *_: pytest.fail("old scoring forbidden"))
    proof = module.paused_predecessor_proof()
    assert proof["old_reserved_requests"] == 28 and proof["old_sealed_physical_results"] == 27
    assert proof["old_unknown_operations"] == ["e04:turn-2:agent:general:1"]
    assert proof["new_authorized_requests"] == 0 and not proof["old_allowance_reused"]
    assert not proof["old_scores_recomputed"] and len(proof["physical_sha256"]) == 27
    assert len(proof["case_sha256"]) == 3 and reads == [(module.EXECUTION_COMMIT, "artifacts")]
    assert tree(root) == before and ledger.status()["attempts"] == 28


@pytest.mark.parametrize("damage", ["source", "journal", "physical", "case", "stop", "old_unknown", "history", "authorization"])
def test_archive_changes_are_rejected_without_repair(tmp_path, monkeypatch, damage):
    root, _, _ = archived(tmp_path, monkeypatch)
    if damage == "source":
        def reject(*_):
            base._fail("archived_sources_changed")
        monkeypatch.setattr(module.previous.history, "_archived_sources", reject)
    elif damage == "history":
        monkeypatch.setattr(module.previous, "archived_proof", lambda: {"changed": True})
    else:
        name = {"journal": "ledger/journal.jsonl", "physical": "physical-001.json",
            "case": "case-e01.json", "stop": "operator-stop-v2.json", "old_unknown": "physical-028.json",
            "authorization": "authorization.json"}[damage]
        path = root / name
        path.write_text("{}")
        path.chmod(0o600)
    damaged = tree(root)
    with pytest.raises(base.TrialError):
        module.paused_predecessor_proof()
    assert tree(root) == damaged


def test_sealed_payload_hash_still_checked_when_manifest_tree_is_independently_rebound(tmp_path, monkeypatch):
    root, _, _ = archived(tmp_path, monkeypatch)
    (root / "physical-001.json").write_text("{}")
    monkeypatch.setattr(module, "TREE_SHA256", module.previous._tree(root))
    with pytest.raises(base.TrialError, match="task_repair_paused_physical_changed"):
        module.paused_predecessor_proof()
