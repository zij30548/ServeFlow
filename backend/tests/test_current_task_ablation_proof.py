"""Synthetic ancestry files only: no actual private archive, store or network."""
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from evaluation import current_task_ablation_proof as proof


REAL_TRACKED = proof._tracked_paths


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def json_write(path, value):
    write(path, json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("current task proof tests forbid network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    root = tmp_path.resolve() / "backend"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(proof.bench, "BACKEND_ROOT", root)
    monkeypatch.setattr(proof, "ROOT", root / "data/benchmarks/current-task-ablation-v1")
    monkeypatch.setattr(proof, "OLD_ROOT", root / "data/benchmarks/order-answer-recheck-v1")
    monkeypatch.setattr(proof, "OLD_REPORT", root / "data/benchmarks/order-answer-recheck-preflight-v1/final-reviewed-report.json")
    names = ["core/order_answers.py", "core/order_evidence.py", "core/diagnostic_evidence.py",
        "agents/agent_orchestrator.py", "evaluation/task_supervisor.py", "tooling/tool_manager.py",
        "knowledge/serveflow-demo-v2.json", "skills/billing/SKILL.md", "requirements.txt",
        "evaluation/benchmark/prospective-tasks-v1.json", "evaluation/prospective_task_runtime.py",
        "evaluation/quality_arm_runtime.py", "tests/test_order_answers.py", "tests/test_benchmark.py"]
    baseline = {name: ("pinned-7ab-" + name).encode() for name in names}
    baseline.update({f"evaluation/frozen_{i:03d}.py": f"fixed source {i}".encode()
                     for i in range(proof.BASELINE_FILES - len(names))})
    for name, raw in baseline.items():
        write(root / name, raw)
    fixed = ["knowledge/serveflow-demo-v2.json", "requirements.txt", "tests/test_benchmark.py"]
    monkeypatch.setattr(proof.bench, "artifact_paths", lambda _: list(fixed))
    monkeypatch.setattr(proof, "_tracked_paths", lambda: set(baseline))
    git_calls = []
    def git(commit, paths):
        git_calls.append((commit, list(paths)))
        assert commit == proof.BASELINE and len(paths) <= 256
        return {name: baseline[name.removeprefix("backend/")] for name in paths}
    monkeypatch.setattr(proof, "_git_blobs", git)
    hook = b"reviewed fixed current-task worker + exact root + 3600 only"
    write(root / proof.HOOK_PATH, hook)
    monkeypatch.setattr(proof, "HOOK_SHA256", digest(hook))
    for name in proof.NEW_FILES:
        write(root / name, ("new offline evaluation file " + name).encode())
    def current():
        return {str(path.relative_to(root)): digest(path.read_bytes())
                for path in sorted(root.rglob("*")) if path.is_file()
                and path.relative_to(root).parts[0] != "data"}
    monkeypatch.setattr(proof.ancestry, "_current_sources", current)
    older_calls = []
    older = {"paused_57": {"reserved_requests": 57, "paused": True},
             "fixed_16": {"reserved_requests": 16, "authorized_requests": 138,
                           "recorded_strict_result": {"passed": 2, "planned": 6, "rescored": False}},
             "old_allowance_reused": False, "old_scores_recomputed": False}
    def previous():
        older_calls.append(True)
        return deepcopy(older)
    monkeypatch.setattr(proof.previous, "prior_proof", previous)
    return SimpleNamespace(root=root, baseline=baseline, git_calls=git_calls,
                           older=older, older_calls=older_calls)


@pytest.fixture
def historical(fixture, monkeypatch):
    root = proof.OLD_ROOT
    plan = proof.fixed_plan()
    cases = [row["case_id"] for row in plan]
    protocol = {"schema_version": "serveflow-order-answer-recheck-protocol-v1",
                "git_head": proof.OLD_COMMIT, "plan": deepcopy(plan)}
    json_write(root / "protocol.json", protocol)
    protocol_hash = digest((root / "protocol.json").read_bytes())
    authorization = {"schema_version": "serveflow-order-answer-recheck-authorization-v1",
        "protocol_sha256": protocol_hash, "max_requests": 24, "plan": deepcopy(plan),
        "root_binding_sha256": proof.base._root_hash(root), "initial_chain_hash": "1" * 64}
    json_write(root / "authorization.json", authorization)
    write(root / "ledger/journal.jsonl", b"six completed cases; sixteen original attempts")
    for index in range(1, 17):
        json_write(root / f"physical-{index:03d}.json", {"attempt": index, "original": True})
    rows = []
    for key in cases:
        json_write(root / f"case-{key}.json", {"original_case": key})
        artifact_hash = digest((root / f"case-{key}.json").read_bytes())
        review = {"logical_case_id": key, "artifact_sha256": artifact_hash,
                  "recorded_assessment": "original, not recalculated"}
        json_write(root / f"review-{key}.json", review)
        rows.append({"logical_case_id": key, "status": "passed", "tool_status": "passed",
                     "artifact_sha256": artifact_hash, "review": review})
    for index in range(74 - len(proof._tree_files(root))):
        write(root / f"frozen-{index:03d}.json", b"unchanged claims, requests and data seals")
    report = {"schema_version": "serveflow-order-answer-recheck-progress-v1",
        "protocol_sha256": protocol_hash, "trial_kind": "order-answer-recheck-v1",
        "candidate_baseline": proof.previous.BASELINE, "trial_request_limit": 24,
        "reserved_requests": 16, "sealed_physical": 16, "pending_physical": 0,
        "paused": False, "planned_jobs": 6, "completed_jobs": 6,
        "semantic_reviews_imported": 6, "review_ready": True,
        "counts": deepcopy(proof._STORED_COUNTS), "tool_counts": deepcopy(proof._STORED_COUNTS),
        "cases": rows}
    json_write(proof.OLD_REPORT, report)
    tree = proof._tree_files(root)
    monkeypatch.setattr(proof, "OLD_PROTOCOL", protocol_hash)
    monkeypatch.setattr(proof, "OLD_AUTHORIZATION", tree["authorization.json"])
    monkeypatch.setattr(proof, "OLD_JOURNAL", tree["ledger/journal.jsonl"])
    monkeypatch.setattr(proof, "OLD_TREE", proof.canonical_payload_hash(tree))
    monkeypatch.setattr(proof, "OLD_CHAIN", "c" * 64)
    monkeypatch.setattr(proof, "OLD_REPORT_SHA256", digest(proof.OLD_REPORT.read_bytes()))
    state = {"attempts": 16, "completed_cases": list(cases), "active_case": None,
             "pending": [], "paused": False, "chain_hash": "c" * 64}
    sealed = {"operations": {str(i): {} for i in range(16)}, "cases": {key: {} for key in cases}}
    replays = []
    def ledger(path, protocol_sha256, requested_plan):
        replays.append((path, protocol_sha256, deepcopy(requested_plan)))
        return SimpleNamespace(initial_chain_hash="1" * 64,
            status=lambda: deepcopy(state), results=lambda: deepcopy(sealed))
    monkeypatch.setattr(proof, "DiagnosticRepairBudget", ledger)
    return SimpleNamespace(fixture=fixture, root=root, tree=tree, state=state, sealed=sealed,
        cases=cases, plan=plan, protocol=protocol, authorization=authorization,
        report=report, replays=replays)


def bind_mutated_history(historical, monkeypatch):
    """Only synthetic tests may rebind digests to reach deeper identity checks."""
    tree = proof._tree_files(historical.root)
    for name, field in [("protocol.json", "OLD_PROTOCOL"), ("authorization.json", "OLD_AUTHORIZATION"),
                        ("ledger/journal.jsonl", "OLD_JOURNAL")]:
        monkeypatch.setattr(proof, field, tree[name])
    monkeypatch.setattr(proof, "OLD_TREE", proof.canonical_payload_hash(tree))
    monkeypatch.setattr(proof, "OLD_REPORT_SHA256", digest(proof.OLD_REPORT.read_bytes()))


def test_exact_198_baseline_and_six_additions_bind_204_current_files(fixture, monkeypatch):
    monkeypatch.setattr(proof, "prior_proof", lambda: pytest.fail("source proof must not replay archives"))
    value = proof.source_proof()
    assert value["baseline_commit"] == "7abdb6216a3a7eb0742b4b406ec58e48aa30c4a7"
    assert (value["baseline_files"], value["current_files"]) == (198, 204)
    assert value["closed_additions"] == list(proof.NEW_FILES)
    assert value["business_changes_after_baseline"] is False
    assert value["model_authorization_granted"] == 0
    assert len(proof.artifacts()) == 204
    assert len(fixture.git_calls) == 1 and len(fixture.git_calls[0][1]) == 198


@pytest.mark.parametrize("name", ["core/order_answers.py", "core/order_evidence.py", "core/diagnostic_evidence.py",
    "agents/agent_orchestrator.py", "tooling/tool_manager.py", "knowledge/serveflow-demo-v2.json",
    "skills/billing/SKILL.md", "evaluation/benchmark/prospective-tasks-v1.json", "requirements.txt",
    "evaluation/prospective_task_runtime.py", "evaluation/quality_arm_runtime.py", "tests/test_order_answers.py",
    "evaluation/task_supervisor.py"])
def test_business_questions_old_runtime_and_tests_cannot_change(fixture, name):
    path = fixture.root / name
    write(path, path.read_bytes() + b"unexpected change")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.source_proof()


def test_no_generic_ignore_hook_or_incomplete_pending_hash(fixture, monkeypatch):
    monkeypatch.setattr(proof, "HOOK_SHA256", "pending")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.source_proof()


@pytest.mark.parametrize("action", ["extra", "missing", "renamed"])
def test_undeclared_source_or_absent_new_file_stops_before_git_read(fixture, action):
    path = fixture.root / proof.NEW_FILES[0]
    if action == "extra":
        write(fixture.root / "evaluation/unapproved.py", b"unapproved helper")
    elif action == "missing":
        path.unlink()
    else:
        path.rename(path.with_name("renamed.py"))
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.source_proof()
    assert fixture.git_calls == []


def test_new_preparation_files_are_hashed_without_fabricated_git_ancestry(fixture):
    before = proof.artifacts()
    path = fixture.root / proof.NEW_FILES[2]
    write(path, path.read_bytes() + b"local preparation change")
    after = proof.source_proof()
    assert proof.artifacts()[proof.NEW_FILES[2]] != before[proof.NEW_FILES[2]]
    assert after["current_source_sha256"] != proof.canonical_payload_hash(before)
    assert all("backend/" + name not in fixture.git_calls[-1][1] for name in proof.NEW_FILES)


@pytest.mark.parametrize("mutation", ["missing_fixed", "overlap", "wrong_count"])
def test_baseline_closure_must_be_complete_and_disjoint(fixture, monkeypatch, mutation):
    tracked = set(fixture.baseline)
    if mutation == "missing_fixed":
        tracked.remove("requirements.txt")
    elif mutation == "overlap":
        tracked.add(proof.NEW_FILES[0])
    else:
        tracked.add("evaluation/extra_old.py")
    monkeypatch.setattr(proof, "_tracked_paths", lambda: tracked)
    with pytest.raises(Exception, match="baseline_closure_invalid"):
        proof.source_proof()


def test_all_three_historical_results_bound_without_old_verify_or_rescore(historical, monkeypatch):
    monkeypatch.setattr(proof, "source_proof", lambda: pytest.fail("history does not inspect new runtime"))
    result = proof.prior_proof()
    assert result["older_trials"] == historical.fixture.older
    assert len(historical.fixture.older_calls) == 1
    latest = result["latest_fixed_16"]
    assert latest["reserved_requests"] == 16 and latest["authorized_requests"] == 24
    assert latest["completed_cases"] == latest["reviews"] == 6
    assert latest["recorded_strict_result"] == latest["recorded_tool_result"] == {
        "passed": 6, "planned": 6, "rescored": False}
    assert latest["report_sha256"] == proof.OLD_REPORT_SHA256
    assert result["old_allowance_reused"] is result["old_scores_recomputed"] is False
    assert result["new_authorization_granted"] == 0
    assert historical.replays == [(historical.root / "ledger", proof.OLD_PROTOCOL, historical.plan)]
    assert proof._tree_files(historical.root) == historical.tree


@pytest.mark.parametrize("name", ["protocol.json", "authorization.json", "ledger/journal.jsonl",
    "case-D-p01.json", "review-D-p07.json", "physical-001.json", "frozen-000.json"])
def test_any_historical_bytes_changed_blocks_before_ledger_replay(historical, name):
    path = historical.root / name
    write(path, path.read_bytes() + b"changed")
    with pytest.raises(Exception, match="prior_history_changed"):
        proof.prior_proof()
    assert historical.replays == []


def test_external_stored_report_hash_is_pinned_independently_of_trial_tree(historical):
    write(proof.OLD_REPORT, proof.OLD_REPORT.read_bytes() + b" ")
    with pytest.raises(Exception, match="prior_history_changed"):
        proof.prior_proof()
    assert historical.replays == []


@pytest.mark.parametrize("field,value", [("attempts", 0), ("attempts", True), ("completed_cases", []),
    ("active_case", "D-p09"), ("pending", ["uncertain"]), ("paused", True), ("chain_hash", "a" * 64)])
def test_reinitialized_paused_uncertain_or_different_chain_is_rejected(historical, field, value):
    historical.state[field] = value
    with pytest.raises(Exception, match="prior_state_changed"):
        proof.prior_proof()


@pytest.mark.parametrize("field,value", [("max_requests", 138), ("max_requests", True),
    ("root_binding_sha256", "a" * 64), ("initial_chain_hash", "b" * 64),
    ("plan", []), ("schema_version", "old-generic-authorization")])
def test_latest_authorization_keeps_24_limit_root_genesis_and_plan(historical, monkeypatch, field, value):
    historical.authorization[field] = value
    json_write(historical.root / "authorization.json", historical.authorization)
    bind_mutated_history(historical, monkeypatch)
    with pytest.raises(Exception, match="prior_state_changed"):
        proof.prior_proof()


@pytest.mark.parametrize("field,value", [("counts", {"planned": 6, "passed": 2}),
    ("tool_counts", {"planned": 6, "passed": 5}), ("reserved_requests", 15),
    ("semantic_reviews_imported", 5), ("review_ready", False), ("trial_request_limit", 138),
    ("candidate_baseline", "other-business-version"), ("cases", [])])
def test_stored_report_identity_and_original_6_of_6_are_not_recomputed(historical, monkeypatch, field, value):
    historical.report[field] = value
    json_write(proof.OLD_REPORT, historical.report)
    bind_mutated_history(historical, monkeypatch)
    with pytest.raises(Exception, match="prior_report_changed"):
        proof.prior_proof()


@pytest.mark.parametrize("field,value", [("status", "failed"), ("tool_status", "failed"),
    ("artifact_sha256", "f" * 64), ("review", {"new_score": "invented"})])
def test_stored_report_rows_must_match_original_case_and_review(historical, monkeypatch, field, value):
    historical.report["cases"][0][field] = value
    json_write(proof.OLD_REPORT, historical.report)
    bind_mutated_history(historical, monkeypatch)
    with pytest.raises(Exception, match="prior_report_binding_changed"):
        proof.prior_proof()


def test_deleted_review_and_symlink_are_not_valid_historical_evidence(historical):
    path = historical.root / "unexpected-link.json"
    path.symlink_to(historical.root / "protocol.json")
    with pytest.raises(Exception, match="old_tree_symlink"):
        proof.prior_proof()
    path.unlink()
    (historical.root / "review-D-p01.json").unlink()
    with pytest.raises(Exception, match="prior_history_changed"):
        proof.prior_proof()


def test_changed_57_or_original_16_138_ancestor_cannot_be_skipped(historical, monkeypatch):
    def changed():
        raise ValueError("older_sealed_history_changed")
    monkeypatch.setattr(proof.previous, "prior_proof", changed)
    with pytest.raises(ValueError, match="older_sealed_history_changed"):
        proof.prior_proof()
    assert historical.replays == []


def test_new_zero_authorization_directory_does_not_change_old_proof_or_allowance(historical):
    before = proof.prior_proof()
    json_write(proof.ROOT / "protocol.json", {"authorized_model_requests": 0})
    write(proof.ROOT / "ledger/journal.jsonl", b"new independent empty-ledger fixture")
    assert proof.prior_proof() == before
    assert proof._tree_files(historical.root) == historical.tree


def test_archive_change_during_read_is_rejected(historical, monkeypatch):
    original = proof._tree_files
    calls = []
    def tree(path):
        value = original(path)
        calls.append(True)
        if len(calls) == 2:
            value["case-D-p01.json"] = "a" * 64
        return value
    monkeypatch.setattr(proof, "_tree_files", tree)
    with pytest.raises(Exception, match="prior_changed_during_read"):
        proof.prior_proof()


@pytest.mark.parametrize("raw", [b"", b"backend/\0", b"backend/../private\0", b"backend/./file.py\0",
    b"backend//file.py\0", b"backend/a.py\0backend/a.py\0", b"frontend/a.py\0",
    b"backend/a.py", b"backend/name\nsecret.py\0", b"backend/name\\escape.py\0"])
def test_fixed_git_parser_rejects_ambiguous_or_outside_paths(fixture, monkeypatch, raw):
    def run(args, **kwargs):
        assert args == ["git", "ls-tree", "-r", "-z", "--name-only", proof.BASELINE, "--", "backend/"]
        assert kwargs["cwd"] == proof.bench.BACKEND_ROOT.parent
        assert kwargs.get("shell", False) is False and kwargs["timeout"] == 15
        return SimpleNamespace(stdout=raw)
    monkeypatch.setattr(proof.subprocess, "run", run)
    with pytest.raises(Exception, match="baseline_paths_invalid"):
        REAL_TRACKED()
