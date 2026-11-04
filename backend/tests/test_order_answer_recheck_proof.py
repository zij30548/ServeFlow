"""All ancestry fixtures are synthetic temporary files; no old trial is read."""
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from evaluation import order_answer_recheck_proof as proof


REAL_TRACKED = proof._tracked_paths


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("order answer proof tests forbid network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    root = tmp_path.resolve()/"backend"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(proof.bench, "BACKEND_ROOT", root)
    monkeypatch.setattr(proof, "ROOT", root/"data/benchmarks/order-answer-recheck-v1")
    monkeypatch.setattr(proof, "OLD_ROOT", root/"data/benchmarks/diagnostic-repair-validation-v1")
    names = ["core/order_answers.py", "core/order_evidence.py", "core/diagnostic_evidence.py",
        "agents/agent_orchestrator.py", "evaluation/task_supervisor.py", "tooling/tool_manager.py",
        "knowledge/serveflow-demo-v2.json", "evaluation/benchmark/diagnostic-repair-v1.json", "requirements.txt",
        "tests/test_order_answers.py", "tests/test_order_answer_integration.py", "tests/test_benchmark.py"]
    baseline = {name: ("frozen-bb19a-"+name).encode() for name in names}
    baseline.update({f"evaluation/frozen_{i:03d}.py": f"frozen source {i}".encode()
                     for i in range(proof.BASELINE_FILES-len(names))})
    for name, raw in baseline.items():
        write(root/name, raw)
    fixed = ["knowledge/serveflow-demo-v2.json", "requirements.txt", "tests/test_benchmark.py"]
    monkeypatch.setattr(proof.bench, "artifact_paths", lambda _: list(fixed))
    monkeypatch.setattr(proof, "_tracked_paths", lambda: set(baseline))
    requests = []
    def git(commit, paths):
        requests.append((commit, list(paths)))
        assert commit == proof.BASELINE and len(paths) <= 256
        return {name: baseline[name.removeprefix("backend/")] for name in paths}
    monkeypatch.setattr(proof, "_git_blobs", git)
    hook = b"original-supervisor-plus-one-fixed-worker-string"
    write(root/proof.HOOK_PATH, hook)
    monkeypatch.setattr(proof, "HOOK_SHA256", digest(hook))
    for name in proof.NEW_FILES:
        write(root/name, ("new preparation-"+name).encode())
    def current():
        return {str(path.relative_to(root)): digest(path.read_bytes())
                for path in sorted(root.rglob("*")) if path.is_file()
                and path.relative_to(root).parts[0] != "data"}
    monkeypatch.setattr(proof.previous, "_current_sources", current)
    old57calls = []
    old57 = {"reserved_requests": 57, "completed_cases": 17, "reviews": 17,
             "paused": True, "old_allowance_reused": False}
    def previous():
        old57calls.append(True)
        return deepcopy(old57)
    monkeypatch.setattr(proof.previous, "prior_proof", previous)
    return SimpleNamespace(root=root, baseline=baseline, requests=requests,
                           old57=old57, old57calls=old57calls)


@pytest.fixture
def historical(fixture, monkeypatch):
    root = proof.OLD_ROOT
    cases = [row["case_id"] for row in proof.fixed_plan()]
    protocol = {"git_head": "e80498f9bd00ebcf9a17c6a398c1fdba4782b36d", "plan": proof.fixed_plan()}
    write(root/"protocol.json", json.dumps(protocol).encode())
    phash = digest((root/"protocol.json").read_bytes())
    write(root/"authorization.json", json.dumps({"protocol_sha256": phash, "max_requests": 138}).encode())
    write(root/"ledger/journal.jsonl", b"original sixteen accounted attempts")
    for index in range(1, 17):
        write(root/f"physical-{index:03d}.json", f"old native output {index}".encode())
    for name in cases:
        write(root/f"case-{name}.json", ("old case "+name).encode())
        write(root/f"review-{name}.json", ("old sealed assessment "+name).encode())
    for index in range(74-len(proof._tree_files(root))):
        write(root/f"frozen-{index:03d}.json", b"original claims requests and data proof")
    tree = proof._tree_files(root)
    monkeypatch.setattr(proof, "OLD_PROTOCOL", phash)
    monkeypatch.setattr(proof, "OLD_AUTHORIZATION", tree["authorization.json"])
    monkeypatch.setattr(proof, "OLD_JOURNAL", tree["ledger/journal.jsonl"])
    monkeypatch.setattr(proof, "OLD_TREE", proof.canonical_payload_hash(tree))
    monkeypatch.setattr(proof, "OLD_CHAIN", "c"*64)
    state = {"attempts": 16, "completed_cases": list(cases), "active_case": None,
             "pending": [], "paused": False, "chain_hash": "c"*64}
    sealed = {"operations": {str(i): {} for i in range(16)}, "cases": {name: {} for name in cases}}
    replays = []
    def budget(path, protocol_hash, plan):
        replays.append((path, protocol_hash, deepcopy(plan)))
        return SimpleNamespace(status=lambda: deepcopy(state), results=lambda: deepcopy(sealed))
    monkeypatch.setattr(proof, "DiagnosticRepairBudget", budget)
    return SimpleNamespace(fixture=fixture, root=root, tree=tree, state=state,
                           sealed=sealed, cases=cases, replays=replays)


def test_fixed_business_and_four_new_files_form_exact_198_source_closure(fixture, monkeypatch):
    monkeypatch.setattr(proof, "prior_proof", lambda: pytest.fail("source does not replay history"))
    value = proof.source_proof()
    assert value["baseline_files"] == 194 and value["current_files"] == 198
    assert value["baseline_commit"] == proof.BASELINE
    assert value["closed_additions"] == list(proof.NEW_FILES)
    assert value["business_changes_after_baseline"] is False and value["model_authorization_granted"] == 0
    assert len(proof.artifacts()) == 198
    assert len(fixture.requests) == 1 and fixture.requests[0][0] == proof.BASELINE


@pytest.mark.parametrize("name", ["core/order_answers.py", "core/order_evidence.py", "core/diagnostic_evidence.py",
    "agents/agent_orchestrator.py", "tooling/tool_manager.py", "knowledge/serveflow-demo-v2.json",
    "evaluation/benchmark/diagnostic-repair-v1.json", "requirements.txt", "tests/test_order_answers.py",
    "tests/test_order_answer_integration.py", "evaluation/task_supervisor.py"])
def test_no_business_data_question_or_existing_test_can_drift(fixture, name):
    path = fixture.root/name
    write(path, path.read_bytes()+b"x")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.source_proof()


def test_unknown_helpers_and_missing_declared_files_are_rejected(fixture):
    path = fixture.root/"evaluation/another_helper.py"
    write(path, b"not an authorized addition")
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.source_proof()
    path.unlink()
    (fixture.root/proof.NEW_FILES[0]).unlink()
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.source_proof()


def test_all_new_files_are_hashed_without_fake_git_ancestry(fixture):
    before = proof.artifacts()
    path = fixture.root/proof.NEW_FILES[0]
    write(path, path.read_bytes()+b"new local preparation")
    assert proof.artifacts()[proof.NEW_FILES[0]] != before[proof.NEW_FILES[0]]
    assert proof.source_proof()["current_source_sha256"] != proof.canonical_payload_hash(before)


def test_historical_57_and_16_are_bound_without_old_trial_verification_or_rescoring(historical, monkeypatch):
    monkeypatch.setattr(proof, "source_proof", lambda: pytest.fail("history does not inspect current execution"))
    result = proof.prior_proof()
    assert result["paused_57"] == historical.fixture.old57
    assert len(historical.fixture.old57calls) == 1
    old16 = result["fixed_16"]
    assert old16["reserved_requests"] == 16 and old16["authorized_requests"] == 138
    assert old16["completed_cases"] == old16["reviews"] == 6
    assert old16["recorded_strict_result"] == {"passed": 2, "planned": 6, "rescored": False}
    assert not old16["paused"] and old16["pending_attempts"] == 0
    assert result["new_authorization_granted"] == 0 and result["old_allowance_reused"] is False
    assert result["old_scores_recomputed"] is False and len(historical.replays) == 1
    assert proof._tree_files(historical.root) == historical.tree


@pytest.mark.parametrize("name", ["protocol.json", "authorization.json", "ledger/journal.jsonl",
    "review-D-p01.json", "case-D-n03.json", "physical-001.json", "frozen-000.json"])
def test_any_old_assessment_authorization_attempt_or_output_change_stops_proof(historical, name):
    path = historical.root/name
    write(path, path.read_bytes()+b"modified")
    with pytest.raises(Exception, match="prior_history_changed"):
        proof.prior_proof()
    assert historical.replays == []


@pytest.mark.parametrize("field,value", [("attempts", 0), ("completed_cases", []), ("active_case", "D-p09"),
    ("pending", ["uncertain"]), ("paused", True), ("chain_hash", "a"*64)])
def test_uncertain_reinitialized_or_changed_old_ledger_does_not_create_allowance(historical, field, value):
    historical.state[field] = value
    with pytest.raises(Exception, match="prior_state_changed"):
        proof.prior_proof()


def test_changed_old_57_proof_stops_instead_of_ignoring_predecessor(historical, monkeypatch):
    def damaged():
        raise ValueError("old_57_bytes_changed")
    monkeypatch.setattr(proof.previous, "prior_proof", damaged)
    with pytest.raises(ValueError, match="old_57_bytes_changed"):
        proof.prior_proof()
    assert historical.replays == []


def test_new_root_is_independent_but_does_not_reset_or_hide_old_records(historical):
    result = proof.prior_proof()
    write(proof.ROOT/"protocol.json", b"new zero-authorization preparation")
    write(proof.ROOT/"ledger/journal.jsonl", b"new separate ledger")
    assert proof.prior_proof() == result
    assert proof._tree_files(historical.root) == historical.tree


def test_old_symlink_or_missing_review_is_not_a_valid_archive(historical):
    path = historical.root/"link.json"
    path.symlink_to(historical.root/"protocol.json")
    with pytest.raises(Exception, match="old_tree_symlink"):
        proof.prior_proof()
    path.unlink()
    (historical.root/"review-D-p01.json").unlink()
    with pytest.raises(Exception, match="prior_history_changed"):
        proof.prior_proof()


@pytest.mark.parametrize("raw", [b"", b"backend/\0", b"backend/../private\0",
    b"backend/a.py\0backend/a.py\0", b"frontend/a.py\0", b"backend/a.py"])
def test_git_name_parser_uses_fixed_ref_and_rejects_malformed_paths(fixture, monkeypatch, raw):
    def run(args, **kwargs):
        assert args == ["git", "ls-tree", "-r", "-z", "--name-only", proof.BASELINE, "--", "backend/"]
        assert kwargs["cwd"] == proof.bench.BACKEND_ROOT.parent
        return SimpleNamespace(stdout=raw)
    monkeypatch.setattr(proof.subprocess, "run", run)
    with pytest.raises(Exception, match="baseline_paths_invalid"):
        REAL_TRACKED()
