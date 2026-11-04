"""Local synthetic proof fixtures only; old evidence and services stay untouched."""
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from evaluation import diagnostic_repair_proof as proof


REAL_TRACKED = proof._tracked_paths


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("diagnostic repair proof tests forbid network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    root = tmp_path.resolve()/"backend"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(proof.bench, "BACKEND_ROOT", root)
    monkeypatch.setattr(proof, "ROOT", root/"data/benchmarks/diagnostic-repair-validation-v1")
    monkeypatch.setattr(proof, "OLD_ROOT", root/"data/benchmarks/prospective-execution-v2-readonly-scope")
    names = ["core/diagnostic_evidence.py", "core/order_evidence.py", "agents/agent_orchestrator.py",
        "evaluation/task_supervisor.py", "evaluation/benchmark_metrics.py", "tooling/tool_manager.py",
        "knowledge/serveflow-demo-v2.json", "skills/general/SKILL.md", "requirements.txt",
        "tests/test_diagnostic_request_binding.py", "tests/test_order_cancel_scope.py",
        "tests/test_benchmark.py"]
    baseline = {name: ("unchanged-e42-"+name).encode() for name in names}
    calls = []
    def blobs(commit, paths):
        calls.append((commit, list(paths)))
        assert commit == proof.BASELINE and len(paths) <= 256
        return {path: baseline[path.removeprefix("backend/")] for path in paths}
    monkeypatch.setattr(proof, "_git_blobs", blobs)
    monkeypatch.setattr(proof, "_tracked_paths", lambda: set(baseline))
    fixed = ["knowledge/serveflow-demo-v2.json", "requirements.txt", "tests/test_benchmark.py"]
    monkeypatch.setattr(proof.bench, "artifact_paths", lambda _: list(fixed))
    for name, raw in baseline.items():
        write(root/name, raw)
    hook = b"unchanged-e42-supervisor-plus-one-reviewed-worker-string"
    write(root/proof.HOOK_PATH, hook)
    monkeypatch.setattr(proof, "HOOK_SHA256", digest(hook))
    routing_hook = b"unchanged-e42-orchestrator-plus-four-line-diagnostic-routing-evidence"
    write(root/proof.ROUTING_HOOK_PATH, routing_hook)
    monkeypatch.setattr(proof, "ROUTING_HOOK_SHA256", digest(routing_hook))
    for name in proof.NEW_FILES:
        write(root/name, ("declared-new-"+name).encode())
    def sources(directory):
        values = {str(path.relative_to(directory)): digest(path.read_bytes())
            for path in sorted(directory.rglob("*")) if path.is_file()
            and path.relative_to(directory).parts[0] in proof._RUNTIME_FOLDERS
            and path.suffix in proof._SOURCE_SUFFIXES}
        values.update({name: digest((directory/name).read_bytes()) for name in fixed})
        return values
    monkeypatch.setattr(proof.design, "_sources", sources)
    return SimpleNamespace(root=root, baseline=baseline, calls=calls, fixed=fixed)


@pytest.fixture
def history_fixture(source_fixture, monkeypatch):
    root = proof.OLD_ROOT
    completed = [f"job-{number:02d}" for number in range(17)]
    plan = [{"case_id": key} for key in [*completed, "D-p09"]]
    write(root/"protocol.json", json.dumps({"git_head": "old-frozen-execution", "plan": plan}).encode())
    write(root/"ledger/journal.jsonl", b"old paused journal, never an execution input")
    for number in range(1, 58):
        write(root/f"physical-{number:03d}.json", f"physical-{number}".encode())
    for key in completed:
        write(root/f"case-{key}.json", key.encode())
        write(root/f"review-{key}.json", ("original score-"+key).encode())
    for number in range(222-len(proof._tree_files(root))):
        write(root/f"frozen-{number:03d}.json", b"untouched old proof and claims")
    tree = proof._tree_files(root)
    monkeypatch.setattr(proof, "OLD_TREE", proof.canonical_payload_hash(tree))
    monkeypatch.setattr(proof, "OLD_PROTOCOL", tree["protocol.json"])
    monkeypatch.setattr(proof, "OLD_JOURNAL", tree["ledger/journal.jsonl"])
    monkeypatch.setattr(proof, "OLD_CHAIN", "a"*64)
    state = {"attempts": 57, "paused": True, "pending": [], "active_case": "D-p09",
             "completed_cases": list(completed), "chain_hash": "a"*64}
    results = {"operations": {str(i): {} for i in range(57)}, "cases": {key: {} for key in completed}}
    replays = []
    def ledger(path, protocol_hash, old_plan):
        replays.append((path, protocol_hash, deepcopy(old_plan)))
        return SimpleNamespace(status=lambda: deepcopy(state), results=lambda: deepcopy(results))
    monkeypatch.setattr(proof, "ProspectiveBudget", ledger)
    return SimpleNamespace(source=source_fixture, root=root, tree=tree, state=state,
                           results=results, completed=completed, replays=replays)


def test_source_closure_binds_old_bytes_and_all_seven_new_files(source_fixture, monkeypatch):
    monkeypatch.setattr(proof, "prior_proof", lambda: pytest.fail("source proof must not scan predecessor"))
    value = proof.source_proof()
    assert value["baseline_commit"] == proof.BASELINE
    assert value["baseline_files"] == len(source_fixture.baseline)
    assert value["current_files"] == len(source_fixture.baseline)+7
    assert value["closed_additions"] == list(proof.NEW_FILES)
    assert value["model_authorization_granted"] == 0
    assert value["business_changes_after_baseline"] is True
    assert value["exact_business_routing_hook"]["path"] == proof.ROUTING_HOOK_PATH
    assert value["exact_business_routing_hook"]["sha256"] == proof.ROUTING_HOOK_SHA256
    assert value["exact_business_routing_hook"]["baseline_sha256"] == digest(
        source_fixture.baseline[proof.ROUTING_HOOK_PATH])
    assert source_fixture.calls[0][0] == proof.BASELINE
    assert set(proof.artifacts()) == set(source_fixture.baseline) | set(proof.NEW_FILES)


@pytest.mark.parametrize("name", ["agents/agent_orchestrator.py", "core/diagnostic_evidence.py",
    "core/order_evidence.py", "tooling/tool_manager.py", "evaluation/benchmark_metrics.py",
    "knowledge/serveflow-demo-v2.json", "skills/general/SKILL.md", "requirements.txt",
    "tests/test_order_cancel_scope.py", "evaluation/task_supervisor.py"])
def test_business_dependencies_scoring_and_old_tests_must_not_drift(source_fixture, name):
    path = source_fixture.root/name
    write(path, path.read_bytes()+b"changed after repair baseline")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.source_proof()


@pytest.mark.parametrize("name", ["evaluation/extra_helper.py", "core/unapproved_rule.py", "tests/extra_test.py"])
def test_unapproved_helpers_or_tests_are_rejected(source_fixture, name):
    write(source_fixture.root/name, b"not in closed additions")
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.source_proof()


def test_missing_new_file_does_not_get_silently_ignored(source_fixture):
    (source_fixture.root/proof.NEW_FILES[0]).unlink()
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.source_proof()


def test_new_files_are_manifest_bound_not_assigned_fake_historical_git_blobs(source_fixture):
    before = proof.artifacts()
    path = source_fixture.root/proof.NEW_FILES[1]
    write(path, path.read_bytes()+b"new local implementation")
    assert proof.artifacts()[proof.NEW_FILES[1]] != before[proof.NEW_FILES[1]]
    assert proof.source_proof()["closed_additions"] == list(proof.NEW_FILES)


@pytest.mark.parametrize("path_name", ["HOOK_PATH", "ROUTING_HOOK_PATH"])
def test_both_reviewed_whole_file_hooks_reject_even_one_other_byte(source_fixture, path_name):
    proof.source_proof()
    path = source_fixture.root/getattr(proof, path_name)
    write(path, path.read_bytes()+b"x")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.source_proof()


def test_batch_git_helper_is_reused_in_chunks_not_given_a_larger_limit(source_fixture):
    for index in range(260):
        name = f"core/extra_e42_{index:03d}.py"
        source_fixture.baseline[name] = b"tracked original source"
        write(source_fixture.root/name, source_fixture.baseline[name])
    proof.source_proof()
    assert len(source_fixture.calls) == 2
    assert all(len(paths) <= 256 for _, paths in source_fixture.calls)


def test_test_symlink_cannot_escape_the_closed_source_set(source_fixture):
    target = source_fixture.root/"tests/test_link.py"
    target.symlink_to(source_fixture.root/"tests/test_benchmark.py")
    with pytest.raises(Exception, match="test_symlink"):
        proof.source_proof()


def test_prior_binds_222_files_57_attempts_and_17_original_reviews(history_fixture, monkeypatch):
    monkeypatch.setattr(proof, "source_proof", lambda: pytest.fail("prior proof must not scan current sources"))
    result = proof.prior_proof()
    assert result["files"] == 222 and result["reserved_requests"] == 57
    assert result["completed_cases"] == result["reviews"] == 17
    assert result["paused"] and result["active_case"] == "D-p09" and result["pending_attempts"] == 0
    assert result["old_scores_recomputed"] is result["old_allowance_reused"] is False
    assert result["new_authorization_granted"] == 0
    assert len(history_fixture.replays) == 1
    assert proof._tree_files(history_fixture.root) == history_fixture.tree


@pytest.mark.parametrize("filename", ["protocol.json", "ledger/journal.jsonl", "review-job-00.json",
                                     "physical-001.json", "case-job-00.json", "frozen-000.json"])
def test_any_historical_byte_change_rejects_without_rescoring_or_replay(history_fixture, filename):
    path = history_fixture.root/filename
    write(path, path.read_bytes()+b"not the original")
    with pytest.raises(Exception, match="paused_history_changed"):
        proof.prior_proof()
    assert history_fixture.replays == []


@pytest.mark.parametrize("kind", ["add", "remove"])
def test_old_tree_cannot_omit_or_add_records(history_fixture, kind):
    if kind == "add":
        write(history_fixture.root/"unexpected.json", b"new result is forbidden")
    else:
        (history_fixture.root/"review-job-00.json").unlink()
    with pytest.raises(Exception, match="paused_history_changed"):
        proof.prior_proof()


@pytest.mark.parametrize("field,value", [("attempts", 0), ("paused", False), ("pending", ["unknown"]),
    ("active_case", None), ("completed_cases", []), ("chain_hash", "f"*64)])
def test_unknown_or_reinitialized_old_state_cannot_become_new_permission(history_fixture, field, value):
    history_fixture.state[field] = value
    with pytest.raises(Exception, match="paused_state_changed"):
        proof.prior_proof()


def test_symlink_in_paused_tree_is_rejected(history_fixture):
    (history_fixture.root/"link.json").symlink_to(history_fixture.root/"protocol.json")
    with pytest.raises(Exception, match="old_tree_symlink"):
        proof.prior_proof()


def test_new_batch_is_not_part_of_the_paused_predecessor_or_a_budget_reset(history_fixture):
    before = proof.prior_proof()
    write(proof.ROOT/"protocol.json", b"new zero-authorization preparation")
    write(proof.ROOT/"ledger/journal.jsonl", b"new independent counter")
    assert proof.prior_proof() == before
    assert proof._tree_files(history_fixture.root) == history_fixture.tree


@pytest.mark.parametrize("raw", [b"", b"backend/core/a.py\0backend/core/a.py\0",
    b"backend/../private\0", b"/absolute/private\0", b"frontend/a.py\0", b"backend/a.py", b"backend/\0"])
def test_fixed_git_name_parser_rejects_duplicates_traversal_and_wrong_prefix(source_fixture, monkeypatch, raw):
    def run(args, **kwargs):
        assert args == ["git", "ls-tree", "-r", "-z", "--name-only", proof.BASELINE, "--", "backend/"]
        assert kwargs["cwd"] == proof.bench.BACKEND_ROOT.parent
        return SimpleNamespace(stdout=raw)
    monkeypatch.setattr(proof.subprocess, "run", run)
    with pytest.raises(Exception, match="baseline_paths_invalid"):
        REAL_TRACKED()
