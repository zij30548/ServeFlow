"""Synthetic offline ancestry fixtures; never read or modify old trial stores."""
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from evaluation import prospective_repaired_proof as proof
from evaluation.prospective_budget import ProspectiveBudget, fixed_plan


REAL_GIT_BLOBS = proof._git_blobs


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def write_json(path, value):
    write(path, json.dumps(value, sort_keys=True).encode())
    return digest(path.read_bytes())


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("repaired proof tests forbid network")
    for target in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, target, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    root = tmp_path.resolve() / "backend"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(proof.bench, "BACKEND_ROOT", root)
    for name, leaf in (("ROOT", "prospective-execution-v2-readonly-scope"),
                       ("DESIGN_ROOT", "prospective-validation-v1"),
                       ("EXECUTION_ROOT", "prospective-execution-v1"),
                       ("REPAIR_ROOT", "readonly-scope-repair-v1")):
        monkeypatch.setattr(proof, name, root / "data/benchmarks" / leaf)
    cases = [{"suite": "tasks", "case_id": f"p{i:02d}", "arm_order": ["A", "D"]} for i in range(1, 13)]
    cases += [{"suite": "retrieval", "case_id": f"q{i:02d}", "arm_order": ["A", "B"]} for i in range(1, 13)]
    monkeypatch.setattr(proof.design, "design", lambda: {"synthetic": "fixed original rubric"})
    monkeypatch.setattr(proof.design, "_plan", lambda _: deepcopy(cases))
    monkeypatch.setattr(proof.design, "validate", lambda: [{"total": 12}, {"total": 12}])
    monkeypatch.setattr(proof.design, "_dependencies", lambda: {"synthetic": "fixed"})
    important = ["agents/agent_orchestrator.py", "core/order_evidence.py",
        "evaluation/benchmark/prospective-tasks-v1.json", "evaluation/benchmark/prospective-retrieval-v1.json",
        "evaluation/benchmark_metrics.py", "knowledge/serveflow-demo-v2.json", "skills/general/SKILL.md",
        "tests/test_prospective_validation.py"]
    original_bytes = {name: ("old-" + name).encode() for name in important}
    original_bytes.update({f"evaluation/frozen_{i:02d}.py": f"frozen-{i}".encode() for i in range(99-len(important))})
    execution_bytes = {**original_bytes, **{name: ("execution-" + name).encode() for name in proof.EXECUTION_FILES}}
    repaired = {**execution_bytes}
    changed = {name: ("repair-" + name).encode() for name in proof.REPAIR_FILES}
    repaired.update({name.removeprefix("backend/"): raw for name, raw in changed.items() if name.startswith("backend/")})
    historical = {
        proof.DESIGN_COMMIT: {"backend/" + name: raw for name, raw in original_bytes.items()},
        proof.EXECUTION_COMMIT: {"backend/" + name: raw for name, raw in execution_bytes.items()},
        proof.REPAIR_COMMIT: {**{"backend/" + name: raw for name, raw in repaired.items()}, **changed},
    }
    monkeypatch.setattr(proof, "_git_blobs", lambda commit, paths: {name: historical[commit][name] for name in paths})
    monkeypatch.setattr(proof, "_changed_paths", lambda: set(proof.REPAIR_FILES))
    for name, raw in repaired.items():
        write(root/name, raw)
    hooks = {proof.RUNNER_PATH: b"exact narrow production runner hook",
             "tests/test_prospective_trial.py": b"exact narrow supervisor fixture hook"}
    hook_hashes = {name: digest(raw) for name, raw in hooks.items()}
    monkeypatch.setattr(proof, "RUNNER_SHA256", hook_hashes[proof.RUNNER_PATH])
    monkeypatch.setattr(proof, "RUNNER_TEST_SHA256", hook_hashes["tests/test_prospective_trial.py"])
    monkeypatch.setattr(proof, "HOOK_HASHES", hook_hashes)
    for name, raw in hooks.items():
        write(root/name, raw)
    for name in proof.NEW_FILES:
        write(root/name, ("new-" + name).encode())
    write(root.parent/"README.md", b"new current description is allowed")
    def sources(backend):
        return {str(path.relative_to(backend)): digest(path.read_bytes())
                for path in sorted(backend.rglob("*")) if path.is_file()
                and path.relative_to(backend).parts[0] != "data"
                and (path.relative_to(backend).parts[0] != "tests"
                     or path.name == "test_prospective_validation.py")}
    monkeypatch.setattr(proof.design, "_sources", sources)
    original = {"schema_version": "serveflow-prospective-design-freeze-v1",
        "source_commit": proof.DESIGN_COMMIT, "root_binding_sha256": proof.design._hash(str(proof.DESIGN_ROOT)),
        "sources": {name: digest(raw) for name, raw in original_bytes.items()},
        "design": proof.design.design(), "plan": cases, "suites": proof.design.validate(),
        "dependencies": proof.design._dependencies(), "live_payloads_frozen": False, "live_trial_ready": False,
        "authorized_model_requests": 0, "executed_model_requests": 0, "external_business_requests": 0}
    design_hash = write_json(proof.DESIGN_ROOT/"protocol.json", original)
    monkeypatch.setattr(proof, "DESIGN_HASH", design_hash)
    execution = {"schema_version": "serveflow-prospective-execution-protocol-v1", "git_head": proof.EXECUTION_COMMIT,
        "root_binding_sha256": proof.base._root_hash(proof.EXECUTION_ROOT),
        "design_proof": {"design_protocol_sha256": design_hash},
        "artifacts": {name: digest(raw) for name, raw in execution_bytes.items()},
        "authorized_model_requests": 0, "judge_requests": 0, "external_business_requests": 0,
        "plan": fixed_plan()}
    execution_hash = write_json(proof.EXECUTION_ROOT/"protocol.json", execution)
    monkeypatch.setattr(proof, "EXECUTION_HASH", execution_hash)
    repair = {"schema_version": "serveflow-readonly-scope-repair-source-freeze-v1",
        "baseline_commit": proof.EXECUTION_COMMIT, "source_commit": proof.REPAIR_COMMIT,
        "old_design_protocol_sha256": design_hash, "old_execution_protocol_sha256": execution_hash,
        "changed_files": {name: digest(raw) for name, raw in changed.items()},
        "live_execution_ready": False, "semantic_quality_claim": False,
        "authorized_model_requests": 0, "executed_model_requests": 0, "external_business_requests": 0}
    monkeypatch.setattr(proof, "REPAIR_HASH", write_json(proof.REPAIR_ROOT/"repair-source-freeze.json", repair))
    ProspectiveBudget(proof.EXECUTION_ROOT/"ledger", execution_hash, execution["plan"], initialize=True)
    return SimpleNamespace(root=root, historical=historical, original=original,
                           execution=execution, repair=repair, hooks=hook_hashes)


def test_closed_ancestry_matches_all_original_git_blobs_and_keeps_old_bytes(fixture):
    before = proof._tree_files(proof.EXECUTION_ROOT)
    result = proof.repaired_design_proof()
    assert result["original_design_files"] == 99 and result["original_execution_files"] == 107
    assert result["old_execution"]["authorized_requests"] == result["old_execution"]["reserved_requests"] == 0
    assert result["exact_current_hook_hashes"] == fixture.hooks
    assert result["current_authorization_granted"] == 0 and result["old_allowance_reused"] is False
    assert proof._tree_files(proof.EXECUTION_ROOT) == before
    artifact = proof.artifacts()
    assert len(artifact) == 113 and set(proof.NEW_FILES) <= set(artifact)
    assert set(proof.READONLY_TESTS) <= set(artifact)
    assert artifact[proof.RUNNER_PATH] == fixture.hooks[proof.RUNNER_PATH]


@pytest.mark.parametrize("which", ["design", "execution", "repair"])
def test_old_protocol_and_repair_freeze_are_pinned_not_rebased_to_current(fixture, which):
    path = {"design": proof.DESIGN_ROOT/"protocol.json", "execution": proof.EXECUTION_ROOT/"protocol.json",
            "repair": proof.REPAIR_ROOT/"repair-source-freeze.json"}[which]
    write(path, path.read_bytes() + b"\n")
    with pytest.raises(Exception, match="reference_changed"):
        proof.repaired_design_proof()


@pytest.mark.parametrize("name", ["agents/agent_orchestrator.py", "core/order_evidence.py",
    "evaluation/benchmark/prospective-retrieval-v1.json", "evaluation/benchmark_metrics.py",
    "knowledge/serveflow-demo-v2.json", "skills/general/SKILL.md", "evaluation/prospective_budget.py",
    "tests/test_prospective_task_runtime.py", "tests/test_readonly_scope_runtime.py",
    "evaluation/prospective_trial.py", "tests/test_prospective_trial.py"])
def test_every_non_new_file_remains_git_or_exact_hook_bound(fixture, name):
    path = fixture.root/name
    write(path, path.read_bytes() + b"unapproved-change")
    with pytest.raises(Exception, match="current_source_changed"):
        proof.repaired_design_proof()


def test_current_readme_may_advance_without_rewriting_historical_readme_proof(fixture):
    write(fixture.root.parent/"README.md", b"new zero-call preparation, no quality claim")
    assert proof.repaired_design_proof()["repair_changed_files"]["README.md"] == digest(
        fixture.historical[proof.REPAIR_COMMIT]["README.md"])


def test_original_git_blob_mismatch_fails_even_if_current_source_is_correct(fixture):
    fixture.historical[proof.DESIGN_COMMIT]["backend/core/order_evidence.py"] = b"wrong archived blob"
    with pytest.raises(Exception, match="archived_source_changed"):
        proof.repaired_design_proof()


def test_repair_diff_is_exactly_the_original_six_files(fixture, monkeypatch):
    monkeypatch.setattr(proof, "_changed_paths", lambda: set(proof.REPAIR_FILES) | {"backend/knowledge/serveflow-demo-v2.json"})
    with pytest.raises(Exception, match="repair_identity_changed"):
        proof.repaired_design_proof()


def test_unknown_helper_is_not_accepted_as_a_general_version_exception(fixture):
    write(fixture.root/"evaluation/unapproved_helper.py", b"not a declared addition")
    with pytest.raises(Exception, match="unapproved_source_set"):
        proof.repaired_design_proof()


@pytest.mark.parametrize("extra", ["authorization.json", "physical-001.json", "ledger/claims/unknown.json"])
def test_old_preparation_must_be_zero_and_have_no_execution_artifacts(fixture, extra):
    write(proof.EXECUTION_ROOT/extra, b"{}")
    with pytest.raises(Exception, match="old_execution_not_empty"):
        proof.repaired_design_proof()


def test_old_nonempty_or_uncertain_attempt_is_not_given_a_new_allowance(fixture):
    write(proof.EXECUTION_ROOT/"ledger/journal.jsonl", b'{"uncertain_attempt":1}\n')
    with pytest.raises(Exception, match="old_execution_not_empty"):
        proof.repaired_design_proof()


def test_symlinks_in_old_evidence_or_current_source_are_rejected(fixture):
    target = proof.EXECUTION_ROOT/"linked.json"
    target.symlink_to(proof.EXECUTION_ROOT/"protocol.json")
    with pytest.raises(Exception, match="old_tree_symlink"):
        proof.repaired_design_proof()


def test_prior_ledgers_excludes_only_new_fixed_root_and_never_rewrites_old(fixture):
    before = proof._tree_files(proof.EXECUTION_ROOT)
    one = proof.prior_ledgers()
    write(proof.ROOT/"ledger/journal.jsonl", b"new current counter")
    write(proof.ROOT/"authorization.json", b"not a real authorization")
    assert proof.prior_ledgers() == one
    other = fixture.root/"data/benchmarks/another-old-batch"
    write(other/"ledger/journal.jsonl", b"old record")
    two = proof.prior_ledgers()
    assert two["old_files"] > one["old_files"] and two["files_sha256"] != one["files_sha256"]
    assert proof._tree_files(proof.EXECUTION_ROOT) == before


def test_new_files_are_hashed_without_inventing_a_prior_git_version(fixture):
    before = proof.artifacts()
    path = fixture.root/proof.NEW_FILES[0]
    write(path, path.read_bytes() + b"new local code")
    assert proof.artifacts()[proof.NEW_FILES[0]] != before[proof.NEW_FILES[0]]
    assert proof.repaired_design_proof()["explicit_additions"] == list(proof.NEW_FILES)


@pytest.mark.parametrize("raw", [b"missing missing\n", b"abc tree 1\nx\n", b"abc blob 5\nx\n", b"abc blob 1\nx\ntrailing"])
def test_bounded_git_batch_parser_rejects_wrong_types_sizes_and_trailing_bytes(fixture, monkeypatch, raw):
    monkeypatch.setattr(proof.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=raw))
    with pytest.raises(Exception, match="git_blob_invalid"):
        REAL_GIT_BLOBS(proof.DESIGN_COMMIT, ["backend/a.py"])


def test_git_batch_parser_rejects_path_traversal_and_duplicate_paths(fixture):
    for paths in (["../private"], ["/absolute"], ["backend/a.py", "backend/a.py"], ["backend/a.py\nprivate"]):
        with pytest.raises(Exception, match="git_input_invalid"):
            REAL_GIT_BLOBS(proof.DESIGN_COMMIT, paths)
