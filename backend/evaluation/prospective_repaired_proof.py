"""Closed offline ancestry proof for the read-only-scope repaired evaluation.

No old protocol is rewritten or made to verify changed production code. Original
design/execution bytes are checked against their Git commits; one separately
recorded repair and one exact runner-hook hash define the only allowed changes.
Preparing this proof grants no model allowance and never opens business stores.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

from evaluation import benchmark as bench, intent_trial as base, prospective_validation as design
from evaluation.intent_transport import canonical_payload_hash
from evaluation.prospective_budget import ProspectiveBudget


ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-execution-v2-readonly-scope"
DESIGN_ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-validation-v1"
EXECUTION_ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-execution-v1"
REPAIR_ROOT = bench.BACKEND_ROOT / "data/benchmarks/readonly-scope-repair-v1"
DESIGN_COMMIT = "bcd7b8aa0d915487d5595037b82104add395c271"
EXECUTION_COMMIT = "4acc516394d290f10101fd6994320304d941bf1f"
REPAIR_COMMIT = "63804c94a9f9eae7c92eaca26e8ae70e391b913f"
DESIGN_HASH = "64eb04c27fa6585b370a4c4ced0468256d0d5b86e937ddf4bc7b7afcd16a7556"
EXECUTION_HASH = "e631ee4aa54080a7f90f2620e7e73417193417dd601f40e269d7bad25e1b5307"
REPAIR_HASH = "831475dba3d032273421b6973f1e96f3f81b7b0c6be602dd87a90a5b0b073bcd"
RUNNER_PATH = "evaluation/prospective_trial.py"
# Exact narrow class-level proof/schema/root hooks and closed supervisor map.
# No command-line parameter can replace this hash or ignore source changes.
RUNNER_SHA256 = "e547e943718d398db39b96d43dc1b58213e4dc01a08581c0f6cd3617517f10c3"
RUNNER_TEST_SHA256 = "dda96f92f70de36c2ab1fede20190221d6d0351110371f640e85908959e62683"
HOOK_HASHES = {RUNNER_PATH: RUNNER_SHA256, "tests/test_prospective_trial.py": RUNNER_TEST_SHA256}
EXECUTION_FILES = tuple(f"{folder}/{name}.py" for folder, names in (
    ("evaluation", ("prospective_budget", "prospective_task_runtime", "prospective_retrieval_runtime", "prospective_trial")),
    ("tests", ("test_prospective_budget", "test_prospective_task_runtime", "test_prospective_retrieval_runtime", "test_prospective_trial")))
    for name in names)
NEW_FILES = ("evaluation/prospective_repaired_proof.py", "evaluation/prospective_repaired_trial.py",
             "tests/test_prospective_repaired_proof.py", "tests/test_prospective_repaired_trial.py")
REPAIR_FILES = ("README.md", "backend/agents/agent_orchestrator.py", "backend/core/order_evidence.py",
                "backend/tests/test_prospective_task_runtime.py", "backend/tests/test_readonly_scope_language.py",
                "backend/tests/test_readonly_scope_runtime.py")
READONLY_TESTS = ("tests/test_readonly_scope_language.py", "tests/test_readonly_scope_runtime.py")


def _safe_file(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        base._fail("prospective_repaired_unsafe_file")
    return path


def _pinned(path, digest):
    if bench.sha256(_safe_file(path)) != digest:
        base._fail("prospective_repaired_reference_changed")
    return bench.read_json(path)


def _git_blobs(commit, paths):
    """One bounded, shell-free batch; repository-relative paths only."""
    if (type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or len(paths) > 256 or len(set(paths)) != len(paths)
            or any(type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
                   or name.startswith("/") or ".." in Path(name).parts for name in paths)):
        base._fail("prospective_repaired_git_input_invalid")
    try:
        result = subprocess.run(["git", "cat-file", "--batch"],
            input="".join(f"{commit}:{name}\n" for name in paths).encode(),
            cwd=bench.BACKEND_ROOT, check=True, capture_output=True, timeout=15)
        raw, position, blobs = result.stdout, 0, {}
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError
        for name in paths:
            end = raw.find(b"\n", position)
            header = raw[position:end].split()
            if end < 0 or len(header) != 3 or header[1] != b"blob":
                raise ValueError
            size = int(header[2]); position = end + 1
            if not 0 <= size <= 2 * 1024 * 1024:
                raise ValueError
            blob = raw[position:position + size]; position += size
            if len(blob) != size or raw[position:position + 1] != b"\n":
                raise ValueError
            blobs[name] = blob; position += 1
        if position != len(raw):
            raise ValueError
        return blobs
    except (OSError, ValueError, subprocess.SubprocessError):
        base._fail("prospective_repaired_git_blob_invalid")


def _changed_paths():
    try:
        result = subprocess.run(["git", "diff", "--name-only", "--no-renames", "-z",
                                 EXECUTION_COMMIT, REPAIR_COMMIT], cwd=bench.BACKEND_ROOT,
                                check=True, capture_output=True, timeout=15)
        if len(result.stdout) > 65536:
            raise ValueError
        names = result.stdout.decode().split("\0")
        if names[-1] != "" or len(names[:-1]) != len(set(names[:-1])):
            raise ValueError
        return set(names[:-1])
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("prospective_repaired_git_diff_invalid")


def _assert_git_hashes(commit, hashes, *, backend_relative):
    paths = ["backend/" + name if backend_relative else name for name in hashes]
    blobs = _git_blobs(commit, paths)
    for name, path in zip(hashes, paths):
        if hashlib.sha256(blobs[path]).hexdigest() != hashes[name]:
            base._fail("prospective_repaired_archived_source_changed")


def _tree_files(root):
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        base._fail("prospective_repaired_old_tree_invalid")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            base._fail("prospective_repaired_old_tree_symlink")
        if path.is_file():
            result[str(path.relative_to(root))] = bench.sha256(_safe_file(path))
        elif not path.is_dir():
            base._fail("prospective_repaired_old_tree_invalid")
    return result


def _old_execution_zero(protocol):
    before = _tree_files(EXECUTION_ROOT)
    if (set(before) != {"protocol.json", "ledger/identity.json", "ledger/run.lock", "ledger/journal.jsonl"}
            or (EXECUTION_ROOT / "ledger/journal.jsonl").read_bytes() != b""
            or any((EXECUTION_ROOT / "ledger/claims").iterdir())):
        base._fail("prospective_repaired_old_execution_not_empty")
    ledger = ProspectiveBudget(EXECUTION_ROOT / "ledger", EXECUTION_HASH, protocol["plan"])
    state, results = ledger.status(), ledger.results()
    if (state["attempts"] != 0 or state["active_case"] is not None or state["pending"]
            or state["completed_cases"] or state["paused"]
            or results != {"operations": {}, "cases": {}} or _tree_files(EXECUTION_ROOT) != before):
        base._fail("prospective_repaired_old_execution_not_empty")
    return {"authorized_requests": 0, "reserved_requests": 0, "sealed_operations": 0,
            "completed_cases": 0, "authorization_present": False,
            "tree_sha256": canonical_payload_hash(before), "chain_hash": state["chain_hash"]}


def repaired_design_proof():
    """Verify exact historical references plus a closed repaired-current set."""
    original = _pinned(DESIGN_ROOT / "protocol.json", DESIGN_HASH)
    execution = _pinned(EXECUTION_ROOT / "protocol.json", EXECUTION_HASH)
    repair = _pinned(REPAIR_ROOT / "repair-source-freeze.json", REPAIR_HASH)
    if (original["schema_version"] != "serveflow-prospective-design-freeze-v1"
            or original["source_commit"] != DESIGN_COMMIT or len(original["sources"]) != 99
            or original["root_binding_sha256"] != design._hash(str(DESIGN_ROOT))
            or original["design"] != design.design() or original["plan"] != design._plan(bench.BACKEND_ROOT)
            or original["suites"] != design.validate() or original["dependencies"] != design._dependencies()
            or original["live_payloads_frozen"] is not False or original["live_trial_ready"] is not False
            or any(type(original[key]) is not int or original[key] != 0
                   for key in ("authorized_model_requests", "executed_model_requests", "external_business_requests"))):
        base._fail("prospective_repaired_original_design_identity_changed")
    if (execution["schema_version"] != "serveflow-prospective-execution-protocol-v1"
            or execution["git_head"] != EXECUTION_COMMIT
            or execution["root_binding_sha256"] != base._root_hash(EXECUTION_ROOT)
            or execution["design_proof"]["design_protocol_sha256"] != DESIGN_HASH
            or set(execution["artifacts"]) != set(original["sources"]) | set(EXECUTION_FILES)
            or any(execution["artifacts"][name] != digest for name, digest in original["sources"].items())
            or any(type(execution[key]) is not int or execution[key] != 0
                   for key in ("authorized_model_requests", "judge_requests", "external_business_requests"))):
        base._fail("prospective_repaired_original_execution_identity_changed")
    if (repair["schema_version"] != "serveflow-readonly-scope-repair-source-freeze-v1"
            or repair["baseline_commit"] != EXECUTION_COMMIT or repair["source_commit"] != REPAIR_COMMIT
            or repair["old_design_protocol_sha256"] != DESIGN_HASH
            or repair["old_execution_protocol_sha256"] != EXECUTION_HASH
            or set(repair["changed_files"]) != set(REPAIR_FILES) or _changed_paths() != set(REPAIR_FILES)
            or repair["live_execution_ready"] is not False or repair["semantic_quality_claim"] is not False
            or any(type(repair[key]) is not int or repair[key] != 0
                   for key in ("authorized_model_requests", "executed_model_requests", "external_business_requests"))):
        base._fail("prospective_repaired_repair_identity_changed")
    _assert_git_hashes(DESIGN_COMMIT, original["sources"], backend_relative=True)
    _assert_git_hashes(EXECUTION_COMMIT, execution["artifacts"], backend_relative=True)
    # README is checked at its historical repair commit, not forced back to an
    # old current-tree body when the new preparation updates current navigation.
    _assert_git_hashes(REPAIR_COMMIT, repair["changed_files"], backend_relative=False)
    current = design._sources(bench.BACKEND_ROOT)
    expected = (set(original["sources"]) | {p for p in EXECUTION_FILES if p.startswith("evaluation/")}
                | {p for p in NEW_FILES if p.startswith("evaluation/")})
    if set(current) != expected:
        base._fail("prospective_repaired_unapproved_source_set")
    previous = _git_blobs(REPAIR_COMMIT, ["backend/" + name for name in execution["artifacts"]]
                          + ["backend/" + name for name in READONLY_TESTS])
    for name in set(execution["artifacts"]) | set(READONLY_TESTS):
        digest = bench.sha256(_safe_file(bench.BACKEND_ROOT / name))
        expected_hash = HOOK_HASHES.get(name, hashlib.sha256(previous["backend/" + name]).hexdigest())
        if digest != expected_hash:
            base._fail("prospective_repaired_current_source_changed")
    for name in NEW_FILES:
        _safe_file(bench.BACKEND_ROOT / name)
    return {"schema_version": "serveflow-prospective-readonly-repair-proof-v1",
            "design_protocol_sha256": DESIGN_HASH, "design_commit": DESIGN_COMMIT,
            "execution_protocol_sha256": EXECUTION_HASH, "execution_commit": EXECUTION_COMMIT,
            "repair_freeze_sha256": REPAIR_HASH, "repair_commit": REPAIR_COMMIT,
            "original_design_files": len(original["sources"]),
            "original_execution_files": len(execution["artifacts"]),
            "original_design_sources_sha256": canonical_payload_hash(original["sources"]),
            "original_execution_sources_sha256": canonical_payload_hash(execution["artifacts"]),
            "repair_changed_files": repair["changed_files"], "old_execution": _old_execution_zero(execution),
            "current_runner_hook_sha256": RUNNER_SHA256,
            "exact_current_hook_hashes": dict(HOOK_HASHES),
            "current_source_set_sha256": canonical_payload_hash(current),
            "explicit_additions": list(NEW_FILES), "old_scores_recomputed": False,
            "old_allowance_reused": False, "current_authorization_granted": 0}


def artifacts():
    """Hash the full new set, including all old/new tests and the proof itself."""
    names = set(design._sources(bench.BACKEND_ROOT)) | set(EXECUTION_FILES) | set(READONLY_TESTS) | set(NEW_FILES)
    return {name: bench.sha256(_safe_file(bench.BACKEND_ROOT / name)) for name in sorted(names)}


def prior_ledgers():
    """Bind every old ledger tree; only this new fixed root is excluded."""
    output = {}
    for journal in sorted((bench.BACKEND_ROOT / "data/benchmarks").glob("*/ledger/journal.jsonl")):
        folder = journal.parent.parent
        if folder == ROOT:
            continue
        for name, digest in _tree_files(folder).items():
            output[str((folder / name).relative_to(bench.BACKEND_ROOT))] = digest
    return {"old_files": len(output), "files_sha256": canonical_payload_hash(output),
            "reuse_allowance": False, "rescore_old_outputs": False}
