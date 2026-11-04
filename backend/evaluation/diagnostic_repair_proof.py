"""Small, closed source and paused-record proof for a targeted repair trial.

No ancestor protocol is made to accept new code. All existing runtime files and
tracked Python tests must equal the fixed repair commit, except exact supervisor
and explicit-diagnostic routing hooks. New files are a closed list and are hashed by the
new execution protocol. This module performs read-only local checks only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from evaluation import benchmark as bench, intent_trial as base, prospective_validation as design
from evaluation.intent_transport import canonical_payload_hash
from evaluation.prospective_budget import ProspectiveBudget
from evaluation.prospective_repaired_proof import _git_blobs, _safe_file, _tree_files


ROOT = bench.BACKEND_ROOT / "data/benchmarks/diagnostic-repair-validation-v1"
BASELINE = "e42eb72a1a4ebd098360dc8cf41dbb5b0b842522"
NEW_FILES = ("evaluation/diagnostic_repair_budget.py", "evaluation/diagnostic_repair_proof.py",
             "evaluation/diagnostic_repair_trial.py", "evaluation/benchmark/diagnostic-repair-v1.json",
             "tests/test_diagnostic_repair_budget.py", "tests/test_diagnostic_repair_proof.py",
             "tests/test_diagnostic_repair_trial.py")
HOOK_PATH = "evaluation/task_supervisor.py"
# Exact reviewed tuple addition for evaluation.diagnostic_repair_trial only.
HOOK_SHA256 = "94ab08a761f8a0b908c72916d6c15f97206f039311ab78471b9edb13e1ccf60b"
ROUTING_HOOK_PATH = "agents/agent_orchestrator.py"
ROUTING_HOOK_SHA256 = "3da4f8b27c235ebd73bb6cc5893a1996d79d19d751b2002f274e056956af76b2"
OLD_ROOT = bench.BACKEND_ROOT / "data/benchmarks/prospective-execution-v2-readonly-scope"
OLD_PROTOCOL = "3779b11d7938e9fe353a760abd214efe61d478468d3fb46614a734bb8e25b290"
OLD_JOURNAL = "823e5de4bd40071bfd643665d397167bba6f0fc69cbbbefd8b19b9af0cbf629f"
OLD_TREE = "1251874f7afa12386263cd4bad0f6bc769b5657b77f9c851b0f7742a82d8849d"
OLD_CHAIN = "89ae980e74e257b37925219c7cb5b17f1943935e7f76c536ae945885679896a6"
_RUNTIME_FOLDERS = frozenset({"agents", "core", "tooling", "memory", "monitor", "api",
                             "demo_business", "evaluation", "skills", "config"})
_SOURCE_SUFFIXES = frozenset({".py", ".json", ".md", ".txt"})


def _tracked_paths():
    """List pinned Git names, never the current index or a user-supplied ref."""
    try:
        result = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
                                cwd=bench.BACKEND_ROOT.parent, check=True, capture_output=True, timeout=15)
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError
        rows = result.stdout.decode().split("\0")
        if rows[-1] != "" or len(rows[:-1]) != len(set(rows[:-1])):
            raise ValueError
        paths = []
        for row in rows[:-1]:
            if (not row.startswith("backend/") or row == "backend/" or "\n" in row or "\r" in row
                    or ".." in Path(row).parts or Path(row).is_absolute()):
                raise ValueError
            paths.append(row.removeprefix("backend/"))
        if not paths:
            raise ValueError
        return set(paths)
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("diagnostic_repair_baseline_paths_invalid")


def _baseline_sources():
    tracked = _tracked_paths()
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    names = {name for name in tracked if (
        Path(name).parts[0] in _RUNTIME_FOLDERS and Path(name).suffix in _SOURCE_SUFFIXES
        and "__pycache__" not in Path(name).parts
        or Path(name).parts[0] == "tests" and Path(name).suffix == ".py"
        or name in fixed)}
    if (not fixed <= names or set(NEW_FILES) & tracked
            or not {HOOK_PATH, ROUTING_HOOK_PATH} <= names):
        base._fail("diagnostic_repair_baseline_closure_invalid")
    return names


def _current_sources():
    values = design._sources(bench.BACKEND_ROOT)
    tests = bench.BACKEND_ROOT / "tests"
    if tests.is_symlink() or tests.resolve() != tests or not tests.is_dir():
        base._fail("diagnostic_repair_tests_path_invalid")
    for path in sorted(tests.rglob("*")):
        if path.is_symlink():
            base._fail("diagnostic_repair_test_symlink")
        if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts:
            values[str(path.relative_to(bench.BACKEND_ROOT))] = bench.sha256(_safe_file(path))
    return values


def source_proof():
    """Current bytes equal e42 plus seven additions and two exact hooks.

    Do not call prior_proof here: the caller binds source/data/dependencies and
    the predecessor separately, avoiding two scans of the same paused archive.
    """
    baseline, current = _baseline_sources(), _current_sources()
    if set(current) != baseline | set(NEW_FILES):
        base._fail("diagnostic_repair_unapproved_source_set")
    paths = ["backend/" + name for name in sorted(baseline)]
    old = {}
    # Reuse the existing helper's <=256-path contract, never enlarge it.
    for start in range(0, len(paths), 256):
        old.update(_git_blobs(BASELINE, paths[start:start + 256]))
    baseline_hashes = {name.removeprefix("backend/"): hashlib.sha256(raw).hexdigest()
                       for name, raw in old.items()}
    hooks = {HOOK_PATH: HOOK_SHA256, ROUTING_HOOK_PATH: ROUTING_HOOK_SHA256}
    for name, old_hash in baseline_hashes.items():
        expected = hooks.get(name, old_hash)
        if current[name] != expected:
            base._fail("diagnostic_repair_current_source_changed")
    return {"schema_version": "serveflow-diagnostic-repair-source-proof-v1",
            "baseline_commit": BASELINE, "baseline_files": len(baseline),
            "baseline_source_sha256": canonical_payload_hash(baseline_hashes),
            "current_files": len(current), "current_source_sha256": canonical_payload_hash(current),
            "closed_additions": list(NEW_FILES),
            "exact_hook": {"path": HOOK_PATH, "sha256": HOOK_SHA256},
            "exact_business_routing_hook": {"path": ROUTING_HOOK_PATH, "sha256": ROUTING_HOOK_SHA256,
                "baseline_sha256": baseline_hashes[ROUTING_HOOK_PATH],
                "scope": "explicit diagnostic requests contribute Technical routing evidence; reuse the existing definition guard to exclude generic term definitions without a bound error code; role prompts, schemas and limits unchanged"},
            "business_changes_after_baseline": True, "model_authorization_granted": 0}


def artifacts():
    """Hash source closure and all seven new files, including this proof."""
    return _current_sources()


def prior_proof():
    """Read the 57-attempt paused predecessor, never resume or rescore it."""
    before = _tree_files(OLD_ROOT)
    if (len(before) != 222 or canonical_payload_hash(before) != OLD_TREE
            or before.get("protocol.json") != OLD_PROTOCOL
            or before.get("ledger/journal.jsonl") != OLD_JOURNAL):
        base._fail("diagnostic_repair_paused_history_changed")
    protocol = bench.read_json(_safe_file(OLD_ROOT / "protocol.json"))
    ledger = ProspectiveBudget(OLD_ROOT / "ledger", OLD_PROTOCOL, protocol["plan"])
    state, results = ledger.status(), ledger.results()
    complete = [row["case_id"] for row in protocol["plan"][:17]]
    reviews = {name: digest for name, digest in before.items() if name.startswith("review-")}
    if (state["attempts"] != 57 or not state["paused"] or state["pending"]
            or state["active_case"] != "D-p09" or state["completed_cases"] != complete
            or state["chain_hash"] != OLD_CHAIN or len(results["operations"]) != 57
            or set(results["cases"]) != set(complete)
            or set(reviews) != {"review-" + case + ".json" for case in complete}
            or _tree_files(OLD_ROOT) != before):
        base._fail("diagnostic_repair_paused_state_changed")
    return {"schema_version": "serveflow-diagnostic-repair-prior-proof-v1",
            "protocol_sha256": OLD_PROTOCOL, "journal_sha256": OLD_JOURNAL,
            "tree_sha256": OLD_TREE, "files": len(before), "chain_hash": OLD_CHAIN,
            "reserved_requests": 57, "completed_cases": 17, "reviews": 17,
            "review_files_sha256": canonical_payload_hash(reviews), "paused": True,
            "active_case": "D-p09", "pending_attempts": 0,
            "old_scores_recomputed": False, "old_allowance_reused": False,
            "new_authorization_granted": 0}
