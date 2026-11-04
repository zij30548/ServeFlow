"""Closed offline proof for the fixed six-case order-answer recheck.

The repaired business version is immutable. Only four declared preparation
files and one exact supervisor whitelist addition may differ from that commit.
Two prior trials are read-only references, never reusable budgets or rescored
answers. Importing this module creates no client, store, counter or allowance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from evaluation import benchmark as bench, intent_trial as base
from evaluation import diagnostic_repair_proof as previous
from evaluation.diagnostic_repair_budget import DiagnosticRepairBudget, fixed_plan
from evaluation.intent_transport import canonical_payload_hash
from evaluation.prospective_repaired_proof import _git_blobs, _safe_file, _tree_files


ROOT = bench.BACKEND_ROOT / "data/benchmarks/order-answer-recheck-v1"
BASELINE = "bb19a1d9a754d7beef9985c5d0ca998ce3bfa9f9"
NEW_FILES = ("evaluation/order_answer_recheck.py", "evaluation/order_answer_recheck_proof.py",
             "tests/test_order_answer_recheck.py", "tests/test_order_answer_recheck_proof.py")
HOOK_PATH = "evaluation/task_supervisor.py"
# The sole changed baseline file only adds this recheck's fixed worker module
# to the supervisor whitelist; every byte is bound, with no ignore-version flag.
HOOK_SHA256 = "82aad333dfe8bce565d85a320f513eeb25f2e4b4e9ec559f47b9df1b1266a40b"
BASELINE_FILES = 194
CURRENT_FILES = 198
OLD_ROOT = bench.BACKEND_ROOT / "data/benchmarks/diagnostic-repair-validation-v1"
OLD_PROTOCOL = "8de1bdd41d130720d678e409b23aa4081155132888b1929cb3e5bc64d8f8e1cb"
OLD_AUTHORIZATION = "9a5b0000f49221d0c8b054c299f8b195b34ebd2ab9b544950d83b0af04e2d88f"
OLD_JOURNAL = "2f9e861953881dfda99cfa6ad8969cf7a3f28b1564628b1606150304452428c7"
OLD_CHAIN = "b2ba6ce2104084179014ba8a91f6afdb91e888b1d8708b598ff5fd3a6dcd7ca2"
OLD_TREE = "1ac30b543a24b1bcfa4cc2f5d30bc83cfadcf513a0e339ae9e0a1a5c5c71ec90"


def _tracked_paths():
    try:
        result = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
                                cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15)
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError
        rows = result.stdout.decode().split("\0")
        if rows[-1] != "" or not rows[:-1] or len(set(rows[:-1])) != len(rows[:-1]):
            raise ValueError
        if any(not row.startswith("backend/") or row == "backend/" or "\n" in row or "\r" in row
               or Path(row).is_absolute() or ".." in Path(row).parts for row in rows[:-1]):
            raise ValueError
        return {row.removeprefix("backend/") for row in rows[:-1]}
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("order_answer_recheck_baseline_paths_invalid")


def _baseline_sources():
    tracked = _tracked_paths()
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    names = {name for name in tracked if (
        Path(name).parts[0] in previous._RUNTIME_FOLDERS and Path(name).suffix in previous._SOURCE_SUFFIXES
        and "__pycache__" not in Path(name).parts
        or Path(name).parts[0] == "tests" and Path(name).suffix == ".py"
        or name in fixed)}
    if (len(names) != BASELINE_FILES or not fixed <= names
            or HOOK_PATH not in names or set(NEW_FILES) & tracked):
        base._fail("order_answer_recheck_baseline_closure_invalid")
    return names


def source_proof():
    """Bind bb19a business, prompts, schemas, corpus, tests and exact hook."""
    original, current = _baseline_sources(), previous._current_sources()
    if len(current) != CURRENT_FILES or set(current) != original | set(NEW_FILES):
        base._fail("order_answer_recheck_unapproved_source_set")
    paths, blobs = ["backend/"+name for name in sorted(original)], {}
    for start in range(0, len(paths), 256):
        blobs.update(_git_blobs(BASELINE, paths[start:start+256]))
    hashes = {name.removeprefix("backend/"): hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
    if any(current[name] != (HOOK_SHA256 if name == HOOK_PATH else digest) for name, digest in hashes.items()):
        base._fail("order_answer_recheck_current_source_changed")
    return {"schema_version": "serveflow-order-answer-recheck-source-proof-v1",
        "baseline_commit": BASELINE, "baseline_files": len(original), "current_files": len(current),
        "baseline_source_sha256": canonical_payload_hash(hashes),
        "current_source_sha256": canonical_payload_hash(current), "closed_additions": list(NEW_FILES),
        "exact_supervisor_hook": {"path": HOOK_PATH, "sha256": HOOK_SHA256},
        "business_changes_after_baseline": False, "model_authorization_granted": 0}


def artifacts():
    """The caller's new protocol binds all current files, including itself."""
    return previous._current_sources()


def prior_proof():
    """Preserve both old trials and their original results without rescoring."""
    paused = previous.prior_proof()
    before = _tree_files(OLD_ROOT)
    if (len(before) != 74 or canonical_payload_hash(before) != OLD_TREE
            or before.get("protocol.json") != OLD_PROTOCOL
            or before.get("authorization.json") != OLD_AUTHORIZATION
            or before.get("ledger/journal.jsonl") != OLD_JOURNAL):
        base._fail("order_answer_recheck_prior_history_changed")
    protocol = bench.read_json(_safe_file(OLD_ROOT/"protocol.json"))
    authorization = bench.read_json(_safe_file(OLD_ROOT/"authorization.json"))
    ledger = DiagnosticRepairBudget(OLD_ROOT/"ledger", OLD_PROTOCOL, protocol["plan"])
    state, sealed = ledger.status(), ledger.results()
    cases = [row["case_id"] for row in fixed_plan()]
    reviews = {name: digest for name, digest in before.items() if name.startswith("review-")}
    if (protocol["git_head"] != "e80498f9bd00ebcf9a17c6a398c1fdba4782b36d"
            or authorization.get("protocol_sha256") != OLD_PROTOCOL
            or type(authorization.get("max_requests")) is not int or authorization["max_requests"] != 138
            or state["attempts"] != 16 or state["paused"] or state["pending"] or state["active_case"] is not None
            or state["completed_cases"] != cases or state["chain_hash"] != OLD_CHAIN
            or len(sealed["operations"]) != 16 or set(sealed["cases"]) != set(cases)
            or set(reviews) != {"review-"+case+".json" for case in cases}
            or _tree_files(OLD_ROOT) != before):
        base._fail("order_answer_recheck_prior_state_changed")
    return {"schema_version": "serveflow-order-answer-recheck-prior-proof-v1",
        "paused_57": paused,
        "fixed_16": {"protocol_sha256": OLD_PROTOCOL, "authorization_sha256": OLD_AUTHORIZATION,
            "journal_sha256": OLD_JOURNAL, "tree_sha256": OLD_TREE, "files": 74, "chain_hash": OLD_CHAIN,
            "reserved_requests": 16, "authorized_requests": 138, "completed_cases": 6, "reviews": 6,
            "review_files_sha256": canonical_payload_hash(reviews),
            "recorded_strict_result": {"passed": 2, "planned": 6, "rescored": False},
            "paused": False, "pending_attempts": 0},
        "old_scores_recomputed": False, "old_allowance_reused": False, "new_authorization_granted": 0}
