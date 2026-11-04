"""Closed, read-only ancestry for the current 12-question A/D comparison.

The common business version, questions, prompts, sources and tests stay equal
to the pinned commit. Only six named evaluation files and one exact fixed-worker
supervisor hook may be added. Old trials remain byte-for-byte historical records;
their stored scores are bound, not recalculated. Importing this module opens no
client, database, counter or authorization.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

from evaluation import benchmark as bench, intent_trial as base
from evaluation import diagnostic_repair_proof as ancestry
from evaluation import order_answer_recheck_proof as previous
from evaluation.diagnostic_repair_budget import DiagnosticRepairBudget, fixed_plan
from evaluation.intent_transport import canonical_payload_hash
from evaluation.prospective_repaired_proof import _git_blobs, _safe_file, _tree_files


ROOT = bench.BACKEND_ROOT / "data/benchmarks/current-task-ablation-v1"
BASELINE = "7abdb6216a3a7eb0742b4b406ec58e48aa30c4a7"
NEW_FILES = (
    "evaluation/current_task_ablation.py", "evaluation/current_task_ablation_budget.py",
    "evaluation/current_task_ablation_proof.py", "tests/test_current_task_ablation.py",
    "tests/test_current_task_ablation_budget.py", "tests/test_current_task_ablation_proof.py",
)
HOOK_PATH = "evaluation/task_supervisor.py"
# Exact reviewed fixed-module/root/3600-second addition; old worker rules stay.
HOOK_SHA256 = "d46d3ef48afe3af319a6aca8cfa7469a5ca96ccc936a23161c6b93d0466b4d29"
BASELINE_FILES = 198
CURRENT_FILES = 204
OLD_ROOT = bench.BACKEND_ROOT / "data/benchmarks/order-answer-recheck-v1"
OLD_REPORT = bench.BACKEND_ROOT / "data/benchmarks/order-answer-recheck-preflight-v1/final-reviewed-report.json"
OLD_COMMIT = "ac7857635a40c415201d3709f54fdfcd9ef01198"
OLD_PROTOCOL = "da6c6c16c4ce31705ad479ea75e98ef2216374b8979c001e57d305bc280090bc"
OLD_AUTHORIZATION = "957d596582ad6afcd5131ae410cad85e4aa7d7603e6e3978aea829b741239cbd"
OLD_JOURNAL = "b1829d7a177abb56c79717f9b20abcc2b805af0fbddbdfb226d175324771fe68"
OLD_CHAIN = "15c0a90a840e8df89f3a898fc2ffc9dd0348ebb6a1db92398c286b5e95b46672"
OLD_TREE = "8c86607a7a45d0f32dd5eeaed9a36feba19eb4d0c4f645be42fc955f2996356f"
OLD_REPORT_SHA256 = "5b8ab099174ba78e3560b0963999f65ada0230c7d85718aa63cf806df87ca4a5"
_STORED_COUNTS = {"planned": 6, "passed": 6, "failed": 0,
                  "pending_review": 0, "unknown": 0, "not_executed": 0}


def _tracked_paths():
    """Read only the fixed baseline's paths; no CLI ref or current index."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--name-only", BASELINE, "--", "backend/"],
            cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=15,
        )
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError
        rows = result.stdout.decode().split("\0")
        if rows[-1] != "" or not rows[:-1] or len(set(rows[:-1])) != len(rows[:-1]):
            raise ValueError
        for name in rows[:-1]:
            if (not name.startswith("backend/") or name == "backend/"
                    or re.fullmatch(r"[A-Za-z0-9_./-]+", name) is None
                    or any(part in {"", ".", ".."} for part in name.split("/"))):
                raise ValueError
        return {name.removeprefix("backend/") for name in rows[:-1]}
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("current_task_ablation_baseline_paths_invalid")


def _baseline_sources():
    tracked = _tracked_paths()
    fixed = set(bench.artifact_paths(bench.BACKEND_ROOT))
    names = {name for name in tracked if (
        Path(name).parts[0] in ancestry._RUNTIME_FOLDERS
        and Path(name).suffix in ancestry._SOURCE_SUFFIXES and "__pycache__" not in Path(name).parts
        or Path(name).parts[0] == "tests" and Path(name).suffix == ".py"
        or name in fixed)}
    if (len(names) != BASELINE_FILES or not fixed <= names
            or HOOK_PATH not in names or set(NEW_FILES) & tracked):
        base._fail("current_task_ablation_baseline_closure_invalid")
    return names


def source_proof():
    """Only the exact worker hook differs among the 198 existing sources."""
    original, current = _baseline_sources(), ancestry._current_sources()
    if len(current) != CURRENT_FILES or set(current) != original | set(NEW_FILES):
        base._fail("current_task_ablation_unapproved_source_set")
    paths, blobs = ["backend/" + name for name in sorted(original)], {}
    for start in range(0, len(paths), 256):
        blobs.update(_git_blobs(BASELINE, paths[start:start + 256]))
    hashes = {name.removeprefix("backend/"): hashlib.sha256(raw).hexdigest()
              for name, raw in blobs.items()}
    if (re.fullmatch(r"[a-f0-9]{64}", HOOK_SHA256) is None
            or any(current[name] != (HOOK_SHA256 if name == HOOK_PATH else value)
                   for name, value in hashes.items())):
        base._fail("current_task_ablation_current_source_changed")
    return {
        "schema_version": "serveflow-current-task-ablation-source-proof-v1",
        "baseline_commit": BASELINE, "baseline_files": len(original), "current_files": len(current),
        "baseline_source_sha256": canonical_payload_hash(hashes),
        "current_source_sha256": canonical_payload_hash(current), "closed_additions": list(NEW_FILES),
        "exact_supervisor_hook": {"path": HOOK_PATH, "sha256": HOOK_SHA256},
        "business_changes_after_baseline": False, "model_authorization_granted": 0,
    }


def artifacts():
    """Bind every current source, including all six newly prepared files."""
    return ancestry._current_sources()


def prior_proof():
    """Bind old 57, old 16/138 and latest 16/24; never rerun old grading."""
    older = previous.prior_proof()
    before = _tree_files(OLD_ROOT)
    if (len(before) != 74 or canonical_payload_hash(before) != OLD_TREE
            or before.get("protocol.json") != OLD_PROTOCOL
            or before.get("authorization.json") != OLD_AUTHORIZATION
            or before.get("ledger/journal.jsonl") != OLD_JOURNAL
            or bench.sha256(_safe_file(OLD_REPORT)) != OLD_REPORT_SHA256):
        base._fail("current_task_ablation_prior_history_changed")
    protocol = bench.read_json(_safe_file(OLD_ROOT / "protocol.json"))
    authorization = bench.read_json(_safe_file(OLD_ROOT / "authorization.json"))
    report = bench.read_json(_safe_file(OLD_REPORT))
    # The stored ledger has the original 138-send identity. Its separately
    # sealed authorization limited this later execution to 24, of which 16 sent.
    ledger = DiagnosticRepairBudget(OLD_ROOT / "ledger", OLD_PROTOCOL, protocol["plan"])
    state, sealed = ledger.status(), ledger.results()
    cases = [row["case_id"] for row in fixed_plan()]
    reviews = {name: value for name, value in before.items() if name.startswith("review-")}
    if (protocol["schema_version"] != "serveflow-order-answer-recheck-protocol-v1"
            or protocol["git_head"] != OLD_COMMIT or protocol["plan"] != fixed_plan()
            or authorization.get("schema_version") != "serveflow-order-answer-recheck-authorization-v1"
            or authorization.get("protocol_sha256") != OLD_PROTOCOL
            or authorization.get("root_binding_sha256") != base._root_hash(OLD_ROOT)
            or authorization.get("initial_chain_hash") != ledger.initial_chain_hash
            or authorization.get("plan") != protocol["plan"]
            or type(authorization.get("max_requests")) is not int or authorization["max_requests"] != 24
            or type(state["attempts"]) is not int or state["attempts"] != 16
            or state["paused"] or state["pending"] or state["active_case"] is not None
            or state["completed_cases"] != cases or state["chain_hash"] != OLD_CHAIN
            or len(sealed["operations"]) != 16 or set(sealed["cases"]) != set(cases)
            or set(reviews) != {"review-" + case + ".json" for case in cases}):
        base._fail("current_task_ablation_prior_state_changed")
    if (report.get("schema_version") != "serveflow-order-answer-recheck-progress-v1"
            or report.get("protocol_sha256") != OLD_PROTOCOL
            or report.get("trial_kind") != "order-answer-recheck-v1"
            or report.get("candidate_baseline") != previous.BASELINE
            or report.get("trial_request_limit") != 24
            or report.get("reserved_requests") != 16 or report.get("sealed_physical") != 16
            or report.get("pending_physical") != 0 or report.get("paused") is not False
            or report.get("planned_jobs") != 6 or report.get("completed_jobs") != 6
            or report.get("semantic_reviews_imported") != 6 or report.get("review_ready") is not True
            or report.get("counts") != _STORED_COUNTS or report.get("tool_counts") != _STORED_COUNTS
            or type(report.get("cases")) is not list
            or [row.get("logical_case_id") for row in report["cases"]] != cases):
        base._fail("current_task_ablation_prior_report_changed")
    for row in report["cases"]:
        key = row["logical_case_id"]
        if (row.get("status") != "passed" or row.get("tool_status") != "passed"
                or row.get("artifact_sha256") != before[f"case-{key}.json"]
                or row.get("review") != bench.read_json(_safe_file(OLD_ROOT / f"review-{key}.json"))):
            base._fail("current_task_ablation_prior_report_binding_changed")
    if (_tree_files(OLD_ROOT) != before
            or bench.sha256(_safe_file(OLD_REPORT)) != OLD_REPORT_SHA256):
        base._fail("current_task_ablation_prior_changed_during_read")
    return {
        "schema_version": "serveflow-current-task-ablation-prior-proof-v1",
        "older_trials": older,
        "latest_fixed_16": {
            "protocol_sha256": OLD_PROTOCOL, "authorization_sha256": OLD_AUTHORIZATION,
            "journal_sha256": OLD_JOURNAL, "tree_sha256": OLD_TREE, "files": 74,
            "chain_hash": OLD_CHAIN, "report_sha256": OLD_REPORT_SHA256,
            "reserved_requests": 16, "authorized_requests": 24, "completed_cases": 6, "reviews": 6,
            "review_files_sha256": canonical_payload_hash(reviews),
            "recorded_strict_result": {"passed": 6, "planned": 6, "rescored": False},
            "recorded_tool_result": {"passed": 6, "planned": 6, "rescored": False},
            "paused": False, "pending_attempts": 0,
        },
        "old_scores_recomputed": False, "old_allowance_reused": False, "new_authorization_granted": 0,
    }
