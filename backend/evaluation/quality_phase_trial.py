"""Independent four-arm trial with unchanged active, phase-bound wire limits.

Only the evaluation deadline contract changes. No historical operation, reply,
review or unused allowance is resumed. Preparation/verification never sends.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from evaluation import benchmark as bench, intent_trial as base
from evaluation import quality_trial as previous
from evaluation import task_business_regression as business, task_deadline_recheck as clock
from evaluation.intent_transport import canonical_payload_hash
from evaluation.phase_deadline_transport import (
    PHASE_DEADLINES, PHASE_HTTP_LIMITS, SDK_TIMEOUT_S, PhaseDeadlineTaskHTTPTransport,
)
from evaluation.quality_budget import QualityBudget, MAX_REQUESTS, fixed_plan
from evaluation.task_recheck import _tree, history


ROOT = bench.BACKEND_ROOT / "data/benchmarks/quality-factorial-v2-phase-deadline"
BATCH = "serveflow-quality-factorial-v2-phase-deadline"
CANDIDATE = "unchanged-2x2-business-with-phase-bound-deadlines"
V1 = {
    "commit": "53eed9cc91b01fb507f407b34616cc46ab8f8923",
    "protocol": "dab45a9b01fe04301c4caedf345d8b4c10b944916bab929c0602b047693cc016",
    "authorization": "3c493f06dde35264edda3b94b1546efc20b1181a8e647427bd444050b15d18d3",
    "journal": "73e141f30d7ef9683fcad81feb36b6069c1bda25d0097b68c00a210d01a1391c",
    "chain": "c1c1b8c6a1992b7702492a00f4b71193cc33bac71a90a80672f71735b90365ec",
    "report": "f7b7601e954bb6c36ccb4b1c6dc1c33433a6600a101e0b53848e89551c7bf8a9",
    "tree": "51321840385bb62a207c7bdfa92ecfb9539cdc3d184d4ab8beee00ec625ebc8c",
}
CLOCK_POLICY = {
    **deepcopy(clock.CLOCK_POLICY),
    "version": "model-stage-and-phase-wire-clock-v2",
    "network_deadline_s": "phase-specific, not the previous global 8s",
    "network_deadlines_by_phase_s": deepcopy(PHASE_DEADLINES),
    "sdk_timeout_s": SDK_TIMEOUT_S,
    "httpx_timeout_s": SDK_TIMEOUT_S,
    "wire_timeout_extensions": "validate four SDK/httpx values are 30, tighten to the bound phase before physical delegate entry",
    "active_model_stage_budgets_changed": False,
    "production": "unchanged application clients, original stage values and original evaluation workers; this trusted runner only",
    "comparison": "new execution contract; do not combine v1 or v6 successes or compare across execution versions as quality uplift",
}
POLICY = {
    **deepcopy(previous.POLICY),
    "stage_clock": deepcopy(CLOCK_POLICY),
    "deadline_contract": "SDK/httpx ceiling 30s; per-request network and each unchanged active stage use 8/15/10/30/8/8s by phase; no retries",
}
_EXTRA = (
    "evaluation/quality_phase_trial.py", "evaluation/phase_deadline_transport.py",
    "tests/test_quality_phase_trial.py", "tests/test_phase_deadline_transport.py",
)
_SOURCE_DIRS = ("core", "agents", "tooling", "demo_business")
_DOCUMENT_DIRS = ("knowledge", "skills")
_SOURCE_PROOF_LIMIT = 8 * 1024 * 1024


def _protected_path(name):
    parts = Path(name).parts
    if (not parts or any(part in {"", ".", ".."} for part in parts)
            or any(ord(char) < 32 for char in name)):
        return False
    if parts[0] in _SOURCE_DIRS and "__pycache__" in parts:
        return False
    suffix = Path(name).suffix.lower()
    return (parts[0] in _SOURCE_DIRS and suffix == ".py"
            or parts[0] in _DOCUMENT_DIRS and suffix in {".md", ".json", ".txt"})


def _git_semantic_files(source_root=bench.BACKEND_ROOT):
    """Read immutable old source in two Git calls; never cache current files."""
    try:
        result = subprocess.run(["git", "ls-tree", "-rz", "--full-tree", V1["commit"], "--",
            *("backend/" + folder for folder in _SOURCE_DIRS + _DOCUMENT_DIRS)],
            cwd=Path(source_root).parent, capture_output=True, check=True, timeout=10)
        if len(result.stdout) > _SOURCE_PROOF_LIMIT:
            raise ValueError
        objects = []
        for item in result.stdout.split(b"\0"):
            if not item:
                continue
            header, path = item.split(b"\t", 1)
            mode, kind, oid = header.split(b" ")
            name = path.decode("utf-8")
            if (mode not in {b"100644", b"100755"} or kind != b"blob"
                    or len(oid) != 40 or any(c not in b"0123456789abcdef" for c in oid)
                    or not name.startswith("backend/")):
                raise ValueError
            relative = name.removeprefix("backend/")
            if _protected_path(relative):
                objects.append((relative, oid))
        if not objects or len({name for name, _ in objects}) != len(objects):
            raise ValueError
        result = subprocess.run(["git", "cat-file", "--batch"],
            input=b"\n".join(oid for _, oid in objects) + b"\n",
            cwd=Path(source_root).parent, capture_output=True, check=True, timeout=10)
        if len(result.stdout) > _SOURCE_PROOF_LIMIT:
            raise ValueError
        position, values = 0, {}
        for name, oid in objects:
            end = result.stdout.index(b"\n", position)
            actual_oid, kind, length = result.stdout[position:end].split(b" ")
            size = int(length)
            if actual_oid != oid or kind != b"blob" or not 0 <= size <= _SOURCE_PROOF_LIMIT:
                raise ValueError
            position = end + 1
            content = result.stdout[position:position + size]
            if len(content) != size or result.stdout[position + size:position + size + 1] != b"\n":
                raise ValueError
            position += size + 1
            values[name] = hashlib.sha256(content).hexdigest()
        if position != len(result.stdout):
            raise ValueError
        return values
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("quality_phase_old_git_source_invalid")


def _current_semantic_files(source_root=bench.BACKEND_ROOT):
    values = {}
    try:
        for folder in _SOURCE_DIRS + _DOCUMENT_DIRS:
            root = Path(source_root) / folder
            if not root.is_dir() or root.is_symlink():
                raise ValueError
            for parent, dirs, files in os.walk(root, followlinks=False):
                for child in dirs:
                    if (Path(parent) / child).is_symlink():
                        raise ValueError
                if folder in _SOURCE_DIRS:
                    dirs[:] = [name for name in dirs if name != "__pycache__"]
                for child in files:
                    path = Path(parent) / child
                    if path.is_symlink():
                        raise ValueError
                    relative = str(path.relative_to(source_root))
                    if _protected_path(relative):
                        if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > _SOURCE_PROOF_LIMIT:
                            raise ValueError
                        values[relative] = bench.sha256(path)
        return values
    except (OSError, ValueError):
        base._fail("quality_phase_current_source_invalid")


def complete_source_proof(source_root=bench.BACKEND_ROOT):
    old, current = _git_semantic_files(source_root), _current_semantic_files(source_root)
    if old != current:
        base._fail("quality_phase_complete_business_inputs_changed")
    return {"old_commit": V1["commit"], "complete_source_file_set_sha256": canonical_payload_hash(old),
            "complete_source_file_count": len(old), "complete_source_file_set_and_bytes_unchanged": True,
            "scope": "core/agents/tooling/demo_business Python; all knowledge/skills Markdown/JSON/TXT including non-SKILL rules"}


def _semantic_files(artifacts):
    # Core is unchanged as well: no broad exemption for timing-related files.
    # The new wire policy lives in evaluation/, not business inputs or Skills.
    explicit = {
        "evaluation/quality_arm_runtime.py", "evaluation/quality_scoring.py",
        "evaluation/benchmark/quality-ablation-v1.json", "evaluation/benchmark/task-e2e-v1.json",
        "evaluation/benchmark/tasks-v1.json",
        "evaluation/task_runtime.py", "evaluation/task_evidence.py", "evaluation/task_data.py",
        "evaluation/evaluator.py", "evaluation/task_business_regression.py", "evaluation/task_trial.py",
        "evaluation/task_repair.py", "evaluation/task_recheck.py", "evaluation/task_deadline_recheck.py",
        "evaluation/intent_trial.py", "evaluation/benchmark.py", "evaluation/quality_budget.py",
        "evaluation/quality_ablation.py",
    }
    return {name: digest for name, digest in artifacts.items()
            if name.startswith(("core/", "agents/", "tooling/", "demo_business/", "knowledge/", "skills/"))
            or name in explicit}


def business_input_proof(old_protocol):
    """Bind the full unchanged semantic file set, not just surviving old names."""
    before = _semantic_files(old_protocol["artifacts"])
    now = _semantic_files(previous._artifacts())
    if not before or before != now:
        base._fail("quality_phase_business_inputs_changed")
    return {"semantic_files_sha256": canonical_payload_hash(before),
            "semantic_file_count": len(before), "semantic_file_set_and_bytes_unchanged": True,
            "complete_source_proof": complete_source_proof(),
            "model": old_protocol["model"],
            "phase_parameters": deepcopy(old_protocol["phase_parameters"]),
            "agent_parameters": deepcopy(old_protocol["agent_parameters"]),
            "review_policy_sha256": canonical_payload_hash(old_protocol["review_policy"]),
            "cases_sha256": old_protocol["cases_sha256"],
            "initial_payload_identity": "all 48 initial intent canonical and wire hashes must match original v1, no old output replay"}


def _archived_v1():
    root = previous.ROOT
    if _tree(root) != V1["tree"]:
        base._fail("quality_phase_old_tree_changed")
    for name, key in (("protocol.json", "protocol"), ("authorization.json", "authorization"),
                      ("ledger/journal.jsonl", "journal"), ("reviewed-paused-report.json", "report")):
        if bench.sha256(base._path(root, name)) != V1[key]:
            base._fail("quality_phase_old_record_changed")
    p = bench.read_json(base._path(root, "protocol.json"))
    history._archived_sources(p, "artifacts")
    if (p["git_head"] != V1["commit"] or p["root_binding_sha256"] != base._root_hash(root)
            or p["plan"] != fixed_plan() or p["prior_proof"] != previous.prior_proof()
            or p["review_policy"] != previous.REVIEW_POLICY
            or p["phase_parameters"] != json.loads(json.dumps(previous.business.engine.PHASE_PARAMETERS))
            or p["agent_parameters"] != json.loads(json.dumps(previous.business.engine.AGENT_PARAMETERS))
            or p["model"] != base.MODEL or p["dependencies"] != previous.business.engine._dependencies()):
        base._fail("quality_phase_old_identity_changed")
    ledger = QualityBudget(root / "ledger", V1["protocol"], p["plan"])
    state, sealed = ledger.status(), ledger.results()
    planned_complete = [row["case_id"] for row in fixed_plan()[:18]]
    if (state["attempts"] != 100 or not state["paused"] or state["pending"]
            or state["active_case"] != "C-e05" or state["completed_cases"] != planned_complete
            or state["chain_hash"] != V1["chain"] or len(sealed["operations"]) != 100
            or set(sealed["cases"]) != set(planned_complete)):
        base._fail("quality_phase_old_ledger_changed")
    a = bench.read_json(base._path(root, "authorization.json"))
    if (a["protocol_sha256"] != V1["protocol"] or a["root_binding_sha256"] != base._root_hash(root)
            or a["initial_chain_hash"] != ledger.initial_chain_hash or a["max_requests"] != MAX_REQUESTS
            or a["plan"] != p["plan"]):
        base._fail("quality_phase_old_authorization_changed")
    for op in sealed["operations"].values():
        if bench.sha256(base._path(root, f"physical-{op['attempt']:03d}.json")) != op["output_sha256"]:
            base._fail("quality_phase_old_physical_changed")
        if previous._read_request(root, op["request_bytes_sha256"])["payload_sha256"] != op["payload_sha256"]:
            base._fail("quality_phase_old_request_changed")
    for identity, item in sealed["cases"].items():
        if bench.sha256(base._path(root, f"case-{identity}.json")) != item["artifact_sha256"]:
            base._fail("quality_phase_old_case_changed")
    if _tree(root) != V1["tree"]:
        base._fail("quality_phase_old_tree_changed_during_read")
    return p, {**deepcopy(V1), "reserved_requests": 100, "sealed_requests": 100,
        "completed_cases": 18, "unknown_cases": 1, "not_executed_cases": 29,
        "paused": True, "active_case": "C-e05", "old_scores_recomputed": False,
        "old_allowance_reused": False}


def prior_proof():
    p, archived = _archived_v1()
    return {"schema_version": "serveflow-quality-v1-readonly-proof-v1", "old_v1": archived,
            "older_proof_sha256": canonical_payload_hash(p["prior_proof"]),
            "unchanged_business_inputs": business_input_proof(p),
            "old_scores_recomputed": False, "old_allowance_reused": False}


def _artifacts():
    return {**previous._artifacts(),
            **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in _EXTRA}}


class QualityPhaseTrial(previous.QualityTrial):
    root_path = ROOT
    batch, candidate = BATCH, CANDIDATE
    protocol_schema = "serveflow-quality-phase-protocol-v2"
    authorization_schema = "serveflow-quality-phase-authorization-v2"
    report_schema = "serveflow-quality-phase-report-v2"

    def artifact_proof(self): return _artifacts()
    def historical_proof(self): return prior_proof()
    def execution_policy(self): return deepcopy(POLICY)
    def http_limits(self): return deepcopy(PHASE_HTTP_LIMITS)
    def sdk_timeout_s(self): return SDK_TIMEOUT_S
    def make_transport(self, *args, **kwargs): return PhaseDeadlineTaskHTTPTransport(*args, **kwargs)

    def verify(self):
        p = super().verify()
        old = bench.read_json(base._path(previous.ROOT, "protocol.json"))
        for key in ("model", "phase_parameters", "agent_parameters", "review_policy", "cases_sha256", "dependencies"):
            if p[key] != old[key]:
                base._fail("quality_phase_business_parameters_changed")
        for current, prior in zip(p["initial_intent_requests"], old["initial_intent_requests"]):
            if current != prior:
                base._fail("quality_phase_initial_payload_changed")
        return p


def main(argv=None):
    return business._main(QualityPhaseTrial, "evaluation.quality_phase_trial", argv)


if __name__ == "__main__":
    raise SystemExit(main())
