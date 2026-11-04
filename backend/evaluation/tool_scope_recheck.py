"""One fixed D-only regression after narrowing required-step advertised tools.

This is a new batch and send budget, never a continuation or rescore of the
four-arm trial. Preparation is offline. Old bytes, scores and ledgers stay intact.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import difflib
import hashlib
import json
import subprocess

from core.model_deadline import EvaluationDeadlinePolicy
from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as business
from evaluation import quality_phase_trial as phase, quality_trial as quality, quality_ablation as design
from evaluation.intent_transport import canonical_payload_hash
from evaluation.phase_deadline_transport import PHASE_HTTP_LIMITS, SDK_TIMEOUT_S, PhaseDeadlineTaskHTTPTransport
from evaluation.quality_budget import QualityBudget
from evaluation.quality_scoring import shared_checks
from evaluation.task_recheck import _tree, history


ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-business-v7-tool-scope"
BATCH = "serveflow-task-business-v7-tool-scope"
CANDIDATE = "required-step-single-tool-exposure-v1"
OLD_ROOT = phase.ROOT
OLD = {
    "commit": "c65b993d2e72faed06858d5f8a5093b7014a4bf6",
    "protocol": "70eff19f27606737e467a2e7f00c60d779aed4109f05199f9881e3aaa4e3a3be",
    "authorization": "c3ec5638df64bd4dc40072217e35c51847c07005f36865f90194eaadeccf5bbd",
    "journal": "036d09c80c9a457ad11340ca24844d35375c245c13475035697b696135e10910",
    "chain": "4d7ae4b1c01157d3d8aeb9b3f6514e0b18d5546af4703f4825625cf715344ee7",
    "report": "273b57ce00e0b2d02f6783c10700aeae5f2254583fc76548930d9ba121d9a22f",
    "tree": "e83387249a0c2aa64a535c1ff0b73081243b8273f650af1251aca9275add5050",
}
AGENT_PATH = "agents/agent_orchestrator.py"
APPROVED_PATCH_SHA256 = "7aa76378d44c13d6eab4c60d1e29ba248bedc5ffeabcc4912e4b079953342949"
_EXTRA = ("evaluation/tool_scope_recheck.py", "tests/test_tool_scope_recheck.py",
          "tests/test_required_tool_exposure.py")
POLICY = {**deepcopy(business.POLICY), "grading_version": quality.REVIEW_POLICY["version"],
    "stage_clock": deepcopy(phase.CLOCK_POLICY),
    "deadline_contract": deepcopy(phase.POLICY["deadline_contract"]),
    "candidate_scope": "only advertise the single program-required tool during its bound step; retain strict native response rejection and the full role registry",
    "payload_persistence": quality.POLICY["payload_persistence"],
    "order": "original twelve D-behavior cases once in original order, thirteen turns; no retries or output selection",
    "comparison": "new independent fixed regression, not a fresh four-arm ablation or historical score replacement"}


def _git_agent():
    try:
        value = subprocess.run(["git", "show", OLD["commit"] + ":backend/" + AGENT_PATH],
            cwd=bench.BACKEND_ROOT.parent, capture_output=True, check=True, timeout=10).stdout
        if not value or len(value) > phase._SOURCE_PROOF_LIMIT:
            raise ValueError
        value.decode("utf-8")
        return value
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        base._fail("tool_scope_old_agent_invalid")


def candidate_proof(old_protocol):
    """Exact file-set proof and one approved diff; no generic source exemption."""
    old, current = phase._git_semantic_files(), phase._current_semantic_files()
    if not old or set(old) != set(current):
        base._fail("tool_scope_business_file_set_changed")
    before = _git_agent()
    # The source tree is public code, not a private 0700 evidence directory.
    # The complete scanner above has already rejected links/nonregular files.
    after = (bench.BACKEND_ROOT / AGENT_PATH).read_bytes()
    if (hashlib.sha256(before).hexdigest() != old.get(AGENT_PATH)
            or old_protocol["artifacts"].get(AGENT_PATH) != old.get(AGENT_PATH)):
        base._fail("tool_scope_old_agent_identity_changed")
    patch = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True),
        after.decode().splitlines(keepends=True), fromfile="before/" + AGENT_PATH,
        tofile="after/" + AGENT_PATH, n=3)).encode()
    changed = [name for name in old if old[name] != current[name]]
    if (changed != [AGENT_PATH] or hashlib.sha256(after).hexdigest() != current[AGENT_PATH]
            or hashlib.sha256(patch).hexdigest() != APPROVED_PATCH_SHA256):
        base._fail("tool_scope_unapproved_business_change")
    # Runtime, Judge, data, questions, parameters and old scoring stay identical.
    # The parent runner's five fixed-default hooks are frozen as new execution
    # infrastructure and tested separately, not treated as a business change.
    prior = phase._semantic_files(old_protocol["artifacts"])
    now = phase._semantic_files(phase._artifacts())
    for values in (prior, now):
        values.pop(AGENT_PATH, None)
        values.pop("evaluation/task_business_regression.py", None)
    if not prior or prior != now:
        base._fail("tool_scope_other_semantic_inputs_changed")
    return {"old_source_commit": phase.V1["commit"], "old_agent_commit": OLD["commit"],
        "complete_business_file_count": len(old), "complete_old_files_sha256": canonical_payload_hash(old),
        "complete_current_files_sha256": canonical_payload_hash(current),
        "only_changed_business_file": AGENT_PATH, "approved_unified_patch_sha256": APPROVED_PATCH_SHA256,
        "old_agent_sha256": old[AGENT_PATH], "candidate_agent_sha256": current[AGENT_PATH],
        "unchanged_other_semantic_files_sha256": canonical_payload_hash(prior)}


def archived_v2():
    """Use original schema/Git/hash chain only; never old current-code verify."""
    if _tree(OLD_ROOT) != OLD["tree"]:
        base._fail("tool_scope_old_tree_changed")
    for name, key in (("protocol.json", "protocol"), ("authorization.json", "authorization"),
                      ("ledger/journal.jsonl", "journal"), ("reviewed-report.json", "report")):
        if bench.sha256(base._path(OLD_ROOT, name)) != OLD[key]:
            base._fail("tool_scope_old_record_changed")
    p = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
    history._archived_sources(p, "artifacts")
    if (p["git_head"] != OLD["commit"] or p["root_binding_sha256"] != base._root_hash(OLD_ROOT)
            or p["plan"] != phase.fixed_plan() or p["review_policy"] != quality.REVIEW_POLICY
            or p["http_limits"] != PHASE_HTTP_LIMITS or p["policy"] != phase.POLICY):
        base._fail("tool_scope_old_identity_changed")
    # Replaying the unchanged accounting schema reads old claims; it cannot
    # append events, reserve sends, restart workers or rewrite a score here.
    ledger = QualityBudget(OLD_ROOT / "ledger", OLD["protocol"], p["plan"])
    state, sealed = ledger.status(), ledger.results()
    if (state["attempts"] != 208 or state["pending"] or state["paused"]
            or state["active_case"] is not None or state["chain_hash"] != OLD["chain"]
            or state["completed_cases"] != [row["case_id"] for row in p["plan"]]
            or len(sealed["operations"]) != 208 or len(sealed["cases"]) != 48):
        base._fail("tool_scope_old_ledger_changed")
    for op in sealed["operations"].values():
        if (bench.sha256(base._path(OLD_ROOT, f"physical-{op['attempt']:03d}.json")) != op["output_sha256"]
                or quality._read_request(OLD_ROOT, op["request_bytes_sha256"])["payload_sha256"] != op["payload_sha256"]):
            base._fail("tool_scope_old_physical_changed")
    for key, value in sealed["cases"].items():
        if bench.sha256(base._path(OLD_ROOT, f"case-{key}.json")) != value["artifact_sha256"]:
            base._fail("tool_scope_old_case_changed")
    older, older_proof = phase._archived_v1()
    if (p["prior_proof"]["old_v1"] != older_proof
            or p["prior_proof"]["older_proof_sha256"] != canonical_payload_hash(older["prior_proof"])
            or _tree(OLD_ROOT) != OLD["tree"]):
        base._fail("tool_scope_old_history_changed")
    return p


def prior_proof():
    old = archived_v2()
    return {"schema_version": "serveflow-tool-scope-readonly-proof-v1", "old_v2": deepcopy(OLD),
        "old_reserved_requests": 208, "old_sealed_requests": 208, "old_completed_cases": 48,
        "old_scores_recomputed": False, "old_allowance_reused": False,
        "older_proof_sha256": canonical_payload_hash(old["prior_proof"]),
        "candidate_proof": candidate_proof(old)}


def _artifacts():
    return {**phase._artifacts(), **phase._current_semantic_files(),
        **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in _EXTRA}}


class ToolScopeRecheck(business.BusinessRegression):
    root_path, batch, candidate = ROOT, BATCH, CANDIDATE
    protocol_schema = "serveflow-task-tool-scope-protocol-v7"
    authorization_schema = "serveflow-task-tool-scope-authorization-v7"
    report_schema = "serveflow-task-tool-scope-report-v7"

    def artifact_proof(self): return _artifacts()
    def historical_proof(self): return prior_proof()
    def execution_policy(self): return deepcopy(POLICY)
    def http_limits(self): return deepcopy(PHASE_HTTP_LIMITS)
    def sdk_timeout_s(self): return SDK_TIMEOUT_S
    def deadline_policy(self): return EvaluationDeadlinePolicy(overhead_limit_s=30, cleanup_grace_s=.25)
    def review_policy(self): return deepcopy(quality.REVIEW_POLICY)
    def grading_version(self): return quality.REVIEW_POLICY["version"]
    def make_transport(self, *args, **kwargs): return PhaseDeadlineTaskHTTPTransport(*args, **kwargs)

    def verify(self):
        p = super().verify()
        old = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
        for key in ("model", "phase_parameters", "agent_parameters", "review_policy", "dependencies",
                    "base_url", "messages_url", "effective_environment", "data_proof_sha256", "offline_data_proof"):
            if p[key] != old[key]:
                base._fail("tool_scope_business_parameters_changed")
        # Four-arm initial inputs differ only in private operation identity.
        # Compare each D request including both exact wire and canonical hashes.
        previous = {item["case_id"]: item for item in old["initial_intent_requests"]}
        for item in p["initial_intent_requests"]:
            prior = deepcopy(previous.get("D-" + item["case_id"]))
            if (prior is None or prior.get("arm") != "D"
                    or prior.get("source_case_id") != item["case_id"]):
                base._fail("tool_scope_old_d_payload_missing")
            prior.pop("arm")
            prior.pop("source_case_id")
            prior["case_id"] = item["case_id"]
            if item != prior:
                base._fail("tool_scope_initial_d_payload_changed")
        if old["cases_sha256"] != p["cases_sha256"]:
            base._fail("tool_scope_questions_changed")
        return p

    def _sealed(self, p):
        sealed = super()._sealed(p)
        for op in sealed["operations"].values():
            if op["output_sha256"] is not None:
                if quality._read_request(self.root, op["request_bytes_sha256"])["payload_sha256"] != op["payload_sha256"]:
                    base._fail("tool_scope_request_changed")
        return sealed

    async def run_case(self, case, sdk, data, *, ensure_healthy):
        result = await super().run_case(case, sdk, data, ensure_healthy=ensure_healthy)
        result["shared_checks"] = shared_checks(case, result)
        return result

    def report(self):
        diagnostic = super().report()
        p = self.verify()
        sealed = self._sealed(p)
        rows = []
        for source, legacy in zip(business.cases(), diagnostic["cases"]):
            key = source["id"]
            row = {"case_id": key, "status": "unknown" if diagnostic["ledger"]["active_case"] == key else "not_executed"}
            if key in sealed["cases"]:
                artifact = bench.read_json(base._path(self.root, f"case-{key}.json"))
                checks = shared_checks(source, artifact)
                if (artifact.get("shared_checks") != checks or artifact.get("case_id") != key
                        or artifact.get("protocol_sha256") != diagnostic["protocol_sha256"]
                        or artifact.get("grading_version") != self.grading_version()):
                    base._fail("tool_scope_case_binding_changed")
                hard = all(checks["checks"].values()) and not checks["blockers"]
                row.update(shared_checks=checks, status="pending_review" if hard else "failed",
                    judge_scores=legacy["judge_scores"], execution_status=artifact["execution_status"],
                    data_kind=artifact["data_kind"], evidence_kind=artifact["evidence_kind"])
                if "semantic_review" in legacy:
                    review = legacy["semantic_review"]
                    row["review"] = review
                    row["status"] = "passed" if hard and all(review["dimensions"].values()) and not review["blocking_errors"] else "failed"
            rows.append(row)

        def metric(items, *, tool=False):
            outcomes = []
            for row in items:
                status = row["status"]
                if tool and "shared_checks" in row:
                    status = ("pending_review" if "review" not in row else "passed"
                        if row["shared_checks"]["checks"]["tool_task_correct"] and row["review"]["dimensions"]["tool_permissions"] else "failed")
                outcomes.append(status)
            counts = Counter(outcomes)
            return {"numerator": counts["passed"], "denominator": len(items), "value": counts["passed"] / len(items),
                "counts": {name: counts[name] for name in ("passed", "failed", "pending_review", "unknown", "not_executed")}}

        physical = [bench.read_json(base._path(self.root, f"physical-{op['attempt']:03d}.json"))
            for op in sealed["operations"].values() if op["output_sha256"] is not None]
        return {"schema_version": self.report_schema, "batch": self.batch, "candidate": self.candidate,
            "protocol_sha256": diagnostic["protocol_sha256"], "review_policy": self.review_policy(),
            "planned_cases": 12, "planned_turns": 13, "planned_denominator": 12,
            "authorized_requests": diagnostic["authorized_requests"], "actual_requests": diagnostic["reserved_requests"],
            "sealed_requests": diagnostic["sealed_requests"], "ledger": diagnostic["ledger"], "cases": rows,
            "correct_handling": metric(rows),
            "supported_completion": metric([row for row in rows if row["case_id"] in design.SUPPORTED]),
            "tool_tasks": metric([row for row in rows if row["case_id"] in design.TOOLS], tool=True),
            "boundary_handling": metric([row for row in rows if row["case_id"] in design.BOUNDARY]),
            "usage": diagnostic["usage"], "physical_resources": diagnostic["physical_resources"],
            "phase_counts": dict(Counter(item["phase"] for item in physical)),
            "outcomes": dict(Counter(item["outcome"] for item in physical)),
            "stage_clock_policy": deepcopy(POLICY["stage_clock"]), "http_limits": deepcopy(p["http_limits"]),
            "review_dimensions": diagnostic["review_dimensions"], "reading_burden": diagnostic["reading_burden"],
            "business_projection_gate_diagnostic_only": {"cases": diagnostic["cases"],
                "counts": diagnostic["business_counts"], "legacy_judge_gated_counts": diagnostic["legacy_judge_gated_counts"]},
            "model_quality_evidence": bool(physical) and all(item["evidence_kind"] == "live-model" for item in physical)
                and all(row.get("data_kind") == "live-mysql-and-isolated-chroma" for row in rows),
            "limitations": ["twelve known author-visible cases and static synthetic records, once each; no generalization or independent blind claim",
                "new D-only candidate regression; do not splice old outputs or scores, or claim a new four-arm causal effect",
                "shared seven-dimensional semantic rubric is primary; fresh reads, exact projection, role names and legacy Judge gates are diagnostics",
                "unchanged active phase values and phase-bound wire values; SDK/httpx ceiling 30s is configuration, not measured network time",
                "unknown, unexecuted and pending review remain in fixed denominators; no automatic retry or refund of uncertain sends",
                "no browser, Redis/background-memory or production-readiness acceptance"]}


def main(argv=None):
    return business._main(ToolScopeRecheck, "evaluation.tool_scope_recheck", argv)


if __name__ == "__main__":
    raise SystemExit(main())
