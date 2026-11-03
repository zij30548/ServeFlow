"""Fixed v5 observation recheck, using the unchanged v4 business rubric.

Only full pre-send verification scheduling changes: an explicit threaded
transport keeps synchronous read-only admission work off the event loop. The
original twelve questions, model, parameters, eight-second network deadline,
deterministic facts and review policy remain unchanged. New authorization is
required; no old paused operation or unused allowance is resumed.
"""
from __future__ import annotations

from copy import deepcopy
import json

from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as business
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_recheck import _tree, history


ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-business-v5-observation"
BATCH = "serveflow-task-business-v5-observation"
CANDIDATE = "v4-business-candidate-threaded-readonly-send-admission"
OLD_ROOT = business.ROOT
OLD_EXECUTION_COMMIT = "81e6b590e41260ebca8de041975c51722e711be9"
OLD_PROTOCOL = "65d3f5580d2fc2761b92a8a0cd4f1718a49dfa5b8d42eae4a93122ff00817728"
OLD_AUTHORIZATION = "41d853d3d9d3c254b2c6602dcacad437f2e4de27d2d00c7ae39925aca4441baf"
OLD_JOURNAL = "fd041e71e443c21b967d7e499231cf98946e086829335c9e4a3e8e5f7e33688a"
OLD_CHAIN = "ab60ebbf92b08bbb47e7b2af79af99e5c10ac1d03c726165f77099039b77932f"
OLD_TREE = "79c27975158d84d77fefcd15c0ae06108de67f8f2383941089fee810b06ac2e0"
OBSERVATION_POLICY = {
    "version": "task-observation-v5",
    "transport": "evaluation.task_transport.ThreadedTaskHTTPTransport",
    "pre_send_verification": "all frozen source/dependency/environment/data/history checks before every send, in a read-only worker thread",
    "send_boundary": "after verification returns, recheck fault/stop/evidence on the event loop, atomically reserve, then send",
    "deadline": "network remains eight seconds; existing SDK/phase/production deadlines and retry limits unchanged",
    "cancellation": "cancelled admission never reserves or sends; an already-sent cancellation remains charged and stops the batch",
    "statistics": "wall starts before queue admission and ends before sealing; includes parsing/admission/scheduling, excludes its own receipt fsync/ledger.finish; network is not supplier-only latency",
    "grading": "task-business-v4 unchanged; Judge remains auxiliary with a separate original-threshold column",
    "history": "v4 sealed at twelve attempts; all old outcomes/scores retained; no continuation or refund",
}


def business_source_proof(protocol):
    """Production facts/logic/config are byte-identical to the sealed v4."""
    def production(paths):
        return {name: digest for name, digest in paths.items() if not name.startswith(("evaluation/", "tests/"))}
    before = production(protocol["artifacts"])
    current = production(business._artifacts())
    if not before or current != before:
        base._fail("task_observation_business_source_changed")
    config = base._configuration()
    if (protocol["model"] != base.MODEL or protocol["base_url"] != config["base_url"]
            or protocol["limits"] != business.LIMITS or protocol["dependencies"] != business.engine._dependencies()
            or protocol["http_limits"] != business.engine.HTTP_LIMITS
            or protocol["phase_parameters"] != json.loads(json.dumps(business.engine.PHASE_PARAMETERS))
            or protocol["agent_parameters"] != json.loads(json.dumps(business.engine.AGENT_PARAMETERS))
            or protocol["effective_environment"] != business.effective_environment()
            or protocol["policy"] != business.POLICY or protocol["review_policy"] != business.REVIEW_POLICY
            or protocol["cases_sha256"] != canonical_payload_hash({"cases": business.cases()})):
        base._fail("task_observation_business_parameters_changed")
    return {"business_source_sha256": canonical_payload_hash({"artifacts": before}),
            "production_files": len(before), "business_source_unchanged": True,
            "question_and_grading_unchanged": True, "model_and_parameters_unchanged": True}


def prior_proof():
    """Verify sealed v4 and its Git sources read-only; chain to v3 and older."""
    older = business.prior_proof()
    before = _tree(OLD_ROOT)
    if before != OLD_TREE:
        base._fail("task_observation_old_tree_changed")
    for name, expected in {"protocol.json": OLD_PROTOCOL, "authorization.json": OLD_AUTHORIZATION,
                           "ledger/journal.jsonl": OLD_JOURNAL}.items():
        if bench.sha256(base._path(OLD_ROOT, name)) != expected:
            base._fail("task_observation_old_evidence_changed")
    protocol = bench.read_json(base._path(OLD_ROOT, "protocol.json"))
    if (protocol.get("schema_version") != "serveflow-task-business-protocol-v4"
            or protocol.get("batch") != business.BATCH or protocol.get("git_head") != OLD_EXECUTION_COMMIT
            or protocol.get("root_binding_sha256") != base._root_hash(OLD_ROOT)
            or protocol.get("prior_proof") != older or protocol.get("authorized_model_requests") != 0):
        base._fail("task_observation_old_protocol_changed")
    history._archived_sources(protocol, "artifacts")
    unchanged = business_source_proof(protocol)
    budget = business.BusinessTaskBudget(OLD_ROOT / "ledger", OLD_PROTOCOL, protocol["plan"])
    state, sealed = budget.status(), budget.results()
    if (state["attempts"] != 12 or state["pending"] or state["paused"] is not True
            or state["active_case"] != "e03" or state["completed_cases"] != ["e01", "e02"]
            or state["chain_hash"] != OLD_CHAIN or len(sealed["operations"]) != 12
            or set(sealed["cases"]) != {"e01", "e02"}):
        base._fail("task_observation_old_ledger_changed")
    approval = bench.read_json(base._path(OLD_ROOT, "authorization.json"))
    if (approval.get("schema_version") != "serveflow-task-business-authorization-v4"
            or approval.get("protocol_sha256") != OLD_PROTOCOL or approval.get("max_requests") != 312
            or approval.get("root_binding_sha256") != base._root_hash(OLD_ROOT)
            or approval.get("initial_chain_hash") != budget.initial_chain_hash or approval.get("plan") != protocol["plan"]):
        base._fail("task_observation_old_authorization_changed")
    for operation in sealed["operations"].values():
        if bench.sha256(base._path(OLD_ROOT, f"physical-{operation['attempt']:03d}.json")) != operation["output_sha256"]:
            base._fail("task_observation_old_physical_changed")
    for key, item in sealed["cases"].items():
        if bench.sha256(base._path(OLD_ROOT, f"case-{key}.json")) != item["artifact_sha256"]:
            base._fail("task_observation_old_case_changed")
    if _tree(OLD_ROOT) != before:
        base._fail("task_observation_old_tree_changed_during_read")
    return {"schema_version": "serveflow-task-v4-readonly-proof-v1", "source_commit": OLD_EXECUTION_COMMIT,
        "tree_sha256": before, "protocol_sha256": OLD_PROTOCOL, "authorization_sha256": OLD_AUTHORIZATION,
        "journal_sha256": OLD_JOURNAL, "chain_hash": OLD_CHAIN, "reserved_requests": 12, "sealed_requests": 12,
        "completed_cases": ["e01", "e02"], "active_case": "e03",
        "business_counts": {"passed": 2, "failed": 0, "unknown": 1, "not_executed": 9},
        "legacy_judge_gated_counts": {"passed": 2, "failed": 0, "unknown": 1, "not_executed": 9},
        "older_proof_sha256": canonical_payload_hash(older), "old_allowance_reused": False,
        "old_scores_recomputed": False, "new_authorized_requests": 0, **unchanged}


def _artifacts():
    additions = ("evaluation/task_observation_recheck.py", "tests/test_task_observation_recheck.py",
                 "tests/test_task_threaded_admission.py")
    return {**business._artifacts(), **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in additions}}


class ObservationRecheck(business.BusinessRegression):
    root_path = ROOT
    batch = BATCH
    candidate = CANDIDATE
    protocol_schema = "serveflow-task-business-protocol-v5-observation"
    authorization_schema = "serveflow-task-business-authorization-v5-observation"
    report_schema = "serveflow-task-business-report-v5-observation"

    def artifact_proof(self):
        return _artifacts()

    def historical_proof(self):
        return prior_proof()

    def execution_policy(self):
        return {**deepcopy(business.POLICY), "observation": deepcopy(OBSERVATION_POLICY)}

    def make_transport(self, *args, **kwargs):
        from evaluation.task_transport import ThreadedTaskHTTPTransport
        return ThreadedTaskHTTPTransport(*args, **kwargs)

    def report(self):
        value = super().report()
        protocol = bench.read_json(base._path(self.root, "protocol.json"))
        return {**value, "grading_version": "task-business-v4",
            "observation_policy": deepcopy(OBSERVATION_POLICY),
            "unchanged_business_proof": {key: protocol["prior_proof"][key]
                for key in ("business_source_sha256", "business_source_unchanged",
                            "question_and_grading_unchanged", "model_and_parameters_unchanged")}}


def main(argv=None):
    return business._main(ObservationRecheck, "evaluation.task_observation_recheck", argv)


if __name__ == "__main__":
    raise SystemExit(main())
