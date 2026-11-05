"""Read-only reporting for the explicitly resumed coverage experiment.

The caller verifies the original protocol, append-only continuation and ledger
before passing their snapshots here. This module never authorizes, sends,
initializes storage, changes an old score, or infers a semantic review. Old stop
files are historical evidence, not a new execution state.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from evaluation import benchmark as bench, benchmark_metrics as metrics
from evaluation import coverage_expansion as preparation
from evaluation.coverage_budget import MAX_REQUESTS, fixed_plan, job_specs
from evaluation.expanded_validation import DIMENSIONS


SCHEMA = "serveflow-coverage-continuation-progress-v1"
FAILURE_SCHEMA = "serveflow-coverage-continuation-case-failure-v1"
REVIEW_FIELDS = {"frame_case_id", "artifact_sha256", "reviewer", "method", "dimensions",
                 "blocking_errors", "reading_burden", "reason"}
TOOL_CHECKS = ("actual_trace_integrity", "actual_tool_permissions", "actual_trusted_identity",
               "actual_tool_arguments", "actual_business_evidence", "required_tools",
               "order_results_valid", "policy_results_valid", "trusted_request_binding")


def _fail(code):
    raise ValueError("coverage_continuation_report_" + code)


def _path(root, name):
    root = Path(root).absolute()
    if (root.resolve() != root or root.is_symlink() or type(name) is not str
            or Path(name).name != name or not name or name in {".", ".."}):
        _fail("path")
    path = root / name
    if path.is_symlink() or not path.is_file():
        _fail("file")
    return path


def _read(root, name, digest=None):
    path = _path(root, name)
    if digest is not None and bench.sha256(path) != digest:
        _fail("sealed_file_changed")
    return bench.read_json(path)


def _rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _protocol(protocol):
    if (type(protocol) is not dict or protocol.get("jobs") != job_specs()
            or protocol.get("plan") != fixed_plan() or type(protocol.get("cases")) is not dict):
        _fail("fixed_plan")
    return protocol


def validate_review(value, protocol, *, root, artifacts=None):
    """Validate one source-bound review without storing or accepting its claim.

    Generic execution failures may be reviewed, but can never become eligible.
    The root runner additionally protects the six pre-continuation reviews from
    replacement; this function does not expose an overwrite operation.
    """
    _protocol(protocol)
    bench._exact(value, REVIEW_FIELDS, "coverage_continuation_review_fields")
    identity = value["frame_case_id"]
    spec = next((row for row in protocol["jobs"]
                 if row["case_id"] == identity and row["suite"] != "retrieval"), None)
    if spec is None:
        _fail("review_case")
    path = _path(root, f"case-{identity}.json")
    if value["artifact_sha256"] != bench.sha256(path):
        _fail("review_binding")
    artifact = bench.read_json(path)
    if artifacts is not None and identity in artifacts and artifacts[identity] != artifact:
        _fail("review_artifact_changed")
    bench._exact(value["dimensions"], set(DIMENSIONS), "coverage_continuation_review_dimensions")
    if (any(type(flag) is not bool for flag in value["dimensions"].values())
            or value["reading_burden"] not in {"none", "some", "heavy"}
            or value["method"] not in {"executor-rubric-review", "controlled-fixture", "independent-human-review"}
            or artifact.get("evidence_kind") not in {"controlled-transport", "live-model"}
            or (artifact["evidence_kind"] == "controlled-transport") != (value["method"] == "controlled-fixture")):
        _fail("review_value")
    for name in ("reviewer", "reason"):
        bench._text(value[name], 4000)
    bench._strings(value["blocking_errors"])
    return value


def _native_checks(case, spec, artifact, source_root):
    if artifact.get("schema_version") == FAILURE_SCHEMA or "continuation_failure" in artifact:
        return {"checks": {"execution_failure": False}, "blockers": ["execution_failure"],
                "semantic_review_required": True, "quality_pass_claim": False}
    try:
        if spec["suite"] == "tasks":
            from evaluation.coverage_runtime import task_checks
            return task_checks(case, spec["arm"], artifact, source_root=source_root)
        from evaluation.coverage_conversation_runtime import conversation_checks
        return conversation_checks(case, artifact, source_root=source_root)
    except (KeyError, IndexError, TypeError, AttributeError):
        # Artifact structure is not a way to bypass the actual checker. Source
        # validation failures (ValueError) deliberately remain administrative.
        return {"checks": {"artifact_structure": False}, "blockers": ["artifact_structure"],
                "semantic_review_required": True, "quality_pass_claim": False}


def _flags_pass(observed):
    flags = observed.get("checks") if type(observed) is dict else None
    return (type(flags) is dict and bool(flags)
            and all(type(value) is bool and value for value in flags.values())
            and not observed.get("blockers"))


def _physical_matches(artifact, operations):
    """Every observed SDK phase must have a corresponding durable send."""
    observed = artifact.get("phase_observations")
    if (type(observed) is not list or type(artifact.get("physical_requests_observed")) is not int
            or artifact["physical_requests_observed"] != len(observed)
            or len(observed) != len(operations)):
        return False
    if any(type(row) is not dict or type(row.get("phase")) is not str
           or row.get("agent") is not None and type(row["agent"]) is not str for row in observed):
        return False
    return Counter((row["phase"], row.get("agent")) for row in observed) == Counter(
        (row["phase"], row.get("agent")) for row in operations)


def _physical_operations(root, state, sealed, planned):
    operations = sealed.get("operations")
    if type(operations) is not dict or type(state.get("pending")) is not list:
        _fail("operations")
    resources, seen, by_case = [], set(), {identity: [] for identity in planned}
    for operation_id, operation in operations.items():
        identity, number = operation.get("case_id"), operation.get("attempt")
        if (identity not in by_case or type(number) is not int or number < 1 or number in seen):
            _fail("physical_identity")
        seen.add(number)
        by_case[identity].append(operation)
        physical = None
        if operation.get("output_sha256"):
            physical = _read(root, f"physical-{number:03d}.json", operation["output_sha256"])
            if (physical.get("operation_id") != operation_id
                    or any(physical.get(key) != operation.get(key) for key in
                           ("case_id", "turn_id", "phase", "agent", "outcome",
                            "payload_sha256", "request_bytes_sha256"))):
                _fail("physical_binding")
        resources.append({"case_id": "physical:" + str(number),
            "status": "unknown" if operation_id in state["pending"] or physical is None
                else "success" if operation["outcome"] == "received" else "failed",
            "latency_ms": physical["latency_ms"] if physical is not None else None,
            "usage": {**(physical["response"]["usage"] if physical is not None else
                        {"input_tokens": None, "output_tokens": None}), "physical_requests": 1}})
    # TaskBudget.results exposes finished operations, status exposes unfinished
    # reservations. Each reservation contributes exactly one resource record.
    for operation_id in state["pending"]:
        if operation_id not in operations:
            resources.append({"case_id": "pending:" + operation_id, "status": "unknown",
                "latency_ms": None, "usage": {"input_tokens": None, "output_tokens": None,
                                              "physical_requests": 1}})
    if len(resources) != state.get("attempts"):
        _fail("physical_count")
    return by_case, resources


def _retrieval_scorer(source_root):
    # Validate one source snapshot, then reuse the existing pure source-row and
    # metric functions. Revalidating all 88 questions separately for every
    # planned empty result needlessly dominates read-only progress reporting.
    corpus, _, cases, _ = preparation.load_validated(source_root)
    by_id = {case["id"]: case for case in cases["retrieval"]}
    chunks = preparation.source_chunks(corpus["documents"])
    facts = {fact["fact_id"]: (doc["document_id"], fact["quote"])
             for doc in corpus["documents"] for fact in doc["facts"]}

    def score(case, rows, status):
        if case != by_id.get(case.get("id")) or status not in {"success", "failed", "cancelled", "unknown", "not_run"}:
            _fail("retrieval_case")
        preparation._validated_rows(rows, chunks)
        if status != "success" and rows:
            _fail("retrieval_failure_rows")
        return preparation._metric(case, rows, status, chunks, facts)
    return score


def _retrieval(case, artifact, physical_ok, status, score):
    results = {}
    generic_failure = artifact is not None and artifact.get("schema_version") == FAILURE_SCHEMA
    for arm in ("baseline", "enhanced"):
        part = artifact.get(arm) if artifact is not None and not generic_failure else None
        if type(part) is dict and set(part) >= {"rows", "status"}:
            valid = arm == "baseline" or physical_ok
            rows, outcome = (part["rows"], part["status"]) if valid else ([], "failed")
        else:
            rows, outcome = [], status
        row = score(case, rows, outcome)
        # Existing scorer validates exact source/chunk/version and complete
        # fact quotes. Only rank=1 is Hit@1; this is MRR restricted to Top3.
        row["document_hit_at_1"] = row["reciprocal_rank_at_3"] == 1
        results[arm] = row
    return results


def _group(rows):
    planned, reviewed = len(rows), [row for row in rows if "review" in row]
    tools = [row for row in rows if row.get("tool_task")]
    return {"planned": planned, "statuses": dict(Counter(row["status"] for row in rows)),
        "recorded": sum(row["status"] not in {"not_executed", "unknown"} for row in rows),
        "correct_handling": _rate(sum(row["status"] == "passed" for row in rows), planned),
        "semantic_reviews": len(reviewed),
        "dimensions": {dimension: _rate(sum(bool(row.get("execution_eligible_for_review")
            and row.get("review", {}).get("dimensions", {}).get(dimension)) for row in rows), planned)
            for dimension in DIMENSIONS},
        "tool_task_correct": _rate(sum(row.get("tool_task_correct") is True for row in tools), len(tools)),
        "tool_task_scope": "planned tool tasks; actual selection/arguments/identity/handler evidence, separate from answer coverage",
        "reading_burden": {**dict(Counter(row["review"]["reading_burden"] for row in reviewed)),
                           "unreviewed": planned - len(reviewed)},
        "measurement_complete": all(row["status"] not in {"unknown", "not_executed", "pending_review"} for row in rows)}


def _retrieval_group(rows, *, out_of_scope):
    n = len(rows)
    facts = sum(row["required_fact_count"] for row in rows)
    covered = sum(len(row["covered_fact_ids_at_3"]) for row in rows)
    return {"planned": n, "statuses": dict(Counter(row["status"] for row in rows)),
        "recorded": sum(row["status"] != "not_run" for row in rows),
        "measurement_complete": all(row["status"] not in {"not_run", "unknown"} for row in rows),
        "hit_at_1": None if out_of_scope else _rate(sum(row["document_hit_at_1"] for row in rows), n),
        "hit_at_3": None if out_of_scope else _rate(sum(row["document_hit_at_3"] for row in rows), n),
        "mrr_at_3": None if out_of_scope else _rate(sum(row["reciprocal_rank_at_3"] for row in rows), n),
        "all_required_facts_at_3": None if out_of_scope else _rate(sum(row["all_required_facts_at_3"] for row in rows), n),
        "micro_required_fact_recall_at_3": None if out_of_scope else _rate(covered, facts),
        "macro_required_fact_recall_at_3": None if out_of_scope else _rate(sum(row["required_fact_recall_at_3"] or 0 for row in rows), n),
        "healthy_empty_rejections": _rate(sum(row["healthy_empty_rejection"] for row in rows), n) if out_of_scope else None,
        "out_of_scope_false_hits": _rate(sum(row["out_of_scope_false_hit"] for row in rows), n) if out_of_scope else None,
        "returned_top3_characters": sum(row["top3_characters"] for row in rows),
        "interpretation": "planned denominator includes failures/unknown/not-run; a boundary document hit does not answer missing individual/live facts"}


def _pair_groups(protocol, rows):
    by_id = {row["case_id"]: row for row in rows}
    pairs = [{"case_id": case["id"], "split": case["split"],
              "A": by_id["A-" + case["id"]]["status"], "D": by_id["D-" + case["id"]]["status"]}
             for case in protocol["cases"].values() if case["id"].startswith("CT-")]
    groups = {}
    for split in ("development", "validation"):
        selected = [row for row in pairs if row["split"] == split]
        complete = [row for row in selected if row["A"] in {"passed", "failed"} and row["D"] in {"passed", "failed"}]
        groups[split] = {"planned_pairs": len(selected), "completed_pairs": len(complete),
            "both_passed": sum(row["A"] == row["D"] == "passed" for row in complete),
            "only_A_passed": sum(row["A"] == "passed" and row["D"] == "failed" for row in complete),
            "only_D_passed": sum(row["D"] == "passed" and row["A"] == "failed" for row in complete),
            "both_failed": sum(row["A"] == row["D"] == "failed" for row in complete),
            "incomplete_pairs": len(selected) - len(complete),
            "A": _rate(sum(row["A"] == "passed" for row in selected), len(selected)),
            "D": _rate(sum(row["D"] == "passed" for row in selected), len(selected)),
            "absolute_improvement": (sum(row["D"] == "passed" for row in selected)
                - sum(row["A"] == "passed" for row in selected)) / len(selected) if selected else None,
            "measurement_complete": len(complete) == len(selected)}
    return pairs, groups


def build_report(protocol, state, sealed, *, root, authorized_requests=480,
                 continuation=None, verify_source=lambda: None, source_root=bench.BACKEND_ROOT):
    """Recompute one immutable snapshot; caller retains and signs the result.

    ``continuation`` is public context, not an authorization. It may include the
    manifest identity, initial_attempts and an operator cleanup attestation. The
    root verifier/ledger are the authority for actual continuation permission.
    """
    verify_source()
    protocol = _protocol(protocol)
    if (type(state) is not dict or type(sealed) is not dict or type(sealed.get("cases")) is not dict
            or type(authorized_requests) is not int or not 0 <= authorized_requests <= MAX_REQUESTS
            or type(state.get("attempts")) is not int or not 0 <= state["attempts"] <= authorized_requests):
        _fail("snapshot")
    context = dict(continuation or {})
    initial = context.get("initial_attempts", 20)
    if type(initial) is not int or initial < 0:
        _fail("initial_attempts")
    root = Path(root).absolute()
    protocol_hash = bench.sha256(_path(root, "protocol.json"))
    planned = {spec["case_id"]: spec for spec in protocol["jobs"]}
    if not set(sealed["cases"]) <= set(planned):
        _fail("unknown_case")
    by_case, resources = _physical_operations(root, state, sealed, planned)
    rows, artifacts, retrieval = [], {}, {"baseline": [], "enhanced": []}
    score_retrieval = _retrieval_scorer(source_root)
    for spec in protocol["jobs"]:
        identity, case = spec["case_id"], protocol["cases"][spec["source_case_id"]]
        operations, artifact = by_case[identity], None
        item = {**spec, "tags": case["tags"], "tool_task": case["expected"]["tool_task"] if spec["suite"] == "tasks" else None,
            "boundary": case["expected"]["boundary"] if spec["suite"] == "tasks" else None,
            "status": "unknown" if state.get("active_case") == identity else "not_executed",
            "execution_segment": "original-prefix" if operations and all(op["attempt"] <= initial for op in operations)
                else "continuation" if operations else "not_executed"}
        if identity in sealed["cases"]:
            artifact = _read(root, f"case-{identity}.json", sealed["cases"][identity]["artifact_sha256"])
            artifacts[identity] = artifact
            physical_ok = bool(operations) and _physical_matches(artifact, operations) and all(
                op["outcome"] == "received" and op.get("output_sha256") for op in operations)
            if spec["suite"] == "retrieval":
                physical_ok = physical_ok and Counter(op["phase"] for op in operations) == {"rag_rewrite": 1, "rag_rerank": 1}
            item.update(execution_status=artifact.get("execution_status", artifact.get("status")),
                        physical_structure_valid=bool(physical_ok))
            if spec["suite"] in {"tasks", "conversations"}:
                observed = _native_checks(case, spec, artifact, source_root)
                item["checks"] = observed
                eligible = bool(physical_ok and _flags_pass(observed))
                flags = observed.get("checks", {})
                item["tool_task_correct"] = bool(item["tool_task"] and physical_ok
                    and all(flags.get(name) is True for name in TOOL_CHECKS))
            else:
                eligible = bool(physical_ok and artifact.get("status", artifact.get("execution_status")) == "success")
            item.update(execution_eligible_for_review=eligible,
                        status="recorded" if spec["suite"] == "retrieval" and eligible else "pending_review" if eligible else "failed")
            review_path = root / f"review-{identity}.json"
            if spec["suite"] != "retrieval" and review_path.exists():
                review = _read(root, review_path.name)
                validate_review(review, protocol, root=root, artifacts=artifacts)
                item.update(review=review, status="passed" if eligible and all(review["dimensions"].values())
                            and not review["blocking_errors"] else "failed")
            if artifact.get("continuation_failure") is not None:
                item["execution_failure"] = artifact["continuation_failure"]
        if spec["suite"] == "retrieval":
            fallback = "failed" if artifact is not None else "unknown" if item["status"] == "unknown" else "not_run"
            result = _retrieval(case, artifact, item.get("physical_structure_valid", False), fallback, score_retrieval)
            for arm in retrieval:
                retrieval[arm].append(result[arm])
        rows.append(item)
    groups = {f"{split}/{suite}/{arm}": _group(selected)
        for split in ("development", "validation") for suite in ("tasks", "conversations") for arm in ("A", "D", "current")
        if (selected := [row for row in rows if (row["split"], row["suite"], row["arm"]) == (split, suite, arm)])}
    retrieval_groups = {arm: {f"{split}/{kind}": _retrieval_group(selected, out_of_scope=kind == "out_of_scope")
        for split in ("development", "validation") for kind in ("answerable", "insufficient", "out_of_scope", "evidence")
        if (selected := [row for row in results if row["split"] == split and
                       (row["kind"] in {"answerable", "insufficient"} if kind == "evidence" else row["kind"] == kind)])}
        for arm, results in retrieval.items()}
    pairs, pair_groups = _pair_groups(protocol, rows)
    report = {"schema_version": SCHEMA, "protocol_sha256": protocol_hash, "continuation": context,
        "authorized_requests": authorized_requests, "reserved_requests": state["attempts"], "maximum_requests": MAX_REQUESTS,
        "remaining_requests": authorized_requests - state["attempts"], "planned_positions": 136, "planned_user_turns": 168,
        "completed_positions": len(sealed["cases"]), "pending_physical": len(state["pending"]), "paused": state["paused"],
        "position_statuses": dict(Counter(row["status"] for row in rows)), "rows": rows, "groups": groups,
        "task_pairs": pairs, "paired_groups": pair_groups, "retrieval": retrieval, "retrieval_groups": retrieval_groups,
        "physical_resources": metrics.aggregate_resources(resources), "quality_pass_claim": False,
        "semantic_reviews": sum("review" in row for row in rows),
        "intent_classification": {"status": "not_measured", "historical_scores_changed": False},
        "execution_complete": len(sealed["cases"]) == 136 and not state["pending"] and not state["paused"]
                              and state.get("active_case") is None,
        "interpretation": "original and continuation execution segments retained; business candidate unchanged; no automatic semantic or production acceptance"}
    verify_source()
    return report
