"""Offline freeze / separately authorized execution of the read-only repair.

No CLI path, model or worker override. Original question, old execution and
repair records are immutable inputs; preparation grants zero real requests.
"""
from copy import deepcopy

from evaluation import prospective_trial as original
from evaluation import prospective_repaired_proof as proof


GROUPS = {
    "new_author_visible_tasks": tuple(f"p{i:02d}" for i in range(1, 12)),
    "seen_scope_regression": ("p12",),
}
GROUP_NOTES = {
    "new_author_visible_tasks": "Not used in this scope repair; authored shared-source questions, some controlled wiring already seen; not blind or population validation.",
    "seen_scope_regression": "p12 was used for development and controlled regression; never an unseen validation claim.",
}
STATES = ("passed", "failed", "pending_review", "unknown", "not_executed")


def groups():
    return {name: {"case_ids": list(ids), "scope": GROUP_NOTES[name]} for name, ids in GROUPS.items()}


def grouped_tasks(report):
    """Additional fixed views; never remove a case or replace whole-plan gates."""
    rows = [row for row in report["cases"] if not row["logical_case_id"].startswith("R-")]
    expected = {f"{arm}-p{i:02d}" for arm in ("A", "D") for i in range(1, 13)}
    if len(rows) != 24 or {row["logical_case_id"] for row in rows} != expected:
        original.base._fail("repaired_group_plan_changed")
    cases = {c["id"]: c for c in original.design._cases(original.bench.BACKEND_ROOT, "tasks")["cases"]}
    output = {}
    for name, ids in GROUPS.items():
        selected = {row["logical_case_id"]: row for row in rows if row["logical_case_id"][2:] in ids}
        metrics = {}
        for arm in ("A", "D"):
            arm_rows = [selected[arm+"-"+key] for key in ids]
            def metric(predicate, status_key="status"):
                subset = [row for row in arm_rows if predicate(cases[row["logical_case_id"][2:]]["expected"])]
                counts = {state: sum(row.get(status_key, row["status"]) == state for row in subset) for state in STATES}
                return {"numerator": counts["passed"], "denominator": len(subset),
                        "applicable": bool(subset), "counts": counts}
            metrics[arm] = {
                "correct_handling": metric(lambda _: True),
                "supported_completion": metric(lambda e: not e["boundary"]),
                "tool_tasks": metric(lambda e: e["tool_task"], "tool_status"),
                "boundary_handling": metric(lambda e: e["boundary"]),
                "dimensions": {dim: {
                    "passed_reviewed": sum(row.get("review", {}).get("dimensions", {}).get(dim) is True for row in arm_rows),
                    "reviewed": sum("review" in row for row in arm_rows), "planned": len(arm_rows)
                } for dim in original.design.DIMENSIONS},
            }
        pairs = [{"case_id": key, **{arm: selected[arm+"-"+key]["status"] for arm in ("A", "D")}} for key in ids]
        counts = {"both_passed": 0, "only_A_passed": 0, "only_D_passed": 0,
                  "both_failed": 0, "unpaired_pending_unknown_or_unexecuted": 0}
        for row in pairs:
            if any(row[arm] not in {"passed", "failed"} for arm in ("A", "D")):
                counts["unpaired_pending_unknown_or_unexecuted"] += 1
            elif row["A"] == row["D"]:
                counts["both_passed" if row["A"] == "passed" else "both_failed"] += 1
            else:
                counts["only_A_passed" if row["A"] == "passed" else "only_D_passed"] += 1
        output[name] = {**groups()[name], "task_metrics": metrics, "pairs": pairs, "pair_counts": counts,
            "delta_percentage_points": ((counts["only_D_passed"]-counts["only_A_passed"])*100/len(ids)
                if not counts["unpaired_pending_unknown_or_unexecuted"] else None)}
    return output


class RepairedProspectiveTrial(original.ProspectiveTrial):
    root_path = proof.ROOT
    protocol_schema = "serveflow-prospective-execution-protocol-v2-readonly-scope"
    authorization_schema = "serveflow-prospective-authorization-v2-readonly-scope"
    progress_schema = "serveflow-prospective-progress-v2-readonly-scope"

    def source_proof(self):
        return proof.repaired_design_proof()

    def artifact_proof(self):
        return proof.artifacts()

    def history_proof(self):
        return proof.prior_ledgers()

    def execution_policy(self):
        policy = deepcopy(original.POLICY)
        policy["scope"] = "Same original 36 jobs after a separately bound common safety repair; p12 is seen regression, p01-p11 not used for this repair."
        policy["safety_repair"] = "Both A/D share read-only mutation-scope protection. No unsafe baseline restoration, no before/after splice, no prompt/retrieval change."
        policy["group_reporting"] = "Fixed groups supplement, never replace, all 12 task pairs and 12 retrieval pairs; failures/unknowns/unexecuted remain in every relevant denominator."
        return policy

    def protocol_extension(self):
        return {"evaluation_groups": groups(),
            "retrieval_group": {"case_ids": [f"q{i:02d}" for i in range(1, 13)],
                "scope": "Original retrieval questions and algorithm; separate from task quality, shared known sources."}}

    def report(self):
        report = super().report()
        report["task_groups"] = grouped_tasks(report)
        report["group_scope"] = "Supplementary fixed views; whole-plan metrics/readiness unchanged. No unseen/blind/population or semantic pass claim from wiring."
        return report


if __name__ == "__main__":
    original.main(RepairedProspectiveTrial)
