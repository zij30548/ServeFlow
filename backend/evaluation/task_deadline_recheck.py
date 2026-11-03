"""New fixed regression after separating audit and model-stage clocks.

Preparation/verification grant zero sends. This is not an A/B quality uplift
experiment: the twelve existing authored cases and rubric are unchanged. The
separate quality_ablation design cannot send or enable runtime experiment arms.
No old paused operation, score or unused allowance is recovered here.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import subprocess

from core.model_deadline import EvaluationDeadlinePolicy, eval_deadlines
from evaluation import benchmark as bench, intent_trial as base
from evaluation import task_business_regression as business
from evaluation.intent_transport import canonical_payload_hash
from evaluation.task_recheck import _tree, history


ROOT = bench.BACKEND_ROOT / "data/benchmarks/task-business-v6-stage-clock"
BATCH = "serveflow-task-business-v6-stage-clock"
CANDIDATE = "same-business-evidence-separate-active-and-audit-deadlines"
V4 = {
    "root": business.ROOT,
    "commit": "81e6b590e41260ebca8de041975c51722e711be9",
    "protocol": "65d3f5580d2fc2761b92a8a0cd4f1718a49dfa5b8d42eae4a93122ff00817728",
    "authorization": "41d853d3d9d3c254b2c6602dcacad437f2e4de27d2d00c7ae39925aca4441baf",
    "journal": "fd041e71e443c21b967d7e499231cf98946e086829335c9e4a3e8e5f7e33688a",
    "chain": "ab60ebbf92b08bbb47e7b2af79af99e5c10ac1d03c726165f77099039b77932f",
    "tree": "79c27975158d84d77fefcd15c0ae06108de67f8f2383941089fee810b06ac2e0",
    "attempts": 12, "active": "e03", "completed": ["e01", "e02"],
    "business_counts": {"passed": 2, "failed": 0, "evaluation_failed": 0,
                        "pending_review": 0, "unknown": 1, "not_executed": 9},
}
V5 = {
    "root": bench.BACKEND_ROOT / "data/benchmarks/task-business-v5-observation",
    "commit": "53f4ff543c820c2e1ed3fa80db5f461ae9a9cddc",
    "protocol": "7be51b9c3e825286b47d4fa69aafd10cc30e44bb9186bde2f9f9465a01c7b36f",
    "authorization": "93d9d9918d30f2b15ddc1ff8187323254d54d560fedb0e757e0715c32568b103",
    "journal": "337bd4adec07d90cf5213d1455273af54edbc806b40a5517783a07d62d823a27",
    "chain": "637e741458ad04133242f37d816cf7b27d2d86c19fadb16b76adddc64497c9ff",
    "tree": "95731509f1285e7c220343ce1ca3640e2dd4c300a862ef98b43af21d403db0fa",
    "attempts": 25, "active": "e06", "completed": [f"e{i:02d}" for i in range(1, 6)],
    "business_counts": {"passed": 5, "failed": 0, "evaluation_failed": 0,
                        "pending_review": 0, "unknown": 1, "not_executed": 6},
}
CLOCK_POLICY = {
    "version": "model-stage-active-clock-v1",
    "model_stage_budgets": {"intent": 8, "agent": 15, "composer": 8,
                            "rag_rewrite": 10, "rag_rerank": 30, "judge": 8},
    "network_deadline_s": 8,
    "excluded_scopes": ["queue", "admission", "seal"],
    "per_call_cumulative_audit_limit_s": 30,
    "per_call_wall_limit": "original stage seconds + 30 seconds + 0.25 seconds cleanup grace; never paused",
    "scope": "only trusted evaluation transport management; never model/network/response parsing",
    "production": "no evaluation opt-in uses original asyncio.wait_for total-time semantics and values",
    "cancellation": "external cancellation stays cancellation; already-sent unknown outcomes remain occupied; no retries",
    "cleanup": "cooperative timers request cancellation; uncooperative threads/cleanup require the unchanged process supervisor",
    "comparison": "a new execution contract; do not merge paused results or claim quality uplift from clock repair",
}


def _production(artifacts):
    return {name: digest for name, digest in artifacts.items()
            if not name.startswith(("evaluation/", "tests/"))}


def _archived(spec):
    """Bind old bytes to Git and their original ledger, never rescore them."""
    root = spec["root"]
    if _tree(root) != spec["tree"]:
        base._fail("task_clock_old_tree_changed")
    for name, digest in {"protocol.json": spec["protocol"], "authorization.json": spec["authorization"],
                         "ledger/journal.jsonl": spec["journal"]}.items():
        if bench.sha256(base._path(root, name)) != digest:
            base._fail("task_clock_old_record_changed")
    p = bench.read_json(base._path(root, "protocol.json"))
    if p["git_head"] != spec["commit"] or p["root_binding_sha256"] != base._root_hash(root):
        base._fail("task_clock_old_identity_changed")
    history._archived_sources(p, "artifacts")
    ledger = business.BusinessTaskBudget(root / "ledger", spec["protocol"], p["plan"])
    state, sealed = ledger.status(), ledger.results()
    if (state["attempts"] != spec["attempts"] or state["chain_hash"] != spec["chain"]
            or state["pending"] or state["paused"] is not True or state["active_case"] != spec["active"]
            or state["completed_cases"] != spec["completed"] or len(sealed["operations"]) != spec["attempts"]
            or set(sealed["cases"]) != set(spec["completed"])):
        base._fail("task_clock_old_ledger_changed")
    a = bench.read_json(base._path(root, "authorization.json"))
    if (a["protocol_sha256"] != spec["protocol"] or a["max_requests"] != business.MAX_REQUESTS
            or a["plan"] != p["plan"] or a["root_binding_sha256"] != base._root_hash(root)
            or a["initial_chain_hash"] != ledger.initial_chain_hash):
        base._fail("task_clock_old_authorization_changed")
    for op in sealed["operations"].values():
        if bench.sha256(base._path(root, f"physical-{op['attempt']:03d}.json")) != op["output_sha256"]:
            base._fail("task_clock_old_physical_changed")
    for key, item in sealed["cases"].items():
        if bench.sha256(base._path(root, f"case-{key}.json")) != item["artifact_sha256"]:
            base._fail("task_clock_old_case_changed")
    report = bench.read_json(base._path(root, "reviewed-paused-report.json"))
    if report["business_counts"] != spec["business_counts"] or _tree(root) != spec["tree"]:
        base._fail("task_clock_old_summary_changed")
    return p, {"source_commit": spec["commit"], "tree_sha256": spec["tree"],
        "protocol_sha256": spec["protocol"], "authorization_sha256": spec["authorization"],
        "journal_sha256": spec["journal"], "chain_hash": spec["chain"],
        "reserved_requests": spec["attempts"], "sealed_requests": len(sealed["operations"]),
        "original_business_counts": deepcopy(spec["business_counts"]), "old_scores_recomputed": False}


class _NormalizeClock(ast.NodeTransformer):
    """Allow only a wrapper/import substitution, preserving all call payloads."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "await_model":
            node.func = ast.Attribute(value=ast.Name(id="asyncio", ctx=ast.Load()), attr="wait_for", ctx=ast.Load())
            for keyword in node.keywords:
                if keyword.arg == "timeout_s":
                    keyword.arg = "timeout"
        return node


def _normalized(source):
    tree = ast.parse(source)
    # Only the exact, top-level additive import is instrumentation. An alias
    # or a nested import changes binding/scope and must remain in the AST.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "core.model_deadline"
        and len(node.names) == 1 and node.names[0].name == "await_model" and node.names[0].asname is None
    )]
    return ast.dump(_NormalizeClock().visit(tree), include_attributes=False)


def business_clock_delta(old_protocol):
    before, now = _production(old_protocol["artifacts"]), _production(business._artifacts())
    allowed = {"core/intent_recognizer.py", "agents/agent_orchestrator.py", "tooling/tool_manager.py"}
    expected_calls = {"core/intent_recognizer.py": 1, "agents/agent_orchestrator.py": 2,
                      "tooling/tool_manager.py": 2}
    if set(now) - set(before) != {"core/model_deadline.py"} or set(before) - set(now):
        base._fail("task_clock_business_file_set_changed")
    changed = []
    for name, digest in before.items():
        if now[name] == digest:
            continue
        if name not in allowed:
            base._fail("task_clock_non_clock_business_change")
        old = subprocess.run(["git", "show", f"{old_protocol['git_head']}:backend/{name}"],
            cwd=bench.BACKEND_ROOT, capture_output=True, check=True).stdout
        current = (bench.BACKEND_ROOT / name).read_bytes()
        tree = ast.parse(current)
        exact_imports = sum(isinstance(node, ast.ImportFrom) and node.level == 0
            and node.module == "core.model_deadline" and len(node.names) == 1
            and node.names[0].name == "await_model" and node.names[0].asname is None for node in tree.body)
        calls = sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "await_model" for node in ast.walk(tree))
        if exact_imports != 1 or calls != expected_calls[name] or _normalized(old) != _normalized(current):
            base._fail("task_clock_payload_or_logic_changed")
        changed.append(name)
    return {"old_business_source_sha256": canonical_payload_hash({"artifacts": before}),
            "current_business_source_sha256": canonical_payload_hash({"artifacts": now}),
            "clock_wrapper_only_changes": sorted(changed), "prompts_and_domain_logic_ast_unchanged": True}


def prior_proof():
    older = business.prior_proof()
    p4, v4 = _archived(V4)
    p5, v5 = _archived(V5)
    if (p4["prior_proof"] != older or p5["prior_proof"]["tree_sha256"] != V4["tree"]
            or p5["prior_proof"]["older_proof_sha256"] != canonical_payload_hash(older)
            or _production(p4["artifacts"]) != _production(p5["artifacts"])):
        base._fail("task_clock_old_history_changed")
    return {"schema_version": "serveflow-stage-clock-history-v1", "v4": v4, "v5": v5,
            "older_proof_sha256": canonical_payload_hash(older), "old_allowance_reused": False,
            "new_authorized_requests": 0, **business_clock_delta(p5)}


def _artifacts():
    additions = ("evaluation/task_deadline_recheck.py", "evaluation/task_deadline_transport.py",
                 "core/model_deadline.py", "tests/test_model_deadline.py",
                 "tests/test_task_deadline_transport.py", "tests/test_task_deadline_recheck.py")
    return {**business._artifacts(), **{name: bench.sha256(bench.BACKEND_ROOT / name) for name in additions}}


class DeadlineRecheck(business.BusinessRegression):
    root_path, batch, candidate = ROOT, BATCH, CANDIDATE
    protocol_schema = "serveflow-task-business-protocol-v6-stage-clock"
    authorization_schema = "serveflow-task-business-authorization-v6-stage-clock"
    report_schema = "serveflow-task-business-report-v6-stage-clock"

    def artifact_proof(self):
        return _artifacts()

    def historical_proof(self):
        return prior_proof()

    def execution_policy(self):
        return {**deepcopy(business.POLICY), "stage_clock": deepcopy(CLOCK_POLICY)}

    def make_transport(self, *args, **kwargs):
        from evaluation.task_deadline_transport import DeadlineTaskHTTPTransport
        return DeadlineTaskHTTPTransport(*args, **kwargs)

    async def run_case(self, case, sdk, data, *, ensure_healthy):
        with eval_deadlines(EvaluationDeadlinePolicy(overhead_limit_s=30, cleanup_grace_s=.25)):
            return await super().run_case(case, sdk, data, ensure_healthy=ensure_healthy)

    def report(self):
        value = super().report()
        return {**value, "stage_clock_policy": deepcopy(CLOCK_POLICY),
                "is_ablation_uplift_report": False,
                "historical_scores_rewritten": False}


def main(argv=None):
    return business._main(DeadlineRecheck, "evaluation.task_deadline_recheck", argv)


if __name__ == "__main__":
    raise SystemExit(main())
