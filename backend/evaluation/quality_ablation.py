"""四组消融的离线设计、清单核对与受控计分，不执行真实对话。

在 backend/ 目录运行：
  .venv/bin/python -B -m evaluation.quality_ablation prepare
  .venv/bin/python -B -m evaluation.quality_ablation verify
  .venv/bin/python -B -m evaluation.quality_ablation template
  .venv/bin/python -B -m evaluation.quality_ablation summary
  .venv/bin/python -B -m evaluation.quality_ablation summary --results <受控结果JSON>

prepare 只新建私有设计目录，已有目录不覆盖；template 生成全部未执行的
受控记录。summary 只检验计分规则，不产生模型质量成绩。没有 run 或授权命令。
当前四臂没有接入运行时；设计/题目冻结不等于提示、完整payload或真实执行已冻结。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


BACKEND = Path(__file__).resolve().parents[1]
MANIFEST = BACKEND / "evaluation/benchmark/quality-ablation-v1.json"
SOURCE_FILES = ("evaluation/benchmark/task-e2e-v1.json", "evaluation/benchmark/tasks-v1.json")
ROOT = BACKEND / "data/benchmarks/quality-ablation-design-v1"
ARMS = ("A", "B", "C", "D")
CASE_IDS = tuple([f"e{i:02d}" for i in range(1, 7)] + [f"t{i:02d}" for i in range(7, 13)])
SUPPORTED = frozenset({"e01", "e02", "e03", "e04", "e05", "t07", "t08", "t11"})
BOUNDARY = frozenset({"e06", "t09", "t10", "t12"})
TOOLS = frozenset(set(CASE_IDS) - {"t10", "t12"})
DIMENSIONS = ("evidence_relevance", "coverage", "conditions_preserved", "missing_correct",
              "status_correct", "citations_correct", "tool_permissions")
MECHANISMS = ("fresh_read_each_relevant_turn", "exact_projection")
OUTCOMES = ("completed", "failed", "unknown", "not_executed")
MAX_JSON = 262_144


class QualityAblationError(ValueError):
    """Fixed local diagnostic code; no input body or credential is echoed."""


def _fail(code):
    raise QualityAblationError(code)


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        _fail("invalid_json")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                _fail("duplicate_json_key")
            value[key] = child
        return value
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            _fail("json_size_limit")
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: _fail("nonfinite_json"))
        _canonical(value)
        return value
    except QualityAblationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        _fail("invalid_json")


def _exact(value, fields, code="invalid_fields"):
    if type(value) is not dict or set(value) != set(fields):
        _fail(code)


def _source_cases():
    first, second = (_read(BACKEND / name) for name in SOURCE_FILES)
    values = first.get("cases") if type(first) is dict else None
    additions = second.get("cases") if type(second) is dict else None
    if (type(values) is not list or type(additions) is not list
            or any(type(c) is not dict for c in values + additions)
            or tuple(c.get("id") for c in values) != CASE_IDS[:6]):
        _fail("original_cases_changed")
    selected = [c for c in additions if c.get("split") == "validation"]
    if tuple(c.get("id") for c in selected) != CASE_IDS[6:]:
        _fail("original_cases_changed")
    result = []
    for index, value in enumerate(values + selected):
        turns = value.get("turns") if index < 6 else [value.get("input", {}).get("message")]
        if (type(turns) is not list or not turns or len(turns) > 2
                or any(type(t) is not str or not 1 <= len(t) <= 2000 for t in turns)
                or type(value.get("expected")) is not dict or type(value.get("references")) is not list):
            _fail("original_cases_changed")
        identity = value["id"]
        customer = value.get("customer_id") if index < 6 else value.get("input", {}).get("customer_id")
        if customer not in {"demo-customer-a", "demo-customer-b"}:
            _fail("original_customer_scope_changed")
        result.append({"case_id": identity, "source_file": SOURCE_FILES[0 if index < 6 else 1],
            "source_case_id": value["id"], "turns": deepcopy(turns), "trusted_customer_id": customer,
            "question_sha256": _hash({"turns": turns}), "expected_sha256": _hash(value["expected"]),
            "references_sha256": _hash({"references": value["references"]}),
            "supported_completion": identity in SUPPORTED, "boundary_handling": identity in BOUNDARY,
            "tool_task": identity in TOOLS,
            "boundary_kind": {"e06": "authorization", "t09": "read_plus_unsupported_refund",
                              "t10": "handoff_preparation_only", "t12": "security_handoff_preparation_only"}.get(identity)})
    if sum(len(c["turns"]) for c in result) != 13:
        _fail("original_cases_changed")
    return result


def expected_manifest():
    return {
        "schema_version": "serveflow-quality-ablation-design-v1",
        "runtime_arms_ready": False,
        "authorized_model_requests": 0,
        "authorized_business_requests": 0,
        "live_payloads_frozen": False,
        "purpose": "known-case diagnostic ablation design; not live execution or model quality evidence",
        "common_base": "one shared current foundation; not an untouched historical EchoMind/Git baseline",
        "arms": {
            "A": {"force_fresh_read": False, "typed_order_projection": False},
            "B": {"force_fresh_read": True, "typed_order_projection": False},
            "C": {"force_fresh_read": False, "typed_order_projection": True},
            "D": {"force_fresh_read": True, "typed_order_projection": True},
        },
        "common_invariants": ["server_bound_identity", "readonly_tools", "request_target_binding", "strict_schema",
            "role_tool_scope", "cancellation_and_versions", "policy_source_projection", "same_skills_and_routing",
            "same_model_data_retrieval_parameters_budget_and_scoring", "isolated_conversations_and_tool_caches_per_case_arm"],
        "arm_semantics": {
            "force_off": "does not prohibit a model's voluntary fresh read or deliberately weaken the baseline",
            "projection_off": "natural summaries may pass if directly supported, complete and source-faithful",
            "projection_on": "only actual validated order evidence may be projected; absent evidence remains partial",
            "not_runtime_input": "arm names, labels and gold never enter model inputs or frontend request flags",
        },
        "primary_gate": {"requires": ["execution_complete", "actual_source_support", "all_seven_review_dimensions",
                                      "no_blocking_error"],
                         "never_requires": ["verbatim_program_projection", "using_the_enhanced_mechanism_itself"],
                         "dimensions": list(DIMENSIONS),
                         "freshness": "static fixtures establish tool trajectory, not changing real-world record freshness",
                         "refusal": "correct refusal is boundary handling, not completion of the requested refund or human connection"},
        "denominators": {"correct_handling": 12, "supported_completion": 8, "tool_tasks": 10,
                         "boundary_handling": 4},
        "mechanism_checks": {"fields": list(MECHANISMS), "part_of_primary_gate": False},
        "cases": _source_cases(),
        "order": [{"case_id": identity, "arm_order": list(ARMS[index % 4:] + ARMS[:index % 4])}
                  for index, identity in enumerate(CASE_IDS)],
        "limitations": ["all existing questions/labels are author-visible and previously used; no independent blind or generalization claim",
            "old paused trials, ledgers, unknown sends and scores remain separate and cannot be resumed or rescored",
            "runtime wiring, full payloads, phase/admission deadlines, durable execution ledger and explicit applicable allowance need a later execution freeze",
            "same fixed deadlines and admission accounting must apply to every arm before true sends",
            "each future planned case/arm executes once; no retries or selection of best outputs",
            "new development and held-out validation samples have not been defined in this design",
            "controlled summary tests demonstrate accounting only, not semantic quality improvement"],
    }


def load_manifest():
    value = _read(MANIFEST)
    if _canonical(value) != _canonical(expected_manifest()):
        _fail("design_or_original_cases_changed")
    return value


def _write(root, name, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    if len(raw) > MAX_JSON:
        _fail("json_size_limit")
    try:
        fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # Preserve a partial file; no cleanup/retry may disguise its state.
            raise
    except FileExistsError:
        _fail("output_already_exists")


class QualityAblation:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root or self.root.is_symlink() or any(p.is_symlink() for p in self.root.parents):
            _fail("unsafe_design_root")
        historical = BACKEND / "data/benchmarks"
        if self.root.is_relative_to(historical) and self.root != ROOT:
            _fail("fixed_design_root_required")

    def _artifacts(self):
        names = (*SOURCE_FILES, "evaluation/quality_ablation.py", "evaluation/benchmark/quality-ablation-v1.json",
                 "tests/test_quality_ablation.py")
        return {name: _file_hash(BACKEND / name) for name in names}

    def prepare(self):
        design = load_manifest()
        if self.root.exists():
            _fail("design_root_already_exists")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND, capture_output=True,
                              text=True, check=True).stdout.strip()
        value = {"schema_version": "serveflow-quality-design-freeze-v1", "prepared_at": datetime.now(timezone.utc).isoformat(),
                 "git_head": head, "root_sha256": _hash({"path": str(self.root)}), "artifacts": self._artifacts(),
                 "manifest_sha256": _file_hash(MANIFEST), "design_sha256": _hash(design),
                 "runtime_arms_ready": False, "live_payloads_frozen": False,
                 "authorized_model_requests": 0, "actual_model_requests": 0,
                 "source_scope": "design and original questions only; no claim of final runtime/model payload freeze"}
        self.root.mkdir(mode=0o700, parents=True)
        _write(self.root, "design-freeze.json", value)
        return value

    def verify(self):
        design = load_manifest()
        value = _read(self.root / "design-freeze.json")
        _exact(value, {"schema_version", "prepared_at", "git_head", "root_sha256", "artifacts", "manifest_sha256",
                       "design_sha256", "runtime_arms_ready", "live_payloads_frozen", "authorized_model_requests",
                       "actual_model_requests", "source_scope"}, "freeze_fields")
        if (value["schema_version"] != "serveflow-quality-design-freeze-v1" or value["artifacts"] != self._artifacts()
                or value["root_sha256"] != _hash({"path": str(self.root)})
                or value["manifest_sha256"] != _file_hash(MANIFEST) or value["design_sha256"] != _hash(design)
                or value["runtime_arms_ready"] is not False or value["live_payloads_frozen"] is not False
                or type(value["authorized_model_requests"]) is not int or value["authorized_model_requests"] != 0
                or type(value["actual_model_requests"]) is not int or value["actual_model_requests"] != 0
                or value["source_scope"] != "design and original questions only; no claim of final runtime/model payload freeze"):
            _fail("frozen_design_changed")
        return value

    def template(self):
        self.verify()
        return {"schema_version": "serveflow-controlled-ablation-results-v1", "evidence_kind": "controlled-fixture",
            "design_sha256": _hash(load_manifest()), "records": [blank_record(arm, identity)
                for identity in CASE_IDS for arm in ARMS]}

    def summary(self, results=None):
        self.verify()
        if results is None:
            results = self.template()
        return summarize(results, load_manifest())


def blank_record(arm, identity):
    return {"arm": arm, "case_id": identity, "outcome": "not_executed", "execution_complete": False,
        "source_supported": None, "dimensions": dict.fromkeys(DIMENSIONS), "blocking_errors": [],
        "supported_task_complete": None, "tool_task_correct": None, "boundary_handled_correctly": None,
        "mechanism_checks": dict.fromkeys(MECHANISMS)}


def _record(value, case):
    _exact(value, blank_record("A", case["case_id"]), "record_fields")
    if (type(value["arm"]) is not str or value["arm"] not in ARMS or value["case_id"] != case["case_id"]
            or type(value["outcome"]) is not str or value["outcome"] not in OUTCOMES
            or type(value["execution_complete"]) is not bool):
        _fail("record_identity_or_status")
    _exact(value["dimensions"], DIMENSIONS, "review_dimensions")
    _exact(value["mechanism_checks"], MECHANISMS, "mechanism_fields")
    judgements = [value["source_supported"], *value["dimensions"].values(), *value["mechanism_checks"].values(),
                  value["supported_task_complete"], value["tool_task_correct"], value["boundary_handled_correctly"]]
    if any(v is not None and type(v) is not bool for v in judgements):
        _fail("judgement_boolean_required")
    if (type(value["blocking_errors"]) is not list or len(value["blocking_errors"]) > 20
            or any(type(v) is not str or not 1 <= len(v) <= 160 or any(ord(c) < 32 for c in v)
                   for v in value["blocking_errors"])):
        _fail("blocking_errors_invalid")
    for flag, field in (("supported_completion", "supported_task_complete"), ("tool_task", "tool_task_correct"),
                        ("boundary_handling", "boundary_handled_correctly")):
        if not case[flag] and value[field] is not None:
            _fail("judgement_outside_metric_scope")
    if value["outcome"] in {"unknown", "not_executed"}:
        if value["execution_complete"] or any(v is not None for v in judgements) or value["blocking_errors"]:
            _fail("unexecuted_cannot_have_a_score")
    elif value["outcome"] == "completed":
        required = [value["source_supported"], *value["dimensions"].values()]
        required += [value[field] for flag, field in (("supported_completion", "supported_task_complete"),
            ("tool_task", "tool_task_correct"), ("boundary_handling", "boundary_handled_correctly")) if case[flag]]
        if not value["execution_complete"] or any(v is None for v in required):
            _fail("completed_record_requires_review")
    elif value["execution_complete"]:
        _fail("failed_execution_cannot_be_complete")
    return deepcopy(value)


def _metric(items, passed):
    statuses = []
    for item in items:
        outcome = item["outcome"]
        statuses.append("passed" if outcome == "completed" and passed(item) else
                        "failed" if outcome in {"failed", "completed"} else outcome)
    counts = {state: statuses.count(state) for state in ("passed", "failed", "unknown", "not_executed")}
    return {"numerator": counts["passed"], "denominator": len(items),
            "value": counts["passed"] / len(items) if items else None, "counts": counts}


def _handled(record):
    return (record["execution_complete"] and record["source_supported"] is True
            and all(v is True for v in record["dimensions"].values()) and not record["blocking_errors"]
            and record["supported_task_complete"] is not False and record["boundary_handled_correctly"] is not False)


def summarize(results, design=None):
    design = load_manifest() if design is None else design
    if _canonical(design) != _canonical(expected_manifest()):
        _fail("design_or_original_cases_changed")
    _exact(results, {"schema_version", "evidence_kind", "design_sha256", "records"}, "result_fields")
    if (results["schema_version"] != "serveflow-controlled-ablation-results-v1"
            or results["evidence_kind"] != "controlled-fixture" or results["design_sha256"] != _hash(design)
            or type(results["records"]) is not list or len(results["records"]) > 48):
        _fail("controlled_results_only")
    cases = {c["case_id"]: c for c in design["cases"]}
    observed = {}
    for item in results["records"]:
        if type(item) is not dict or type(item.get("case_id")) is not str or item["case_id"] not in cases:
            _fail("unknown_case")
        record = _record(item, cases[item["case_id"]])
        key = (record["arm"], record["case_id"])
        if key in observed:
            _fail("duplicate_arm_case")
        observed[key] = record
    completed = {(arm, identity): observed.get((arm, identity), blank_record(arm, identity))
                 for arm in ARMS for identity in CASE_IDS}
    arm_reports = {}
    for arm in ARMS:
        items = [completed[arm, key] for key in CASE_IDS]
        arm_reports[arm] = {
            "correct_handling": _metric(items, _handled),
            "supported_completion": _metric([r for r in items if cases[r["case_id"]]["supported_completion"]],
                lambda r: _handled(r) and r["supported_task_complete"] is True),
            "tool_tasks": _metric([r for r in items if cases[r["case_id"]]["tool_task"]],
                lambda r: r["execution_complete"] and r["tool_task_correct"] is True
                    and r["dimensions"]["tool_permissions"] is True),
            "boundary_handling": _metric([r for r in items if cases[r["case_id"]]["boundary_handling"]],
                lambda r: _handled(r) and r["boundary_handled_correctly"] is True),
            "mechanism_observations": {field: {"true": sum(r["mechanism_checks"][field] is True for r in items),
                "false": sum(r["mechanism_checks"][field] is False for r in items),
                "unknown": sum(r["mechanism_checks"][field] is None for r in items)} for field in MECHANISMS},
        }
    pairs = {}
    for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D")):
        counts = dict(both_passed=0, only_left_passed=0, only_right_passed=0, neither_passed=0, unpaired_unknown_or_unexecuted=0)
        for identity in CASE_IDS:
            a, b = completed[left, identity], completed[right, identity]
            if any(r["outcome"] in {"unknown", "not_executed"} for r in (a, b)):
                counts["unpaired_unknown_or_unexecuted"] += 1
                continue
            ap, bp = a["outcome"] == "completed" and _handled(a), b["outcome"] == "completed" and _handled(b)
            counts["both_passed" if ap and bp else "only_left_passed" if ap else "only_right_passed" if bp else "neither_passed"] += 1
        complete = counts["unpaired_unknown_or_unexecuted"] == 0
        pairs[left + "_vs_" + right] = {"planned_denominator": 12, **counts,
            "paired_terminal_count": 12 - counts["unpaired_unknown_or_unexecuted"],
            "full_plan_delta_percentage_points": (100 * (arm_reports[right]["correct_handling"]["value"]
                                                        - arm_reports[left]["correct_handling"]["value"])) if complete else None}
    return {"schema_version": "serveflow-controlled-ablation-summary-v1", "evidence_kind": "controlled-fixture",
        "runtime_arms_ready": False, "model_quality_evidence": False, "actual_model_requests": 0,
        "authorized_model_requests": 0, "planned_arm_cases": 48, "planned_arm_turns": 52,
        "denominators_per_arm": deepcopy(design["denominators"]), "arms": arm_reports, "pairs": pairs,
        "rate_interpretation": "confirmed passes divided by the full plan; unknown and unexecuted remain explicit, not an estimated success probability",
        "claim_limit": "controlled scoring checks only; no real accuracy, improvement or complete live payload freeze"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "template", "summary"))
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    try:
        tool = QualityAblation()
        if args.operation == "prepare":
            value = tool.prepare()
        elif args.operation == "verify":
            tool.verify()
            value = {"verified": True, "runtime_arms_ready": False, "actual_model_requests": 0}
        elif args.operation == "template":
            value = tool.template()
            _write(tool.root, "controlled-template.json", value)
        else:
            value = tool.summary(_read(args.results) if args.results else None)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    except (QualityAblationError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"error_code": str(error) if isinstance(error, QualityAblationError) else "local_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
