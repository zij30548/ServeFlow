"""Pure, offline metrics for one planned split and one benchmark arm.

The functions consume already selected cases and already grouped records. They
never load a manifest, inspect the environment, execute a case, or infer a rubric
review from answer text. Missing records stay in every planned-case denominator.
All rates use ``{numerator, denominator, value}``; a zero population has a null
value, except per-label classification metrics whose declared zero-division
convention is zero. Resource populations contain recorded values only.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


STATUSES = ("success", "failed", "cancelled", "unknown")
USAGE_FIELDS = ("input_tokens", "output_tokens", "physical_requests")
REVIEW_DIMENSIONS = (
    "evidence_relevance",
    "coverage",
    "conditions_preserved",
    "missing_correct",
    "status_correct",
    "citations_correct",
    "tool_permissions",
)
REVIEW_METHODS = (
    "executor-rubric-review",
    "independent-human-review",
    "controlled-fixture",
)
_COMMON_FIELDS = {"case_id", "status", "latency_ms", "usage"}
_ARM_FIELDS = {
    "intent": {"predicted_intent"},
    "retrieval": {"ranked_titles"},
    "tasks": {"answer", "review", "execution"},
}
_REVIEW_FIELDS = {
    "reviewer", "method", "evidence_notes", "dimensions",
    "blocking_errors", "reading_burden",
}


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _string(value: Any, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}string")
    return value


def _strings(value: Any, name: str, *, unique: bool = False, nonempty: bool = False) -> list[str]:
    values = _list(value, name)
    for item in values:
        _string(item, name, nonempty=nonempty)
    if unique and len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _keys(value: dict, expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields must be exactly {', '.join(sorted(expected))}")


def _rate(numerator: int | float, denominator: int, *, zero: float | None = None) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else zero,
    }


def _review(value: Any) -> None:
    if value is None:
        return
    value = _object(value, "review")
    _keys(value, _REVIEW_FIELDS, "review")
    _string(value["reviewer"], "review.reviewer", nonempty=True)
    _string(value["evidence_notes"], "review.evidence_notes", nonempty=True)
    if value["method"] not in REVIEW_METHODS:
        raise ValueError("review.method is unsupported")
    dimensions = _object(value["dimensions"], "review.dimensions")
    _keys(dimensions, set(REVIEW_DIMENSIONS), "review.dimensions")
    if any(type(item) is not bool for item in dimensions.values()):
        raise ValueError("review dimensions must be booleans")
    _strings(value["blocking_errors"], "review.blocking_errors")
    if value["reading_burden"] not in ("none", "some", "heavy"):
        raise ValueError("review.reading_burden is unsupported")


def _execution(value: Any) -> None:
    value = _object(value, "execution")
    _keys(value, {"agents", "tool_calls"}, "execution")
    _strings(value["agents"], "execution.agents", unique=True, nonempty=True)
    for call in _list(value["tool_calls"], "execution.tool_calls"):
        call = _object(call, "execution.tool_calls item")
        _keys(call, {"agent", "name", "success", "result_success"}, "execution.tool_calls item")
        _string(call["agent"], "execution.tool_calls.agent", nonempty=True)
        _string(call["name"], "execution.tool_calls.name", nonempty=True)
        if type(call["success"]) is not bool:
            raise ValueError("execution.tool_calls.success must be a boolean")
        if call["result_success"] is not None and type(call["result_success"]) is not bool:
            raise ValueError("execution.tool_calls.result_success must be a boolean or null")


def _record(value: Any, arm: str | None) -> dict:
    value = _object(value, "record")
    if arm is None:
        extra = set(value) - _COMMON_FIELDS
        matches = [name for name, fields in _ARM_FIELDS.items() if extra == fields]
        if not extra:
            expected = _COMMON_FIELDS
        elif len(matches) == 1:
            arm = matches[0]
            expected = _COMMON_FIELDS | _ARM_FIELDS[arm]
        else:
            raise ValueError("record has unsupported output fields")
    else:
        expected = _COMMON_FIELDS | _ARM_FIELDS[arm]
    _keys(value, expected, "record")
    _string(value["case_id"], "record.case_id", nonempty=True)
    if value["status"] not in STATUSES:
        raise ValueError("record.status is unsupported")
    latency = value["latency_ms"]
    if latency is not None and (
        type(latency) not in (int, float)
        or latency < 0
        or (isinstance(latency, float) and not math.isfinite(latency))
    ):
        raise ValueError("record.latency_ms must be a nonnegative finite number or null")
    usage = _object(value["usage"], "record.usage")
    _keys(usage, set(USAGE_FIELDS), "record.usage")
    for key in USAGE_FIELDS:
        number = usage[key]
        if number is not None and (type(number) is not int or number < 0):
            raise ValueError(f"record.usage.{key} must be a nonnegative integer or null")
    if arm == "intent":
        if value["predicted_intent"] is not None:
            _string(value["predicted_intent"], "record.predicted_intent")
    elif arm == "retrieval":
        _strings(value["ranked_titles"], "record.ranked_titles", unique=True)
    elif arm == "tasks":
        _string(value["answer"], "record.answer")
        if value["status"] == "success" and not value["answer"].strip():
            raise ValueError("successful task record.answer must be nonempty")
        _execution(value["execution"])
        _review(value["review"])
    return value


def _inputs(cases: Any, records: Any, arm: str) -> tuple[list[dict], dict[str, dict]]:
    cases = _list(cases, "cases")
    records = _list(records, "records")
    ids = set()
    for case in cases:
        case = _object(case, "case")
        case_id = _string(case.get("id"), "case.id", nonempty=True)
        if case_id in ids:
            raise ValueError("duplicate case id")
        ids.add(case_id)
        expected = _object(case.get("expected"), "case.expected")
        if arm == "intent":
            _string(expected.get("intent"), "case.expected.intent", nonempty=True)
        elif arm == "retrieval":
            if expected.get("kind") not in ("answerable", "insufficient", "out_of_scope"):
                raise ValueError("case.expected.kind is unsupported")
            titles = _strings(expected.get("relevant_titles"), "case.expected.relevant_titles", unique=True)
            if (expected["kind"] == "out_of_scope") != (not titles):
                raise ValueError("evidence cases need relevant titles; out_of_scope cases must have none")
        elif arm == "tasks":
            for field in ("required_agents", "required_tools"):
                _strings(expected.get(field, []), f"case.expected.{field}", unique=True, nonempty=True)
    indexed = {}
    for raw in records:
        record = _record(raw, arm)
        case_id = record["case_id"]
        if case_id not in ids:
            raise ValueError("unknown case_id")
        if case_id in indexed:
            raise ValueError("duplicate record case_id")
        indexed[case_id] = record
    return cases, indexed


def _resources(records: list[dict]) -> dict:
    latencies = sorted(record["latency_ms"] for record in records if record["latency_ms"] is not None)

    def percentile(fraction: float) -> int | float | None:
        return latencies[math.ceil(fraction * len(latencies)) - 1] if latencies else None

    usage = {}
    for field in USAGE_FIELDS:
        values = [record["usage"][field] for record in records if record["usage"][field] is not None]
        usage[field] = {
            "known_sum": sum(values) if values else None,
            "known_count": len(values),
            "unknown_count": len(records) - len(values),
        }
    return {
        "scope": "recorded_records_only; not_executed cases have no inferred resources",
        "recorded_total": len(records),
        "latency_ms": {
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "known_count": len(latencies),
            "unknown_count": len(records) - len(latencies),
            "method": "nearest-rank",
            "population": "recorded non-null latency, including recorded failures; no inferred timeouts",
        },
        "usage": usage,
    }


def aggregate_resources(records: list[dict]) -> dict:
    """Aggregate common resource fields; reject malformed or duplicate records.

    Records may have only the four common fields, or the exact output fields for
    one supported arm. Unknown resource values remain unknown; an entirely
    unknown field has a null known_sum rather than a fabricated zero.
    """
    records = [_record(record, None) for record in _list(records, "records")]
    ids = [record["case_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record case_id")
    return _resources(records)


def _summary(cases: list[dict], indexed: dict[str, dict]) -> dict:
    return {
        "planned_total": len(cases),
        "recorded_total": len(indexed),
        "not_executed": len(cases) - len(indexed),
        "status_counts": {status: sum(record["status"] == status for record in indexed.values()) for status in STATUSES},
        "resources": _resources(list(indexed.values())),
    }


def score_intent(cases: list[dict], records: list[dict]) -> dict:
    """Score successful predictions against the fixed expected-label population.

    Non-success predictions are retained in per_case but cannot create true or
    false positives. They create a false negative for their expected label.
    out_of_label_predictions counts all recorded non-null out-of-label values.
    """
    cases, indexed = _inputs(cases, records, "intent")
    labels = sorted({case["expected"]["intent"] for case in cases})
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}
    per_case = []
    for case in cases:
        record = indexed.get(case["id"])
        expected = case["expected"]["intent"]
        prediction = record["predicted_intent"] if record else None
        valid = bool(record and record["status"] == "success")
        correct = valid and prediction == expected
        counts[expected]["support"] += 1
        counts[expected]["tp" if correct else "fn"] += 1
        if valid and prediction in counts and prediction != expected:
            counts[prediction]["fp"] += 1
        per_case.append({
            "case_id": case["id"],
            "status": record["status"] if record else "not_executed",
            "expected_intent": expected,
            "predicted_intent": prediction,
            "correct": correct,
            "out_of_label_prediction": prediction is not None and prediction not in counts,
        })
    per_label = {}
    for label, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        per_label[label] = {
            **values,
            "precision": _rate(tp, tp + fp, zero=0.0),
            "recall": _rate(tp, tp + fn, zero=0.0),
            "f1": _rate(2 * tp, 2 * tp + fp + fn, zero=0.0),
        }
    correct_total = sum(item["correct"] for item in per_case)
    return {
        **_summary(cases, indexed),
        "labels": labels,
        "accuracy": _rate(correct_total, len(cases)),
        "macro_f1": sum(item["f1"]["value"] for item in per_label.values()) / len(labels) if labels else None,
        "classification_zero_division": 0,
        "per_label": per_label,
        "out_of_label_predictions": sum(item["out_of_label_prediction"] for item in per_case),
        "per_case": per_case,
    }


def _retrieval_rates(items: list[dict]) -> dict:
    return {
        "planned_total": len(items),
        "hit_at_1": _rate(sum(item["hit_at_1"] for item in items), len(items)),
        "hit_at_3": _rate(sum(item["hit_at_3"] for item in items), len(items)),
        "all_relevant_at_3": _rate(sum(item["all_relevant_at_3"] for item in items), len(items)),
        "document_recall_at_3": _rate(sum(item["document_recall_at_3"] for item in items), len(items)),
        "mrr": _rate(sum(item["reciprocal_rank"] for item in items), len(items)),
        "not_executed": sum(item["status"] == "not_executed" for item in items),
        "status_counts": {status: sum(item["status"] == status for item in items) for status in STATUSES},
    }


def score_retrieval(cases: list[dict], records: list[dict]) -> dict:
    """Score evidence retrieval separately from refusal on out-of-scope cases.

    An insufficient case still has relevant evidence, but retrieving it does not
    establish answerability. mis_hit counts any observed nonempty title list,
    including a partial list on a failed record; only successful empty results
    count as correct_rejection. Titles are compared exactly, without deduping.
    """
    cases, indexed = _inputs(cases, records, "retrieval")
    per_case = []
    for case in cases:
        record = indexed.get(case["id"])
        titles = list(record["ranked_titles"]) if record else []
        kind = case["expected"]["kind"]
        successful = bool(record and record["status"] == "success")
        relevant = set(case["expected"]["relevant_titles"])
        rank = next((index for index, title in enumerate(titles, 1) if title in relevant), None) if successful else None
        no_evidence = kind == "out_of_scope"
        retrieved_relevant = len(relevant.intersection(titles[:3])) if successful else 0
        per_case.append({
            "case_id": case["id"],
            "status": record["status"] if record else "not_executed",
            "kind": kind,
            "ranked_titles": titles,
            "first_relevant_rank": rank,
            "hit_at_1": rank == 1 if not no_evidence else None,
            "hit_at_3": rank is not None and rank <= 3 if not no_evidence else None,
            "all_relevant_at_3": retrieved_relevant == len(relevant) if not no_evidence else None,
            "document_recall_at_3": retrieved_relevant / len(relevant) if not no_evidence else None,
            "relevant_count": len(relevant),
            "retrieved_relevant_at_3": retrieved_relevant if not no_evidence else None,
            "reciprocal_rank": 1 / rank if rank else 0.0 if not no_evidence else None,
            "mis_hit": bool(titles) if no_evidence else None,
            "correct_rejection": successful and not titles if no_evidence else None,
        })
    evidence = [item for item in per_case if item["kind"] != "out_of_scope"]
    no_evidence = [item for item in per_case if item["kind"] == "out_of_scope"]
    return {
        **_summary(cases, indexed),
        "evidence": _retrieval_rates(evidence),
        "by_kind": {kind: _retrieval_rates([item for item in evidence if item["kind"] == kind]) for kind in ("answerable", "insufficient")},
        "insufficient_total": sum(item["kind"] == "insufficient" for item in per_case),
        "insufficient_note": "retrieval hit does not establish that an insufficient question is answerable",
        "no_evidence": {
            "planned_total": len(no_evidence),
            "mis_hit": _rate(sum(item["mis_hit"] for item in no_evidence), len(no_evidence)),
            "correct_rejection": _rate(sum(item["correct_rejection"] for item in no_evidence), len(no_evidence)),
            "not_executed": sum(item["status"] == "not_executed" for item in no_evidence),
            "status_counts": {status: sum(item["status"] == status for item in no_evidence) for status in STATUSES},
        },
        "per_case": per_case,
    }


def score_tasks(cases: list[dict], records: list[dict]) -> dict:
    """Score explicit rubric reviews; answer keywords never produce a pass.

    A non-success execution or a missing required agent/tool is failed even if
    its review is absent. A successful execution meeting these deterministic
    checks without review is pending. Each dimension uses all planned cases as
    its denominator, and needs a successful execution and a true review.
    Passing also needs observed required agents and successful required tool
    calls. Tool names do not establish permissions; the explicit tool_permissions
    rubric dimension still needs a separate review of each calling agent.
    Evidence-kind restrictions on controlled-fixture reviews belong to the
    caller, which owns the record envelope and its evidence classification.
    """
    cases, indexed = _inputs(cases, records, "tasks")
    outcomes = Counter({key: 0 for key in ("passed", "failed", "pending", "not_executed")})
    dimensions = Counter({key: 0 for key in REVIEW_DIMENSIONS})
    blockers = Counter()
    reading_burden = Counter({key: 0 for key in ("none", "some", "heavy")})
    review_methods = Counter({key: 0 for key in REVIEW_METHODS})
    per_case = []
    for case in cases:
        record = indexed.get(case["id"])
        review = record["review"] if record else None
        execution = record["execution"] if record else {"agents": [], "tool_calls": []}
        observed_tools = {
            call["name"] for call in execution["tool_calls"]
            if call["success"] and call["result_success"] is not False
        }
        deterministic_checks = {
            "required_agents": set(case["expected"].get("required_agents", [])).issubset(execution["agents"]),
            "required_tools": set(case["expected"].get("required_tools", [])).issubset(observed_tools),
        }
        successful = bool(record and record["status"] == "success")
        if record is None:
            outcome = "not_executed"
        elif not successful:
            outcome = "failed"
        elif not all(deterministic_checks.values()):
            outcome = "failed"
        elif review is None:
            outcome = "pending"
        elif all(review["dimensions"].values()) and not review["blocking_errors"]:
            outcome = "passed"
        else:
            outcome = "failed"
        outcomes[outcome] += 1
        if review is not None:
            review_methods[review["method"]] += 1
            reading_burden[review["reading_burden"]] += 1
            blockers.update(review["blocking_errors"])
            for dimension in REVIEW_DIMENSIONS:
                dimensions[dimension] += successful and review["dimensions"][dimension]
        per_case.append({
            "case_id": case["id"],
            "status": record["status"] if record else "not_executed",
            "outcome": outcome,
            "answer": record["answer"] if record else None,
            "execution": {
                "agents": list(execution["agents"]),
                "tool_calls": [dict(call) for call in execution["tool_calls"]],
            } if record else None,
            "deterministic_checks": deterministic_checks,
            "review": {
                **review,
                "dimensions": dict(review["dimensions"]),
                "blocking_errors": list(review["blocking_errors"]),
            } if review else None,
        })
    return {
        **_summary(cases, indexed),
        "outcome_counts": dict(outcomes),
        "strict_pass": _rate(outcomes["passed"], len(cases)),
        "dimensions": {key: _rate(dimensions[key], len(cases)) for key in REVIEW_DIMENSIONS},
        "dimension_population": "all planned cases; numerator requires success and a true explicit review",
        "reviewed_total": sum(review_methods.values()),
        "review_method_counts": dict(review_methods),
        "reading_burden_counts": dict(reading_burden),
        "blocking_error_total": sum(blockers.values()),
        "blocking_error_counts": dict(sorted(blockers.items())),
        "per_case": per_case,
    }
