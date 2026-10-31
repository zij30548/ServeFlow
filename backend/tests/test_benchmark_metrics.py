from copy import deepcopy

import pytest

from evaluation.benchmark_metrics import (
    REVIEW_DIMENSIONS,
    aggregate_resources,
    score_intent,
    score_retrieval,
    score_tasks,
)


def intent_case(case_id, label="GENERAL"):
    return {"id": case_id, "expected": {"intent": label}}


def retrieval_case(case_id, kind="answerable", titles=None):
    return {
        "id": case_id,
        "expected": {
            "kind": kind,
            "relevant_titles": titles if titles is not None else ([] if kind == "out_of_scope" else ["政策A"]),
        },
    }


def task_case(case_id):
    return {
        "id": case_id,
        "expected": {
            "required_agents": ["billing"],
            "required_tools": ["lookup_order"],
            "required_facts": ["核对实际退款状态", "说明到账条件"],
            "forbidden_claims": ["已执行退款"],
            "rationale": "应核对业务记录并保留政策条件",
        },
    }


def record(case_id, arm="intent", status="success", **changes):
    value = {
        "case_id": case_id,
        "status": status,
        "latency_ms": None,
        "usage": {"input_tokens": None, "output_tokens": None, "physical_requests": None},
    }
    if arm == "intent":
        value["predicted_intent"] = "GENERAL"
    elif arm == "retrieval":
        value["ranked_titles"] = ["政策A"]
    elif arm == "tasks":
        value.update(
            answer="核对了记录并说明条件",
            review=review(),
            execution={
                "agents": ["billing"],
                "tool_calls": [{"agent": "billing", "name": "lookup_order", "success": True, "result_success": True}],
            },
        )
    value.update(changes)
    return value


def review(**changes):
    value = {
        "reviewer": "fixture-reviewer",
        "method": "controlled-fixture",
        "evidence_notes": "受控样例逐项核对，不是模型实测结果",
        "dimensions": dict.fromkeys(REVIEW_DIMENSIONS, True),
        "blocking_errors": [],
        "reading_burden": "none",
    }
    value.update(changes)
    return value


def assert_rate(actual, numerator, denominator):
    assert actual["numerator"] == pytest.approx(numerator)
    assert actual["denominator"] == denominator
    if denominator:
        assert actual["value"] == pytest.approx(numerator / denominator)
    else:
        assert actual["value"] is None


def test_intent_failure_missing_and_unknown_keep_fixed_labels_and_denominators():
    cases = [intent_case(f"i{index}", label) for index, label in enumerate(["A", "B", "A", "B", "A", "B"])]
    records = [
        record("i0", predicted_intent="A"),
        record("i1", predicted_intent="A"),
        record("i2", status="failed", predicted_intent="A"),
        record("i3", status="unknown", predicted_intent="B"),
        record("i5", predicted_intent="NEW_LABEL"),
    ]
    result = score_intent(cases, records)
    assert_rate(result["accuracy"], 1, 6)
    assert result["planned_total"] == 6 and result["recorded_total"] == 5
    assert result["not_executed"] == 1
    assert result["status_counts"] == {"success": 3, "failed": 1, "cancelled": 0, "unknown": 1}
    assert result["labels"] == ["A", "B"]
    assert result["out_of_label_predictions"] == 1
    assert result["per_label"]["A"]["tp"] == 1
    assert result["per_label"]["A"]["fp"] == 1
    assert result["per_label"]["A"]["fn"] == 2
    assert_rate(result["per_label"]["A"]["precision"], 1, 2)
    assert_rate(result["per_label"]["A"]["recall"], 1, 3)
    assert result["per_label"]["A"]["f1"]["value"] == pytest.approx(0.4)
    assert result["per_label"]["B"]["precision"]["value"] == 0
    assert result["macro_f1"] == pytest.approx(0.2)
    assert result["per_case"][2]["predicted_intent"] == "A"
    assert result["per_case"][2]["correct"] is False
    assert result["per_case"][3]["status"] == "unknown"
    assert result["per_case"][4]["status"] == "not_executed"


def test_intent_null_and_missing_are_never_fabricated_as_other_predictions():
    cases = [intent_case("null", "OTHER"), intent_case("missing", "OTHER")]
    result = score_intent(cases, [record("null", predicted_intent=None)])
    assert_rate(result["accuracy"], 0, 2)
    assert result["per_label"]["OTHER"]["fn"] == 2
    assert all(item["predicted_intent"] is None for item in result["per_case"])


def test_retrieval_ranks_and_insufficient_cases_are_not_answer_quality():
    cases = [
        retrieval_case("rank2"), retrieval_case("rank4"),
        retrieval_case("insufficient", "insufficient"),
        retrieval_case("failed"), retrieval_case("missing"),
    ]
    records = [
        record("rank2", "retrieval", ranked_titles=["其他", "政策A"]),
        record("rank4", "retrieval", ranked_titles=["其他1", "其他2", "其他3", "政策A"]),
        record("insufficient", "retrieval"),
        record("failed", "retrieval", status="failed"),
    ]
    result = score_retrieval(cases, records)
    assert_rate(result["evidence"]["hit_at_1"], 1, 5)
    assert_rate(result["evidence"]["hit_at_3"], 2, 5)
    assert_rate(result["evidence"]["mrr"], 1.75, 5)
    assert result["per_case"][0]["first_relevant_rank"] == 2
    assert result["per_case"][1]["first_relevant_rank"] == 4
    assert result["per_case"][3]["first_relevant_rank"] is None
    assert result["per_case"][3]["reciprocal_rank"] == 0
    assert result["insufficient_total"] == 1
    assert_rate(result["by_kind"]["insufficient"]["hit_at_1"], 1, 1)
    assert "does not establish" in result["insufficient_note"]


def test_out_of_scope_false_hits_do_not_hide_failures_or_omitted_execution():
    cases = [retrieval_case(case_id, "out_of_scope") for case_id in ("empty", "hit", "failed", "failed_hit", "unknown", "missing")]
    records = [
        record("empty", "retrieval", ranked_titles=[]),
        record("hit", "retrieval", ranked_titles=["无关政策"]),
        record("failed", "retrieval", status="failed", ranked_titles=[]),
        record("failed_hit", "retrieval", status="failed", ranked_titles=["无关政策"]),
        record("unknown", "retrieval", status="unknown", ranked_titles=[]),
    ]
    result = score_retrieval(cases, records)
    no_evidence = result["no_evidence"]
    assert_rate(no_evidence["mis_hit"], 2, 6)
    assert_rate(no_evidence["correct_rejection"], 1, 6)
    assert no_evidence["not_executed"] == 1
    assert no_evidence["status_counts"]["failed"] == 2
    assert no_evidence["status_counts"]["unknown"] == 1
    assert result["per_case"][-1]["correct_rejection"] is False
    assert result["per_case"][-1]["reciprocal_rank"] is None
    assert_rate(result["evidence"]["mrr"], 0, 0)


def test_multi_source_retrieval_distinguishes_any_hit_from_complete_coverage():
    cases = [
        retrieval_case("partial", titles=["政策A", "政策B"]),
        retrieval_case("complete", titles=["政策A", "政策B"]),
        retrieval_case("failed", titles=["政策A", "政策B"]),
        retrieval_case("missing", titles=["政策A", "政策B"]),
        retrieval_case("outside", "out_of_scope"),
    ]
    records = [
        record("partial", "retrieval", ranked_titles=["政策A", "其他1", "其他2", "政策B"]),
        record("complete", "retrieval", ranked_titles=["政策B", "其他1", "政策A"]),
        record("failed", "retrieval", status="failed", ranked_titles=["政策A", "政策B"]),
        record("outside", "retrieval", ranked_titles=[]),
    ]
    result = score_retrieval(cases, records)
    assert_rate(result["evidence"]["hit_at_3"], 2, 4)
    assert_rate(result["evidence"]["all_relevant_at_3"], 1, 4)
    assert_rate(result["evidence"]["document_recall_at_3"], 1.5, 4)
    assert result["per_case"][0]["document_recall_at_3"] == 0.5
    assert result["per_case"][0]["all_relevant_at_3"] is False
    assert result["per_case"][1]["retrieved_relevant_at_3"] == 2
    assert result["per_case"][2]["document_recall_at_3"] == 0
    assert result["per_case"][3]["document_recall_at_3"] == 0
    assert result["per_case"][4]["document_recall_at_3"] is None
    assert result["per_case"][4]["all_relevant_at_3"] is None


def test_task_requires_explicit_review_and_all_dimensions_and_no_blockers():
    cases = [task_case(case_id) for case_id in ("pass", "block", "wrong", "pending", "failed", "unknown", "missing")]
    wrong_dimensions = dict.fromkeys(REVIEW_DIMENSIONS, True)
    wrong_dimensions.update(evidence_relevance=False, coverage=False)
    records = [
        record("pass", "tasks"),
        record("block", "tasks", review=review(blocking_errors=["声称执行了退款"])),
        # Legal JSON and complete structure still fail when the reviewer finds
        # the wrong policy was selected and part of the question unanswered.
        record("wrong", "tasks", answer="引用了无关政策，遗漏到账条件", review=review(
            dimensions=wrong_dimensions, blocking_errors=["资料不适用", "漏答到账条件"], reading_burden="heavy")),
        record("pending", "tasks", answer="billing lookup_order 核对实际退款状态 说明到账条件", review=None),
        record("failed", "tasks", status="failed"),
        record("unknown", "tasks", status="unknown", review=None),
    ]
    result = score_tasks(cases, records)
    assert result["outcome_counts"] == {"passed": 1, "failed": 4, "pending": 1, "not_executed": 1}
    assert_rate(result["strict_pass"], 1, 7)
    assert_rate(result["dimensions"]["evidence_relevance"], 2, 7)
    assert_rate(result["dimensions"]["coverage"], 2, 7)
    assert_rate(result["dimensions"]["tool_permissions"], 3, 7)
    assert result["reviewed_total"] == 4
    assert result["blocking_error_total"] == 3
    assert result["blocking_error_counts"] == {"声称执行了退款": 1, "漏答到账条件": 1, "资料不适用": 1}
    assert result["reading_burden_counts"] == {"none": 3, "some": 0, "heavy": 1}
    assert result["review_method_counts"]["controlled-fixture"] == 4
    assert result["per_case"][3]["outcome"] == "pending"
    assert result["per_case"][4]["outcome"] == "failed"
    assert result["per_case"][5]["status"] == "unknown"


@pytest.mark.parametrize("status", ["failed", "cancelled", "unknown"])
def test_task_execution_failure_without_review_is_failed(status):
    result = score_tasks([task_case("case")], [record("case", "tasks", status=status, review=None)])
    assert result["outcome_counts"]["failed"] == 1
    assert result["outcome_counts"]["pending"] == 0
    assert_rate(result["strict_pass"], 0, 1)


def test_plausible_answer_and_positive_review_cannot_replace_required_tool_execution():
    result = score_tasks([task_case("case")], [record(
        "case", "tasks", answer="订单已核对，退款实际状态及到账条件如下。",
        execution={"agents": ["billing"], "tool_calls": []},
    )])
    assert result["per_case"][0]["deterministic_checks"] == {"required_agents": True, "required_tools": False}
    assert result["outcome_counts"]["failed"] == 1
    assert_rate(result["strict_pass"], 0, 1)


def test_missing_required_agent_fails_even_with_required_tool_and_positive_review():
    result = score_tasks([task_case("case")], [record(
        "case", "tasks", execution={
            "agents": ["general"],
            "tool_calls": [{"agent": "general", "name": "lookup_order", "success": True, "result_success": True}],
        },
    )])
    assert result["per_case"][0]["deterministic_checks"] == {"required_agents": False, "required_tools": True}
    assert result["outcome_counts"]["failed"] == 1


@pytest.mark.parametrize("agents,tool_calls", [
    (["billing"], []),
    (["general"], [{"agent": "general", "name": "lookup_order", "success": True, "result_success": True}]),
])
def test_deterministic_failure_is_failed_before_pending_review(agents, tool_calls):
    result = score_tasks([task_case("case")], [record(
        "case", "tasks", review=None, execution={"agents": agents, "tool_calls": tool_calls},
    )])
    assert result["outcome_counts"]["failed"] == 1
    assert result["outcome_counts"]["pending"] == 0
    assert result["per_case"][0]["review"] is None


@pytest.mark.parametrize("success,result_success,expected", [
    (True, True, True),
    (True, None, True),
    (True, False, False),
    (False, True, False),
    (False, None, False),
    (False, False, False),
])
def test_required_tools_need_successful_execution_without_explicit_failed_result(success, result_success, expected):
    result = score_tasks([task_case("case")], [record(
        "case", "tasks", execution={
            "agents": ["billing"],
            "tool_calls": [{"agent": "billing", "name": "lookup_order", "success": success, "result_success": result_success}],
        },
    )])
    assert result["per_case"][0]["deterministic_checks"]["required_tools"] is expected
    assert result["outcome_counts"]["passed"] == int(expected)


def test_tool_name_does_not_replace_explicit_permission_review():
    dimensions = dict.fromkeys(REVIEW_DIMENSIONS, True)
    dimensions["tool_permissions"] = False
    result = score_tasks([task_case("case")], [record("case", "tasks", review=review(dimensions=dimensions))])
    assert all(result["per_case"][0]["deterministic_checks"].values())
    assert result["outcome_counts"]["failed"] == 1


@pytest.mark.parametrize("expected", [
    {},
    {"required_agents": [], "required_tools": []},
    {"required_agents": ["escalation"], "required_tools": []},
])
def test_task_without_required_tools_can_pass_with_no_tool_calls(expected):
    result = score_tasks([{"id": "case", "expected": expected}], [record(
        "case", "tasks", execution={"agents": ["escalation"], "tool_calls": []},
    )])
    assert result["per_case"][0]["deterministic_checks"] == {"required_agents": True, "required_tools": True}
    assert result["outcome_counts"]["passed"] == 1


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("execution"),
    lambda value: value.update(execution=None),
    lambda value: value["execution"].pop("agents"),
    lambda value: value["execution"].pop("tool_calls"),
    lambda value: value["execution"].update(raw_credentials="forbidden"),
    lambda value: value["execution"].update(agents="billing"),
    lambda value: value["execution"].update(agents=["billing", "billing"]),
    lambda value: value["execution"].update(agents=[""]),
    lambda value: value["execution"].update(agents=[" \t"]),
    lambda value: value["execution"].update(agents=[True]),
    lambda value: value["execution"].update(tool_calls={}),
    lambda value: value["execution"].update(tool_calls=[None]),
    lambda value: value["execution"]["tool_calls"][0].pop("result_success"),
    lambda value: value["execution"]["tool_calls"][0].update(arguments={"customer_token": "forbidden"}),
    lambda value: value["execution"]["tool_calls"][0].update(agent=""),
    lambda value: value["execution"]["tool_calls"][0].update(name="\n"),
    lambda value: value["execution"]["tool_calls"][0].update(success=1),
    lambda value: value["execution"]["tool_calls"][0].update(result_success=0),
])
def test_task_execution_observation_contract_rejects_malformed_or_extra_fields(mutation):
    value = record("case", "tasks")
    mutation(value)
    with pytest.raises(ValueError):
        score_tasks([task_case("case")], [value])


@pytest.mark.parametrize("answer", ["", " ", "\n\t"])
def test_successful_task_must_have_an_answer_even_when_review_says_pass(answer):
    with pytest.raises(ValueError, match="answer must be nonempty"):
        score_tasks([task_case("case")], [record("case", "tasks", answer=answer)])
    result = score_tasks([task_case("case")], [record("case", "tasks", status="failed", answer=answer, review=None)])
    assert result["outcome_counts"]["failed"] == 1


@pytest.mark.parametrize("scorer,metric", [(score_intent, "accuracy"), (score_tasks, "strict_pass")])
def test_zero_population_has_no_claimed_quality(scorer, metric):
    result = scorer([], [])
    assert_rate(result[metric], 0, 0)
    assert result["planned_total"] == result["recorded_total"] == result["not_executed"] == 0
    assert result["per_case"] == []
    assert result["resources"]["latency_ms"]["p50"] is None
    if scorer is score_intent:
        assert result["macro_f1"] is None and result["labels"] == []
    else:
        assert all(item["value"] is None for item in result["dimensions"].values())


def test_empty_retrieval_populations_are_explicit():
    result = score_retrieval([], [])
    assert_rate(result["evidence"]["hit_at_1"], 0, 0)
    assert_rate(result["no_evidence"]["mis_hit"], 0, 0)
    assert_rate(result["by_kind"]["insufficient"]["mrr"], 0, 0)
    assert_rate(result["evidence"]["all_relevant_at_3"], 0, 0)
    assert_rate(result["evidence"]["document_recall_at_3"], 0, 0)


@pytest.mark.parametrize("scorer,case,arm", [
    (score_intent, intent_case("case"), "intent"),
    (score_retrieval, retrieval_case("case"), "retrieval"),
    (score_tasks, task_case("case"), "tasks"),
])
def test_all_scorers_reject_unknown_and_duplicate_ids(scorer, case, arm):
    with pytest.raises(ValueError, match="unknown case_id"):
        scorer([case], [record("different", arm)])
    with pytest.raises(ValueError, match="duplicate record"):
        scorer([case], [record("case", arm), record("case", arm)])
    with pytest.raises(ValueError, match="duplicate case"):
        scorer([case, case], [])


@pytest.mark.parametrize("latency", [True, False, -1, float("nan"), float("inf"), -float("inf"), "12", []])
def test_nonfinite_negative_boolean_or_non_number_latencies_are_rejected(latency):
    with pytest.raises(ValueError, match="latency_ms"):
        score_intent([intent_case("case")], [record("case", latency_ms=latency)])


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "physical_requests"])
@pytest.mark.parametrize("amount", [True, False, -1, 1.5, float("nan"), float("inf"), "1"])
def test_usage_requires_nonnegative_integer_or_explicit_unknown(field, amount):
    value = record("case")
    value["usage"][field] = amount
    with pytest.raises(ValueError, match=field):
        score_intent([intent_case("case")], [value])


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("latency_ms"),
    lambda value: value.update(extra="unrecognized"),
    lambda value: value["usage"].pop("physical_requests"),
    lambda value: value["usage"].update(total_tokens=10),
    lambda value: value.update(status="timeout"),
    lambda value: value.update(predicted_intent=1),
    lambda value: value.update(case_id=True),
])
def test_record_contract_rejects_missing_extra_or_malformed_fields(mutation):
    value = record("case")
    mutation(value)
    with pytest.raises(ValueError):
        score_intent([intent_case("case")], [value])


@pytest.mark.parametrize("titles", [["政策A", "政策A"], "政策A", [1]])
def test_retrieval_does_not_silently_repair_invalid_title_lists(titles):
    with pytest.raises(ValueError):
        score_retrieval([retrieval_case("case")], [record("case", "retrieval", ranked_titles=titles)])


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("reviewer"),
    lambda value: value.update(extra=True),
    lambda value: value.update(reviewer=None),
    lambda value: value.update(reviewer=""),
    lambda value: value.update(reviewer=" \n"),
    lambda value: value.update(evidence_notes=1),
    lambda value: value.update(evidence_notes=""),
    lambda value: value.update(evidence_notes="\t"),
    lambda value: value.update(method="automatic-keyword-pass"),
    lambda value: value.update(reading_burden="low"),
    lambda value: value.update(blocking_errors="error"),
    lambda value: value["dimensions"].pop("coverage"),
    lambda value: value["dimensions"].update(extra=True),
    lambda value: value["dimensions"].update(coverage=1),
])
def test_task_review_contract_is_strict(mutation):
    value = review()
    mutation(value)
    with pytest.raises(ValueError):
        score_tasks([task_case("case")], [record("case", "tasks", review=value)])


def test_resources_use_recorded_nearest_rank_latencies_and_partial_known_usage():
    values = []
    for index, latency in enumerate([50, 10, 20, 40, 30, None]):
        value = record(str(index), status="failed" if index == 0 else "success", latency_ms=latency)
        value["usage"] = {
            "input_tokens": 10 if index < 2 else None,
            "output_tokens": None,
            "physical_requests": 0 if index == 0 else None,
        }
        values.append(value)
    resources = aggregate_resources(values)
    assert resources["latency_ms"]["p50"] == 30
    assert resources["latency_ms"]["p95"] == 50
    assert resources["latency_ms"]["known_count"] == 5
    assert resources["latency_ms"]["unknown_count"] == 1
    assert resources["latency_ms"]["method"] == "nearest-rank"
    assert "no inferred timeouts" in resources["latency_ms"]["population"]
    assert resources["usage"]["input_tokens"] == {"known_sum": 20, "known_count": 2, "unknown_count": 4}
    assert resources["usage"]["output_tokens"] == {"known_sum": None, "known_count": 0, "unknown_count": 6}
    assert resources["usage"]["physical_requests"] == {"known_sum": 0, "known_count": 1, "unknown_count": 5}
    result = score_intent([intent_case(str(index)) for index in range(7)], values)
    assert result["not_executed"] == 1
    assert result["resources"] == resources


def test_aggregate_resource_records_can_be_common_only_and_do_not_fabricate_empty_totals():
    value = record("case")
    del value["predicted_intent"]
    assert aggregate_resources([value])["usage"]["input_tokens"]["unknown_count"] == 1
    empty = aggregate_resources([])
    assert empty["latency_ms"]["p95"] is None
    assert empty["usage"]["input_tokens"]["known_sum"] is None
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_resources([value, value])


def test_unknown_id_validation_does_not_echo_untrusted_input():
    private_id = "private\nuser-supplied-sensitive-identifier"
    with pytest.raises(ValueError) as caught:
        score_intent([intent_case("case")], [record(private_id)])
    assert str(caught.value) == "unknown case_id"
    assert private_id not in str(caught.value)


@pytest.mark.parametrize("scorer,cases,records", [
    (score_intent, [intent_case("case")], [record("case")]),
    (score_retrieval, [retrieval_case("case")], [record("case", "retrieval")]),
    (score_tasks, [task_case("case")], [record("case", "tasks")]),
])
def test_scorers_do_not_mutate_inputs(scorer, cases, records):
    before = deepcopy((cases, records))
    scorer(cases, records)
    assert (cases, records) == before
